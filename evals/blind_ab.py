"""Слепой тест моделей на реальном пайплайне бота.

Для каждого запроса из cases бот-пайплайн (твой system prompt + канон из pgvector)
генерирует сцену тремя моделями. Факты подбираются ОДИН раз на запрос, чтобы
различия давала только модель. Модель-судья проверяет каждую сцену на
противоречия канону, длина считается автоматически. Тексты тасуются под буквами
A/B/C и уходят тебе в Telegram, соответствие букв моделям лежит в key.json.

Запуск (внутри контейнера бота, там есть .env, промпт и доступ к БД):
    python evals/blind_ab.py --user-id <твой_id>
Оценка: python evals/blind_score.py <папка_прогона> 1BAC 2CAB ...
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import claude_tg_bot as bot  # noqa: E402

CASES_LOCAL = ROOT / "evals" / "cases.local.jsonl"  # приватные, в гите не лежат
CASES_EXAMPLE = ROOT / "evals" / "cases.example.jsonl"
RUNS_DIR = ROOT / "eval_runs"
ALIASES = ["haiku", "sonnet", "opus"]
LETTERS = "ABC"
JUDGE_ALIAS = "sonnet"

JUDGE_SYSTEM = (
    "Ты проверяешь художественный текст на непротиворечивость списку фактов-канона. "
    "Верни ТОЛЬКО JSON без пояснений вида "
    '{"contradictions": [{"fact": "...", "quote": "цитата до 12 слов", "why": "..."}], '
    '"facts_used": <сколько фактов из списка отражено в тексте>}. '
    "Противоречие — только прямое нарушение факта (например, другой цвет волос или "
    "привычка), а не отсутствие факта в тексте и не выдуманные детали, которым канон "
    "не противоречит. Если противоречий нет, верни пустой список."
)


def count_words(text: str) -> int:
    return len(re.findall(r"\w+", text))


def load_cases(path: Path | None) -> list[dict]:
    path = path or (CASES_LOCAL if CASES_LOCAL.exists() else CASES_EXAMPLE)
    cases = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"кейсов: {len(cases)} из {path.name}")
    return cases


def generate(alias: str, base_prompt: str, canon: str, prompt: str) -> dict:
    """Один прогон через те же функции, что в боте (system + канон в хвосте)."""
    model = bot.MODELS[alias]
    history = [{"role": "user", "content": prompt}]
    t0 = time.time()
    with bot.client.messages.stream(
        model=model,
        max_tokens=bot.MAX_TOKENS,
        system=bot.system_block(base_prompt),
        messages=bot.build_messages(history, canon),
    ) as stream:
        final = stream.get_final_message()
    text = bot.response_text(final).strip()
    usage = bot.usage_bucket(final.usage)
    return {
        "alias": alias,
        "model": model,
        "text": text,
        "words": count_words(text),
        "usage": usage,
        "cost": bot.cost_for_model(model, usage),
        "stop_reason": getattr(final, "stop_reason", None),
        "seconds": round(time.time() - t0, 1),
    }


def judge(facts: list[str], text: str) -> dict:
    """Судья видит факты и текст, но не знает, какая модель писала."""
    if not text:
        return {"judge_failed": True, "raw": "пустой текст"}
    user = "ФАКТЫ:\n" + "\n".join(f"- {f}" for f in facts) + f"\n\nТЕКСТ:\n{text}"
    try:
        resp = bot.client.messages.create(
            model=bot.MODELS[JUDGE_ALIAS],
            max_tokens=2000,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user}],
        )
        raw = bot.response_text(resp)
        m = re.search(r"\{.*\}", raw, re.S)
        data = json.loads(m.group(0))
        return {
            "contradictions": data.get("contradictions", []),
            "violations": len(data.get("contradictions", [])),
            "facts_used": data.get("facts_used"),
            "judge_cost": bot.cost_for_model(
                bot.MODELS[JUDGE_ALIAS], bot.usage_bucket(resp.usage)
            ),
        }
    except Exception as e:  # noqa: BLE001 — судья не должен валить весь прогон
        return {"judge_failed": True, "raw": f"{type(e).__name__}: {e}"[:200]}


def in_bounds(case: dict, words: int):
    lo, hi = case.get("min_words"), case.get("max_words")
    if lo is None and hi is None:
        return None
    return (lo is None or words >= lo) and (hi is None or words <= hi)


def run_case(i: int, case: dict, user_id: int, seed: int, models: list[str]) -> dict:
    base_prompt = bot.load_prompt(user_id)  # промпт чата = id юзера в личке
    history = [{"role": "user", "content": case["prompt"]}]
    facts, source = bot.get_relevant_facts(user_id, history)
    canon = bot.build_canon_block(facts)
    results = {}
    for alias in models:
        r = generate(alias, base_prompt, canon, case["prompt"])
        r["in_bounds"] = in_bounds(case, r["words"])
        r["judge"] = judge(facts, r["text"])
        results[alias] = r
    order = random.Random(seed * 1000 + i).sample(models, len(models))
    key = {LETTERS[j]: alias for j, alias in enumerate(order)}
    print(f"  кейс {i + 1} ({case['id']}) готов, фактов {len(facts)} ({source})")
    return {"index": i + 1, "case": case, "facts_n": len(facts), "results": results, "key": key}


def tg_send(token: str, chat_id: int, text: str):
    req = urllib.request.Request(
        f"https://api.telegram.org/bot{token}/sendMessage",
        data=json.dumps({"chat_id": chat_id, "text": text}).encode(),
        headers={"Content-Type": "application/json"},
    )
    urllib.request.urlopen(req, timeout=20).read()
    time.sleep(0.4)


def send_pack(rows: list[dict], chat_id: int):
    n = len(rows)
    tg_send(bot.TELEGRAM_BOT_TOKEN, chat_id,
            f"Слепой тест: {n} кейсов, по 3 варианта. Читай и ранжируй от лучшего к худшему.")
    for row in rows:
        tg_send(bot.TELEGRAM_BOT_TOKEN, chat_id, f"━━ Кейс {row['index']}/{n} ━━\n{row['case']['prompt']}")
        for letter in LETTERS[: len(row["key"])]:
            text = row["results"][row["key"][letter]]["text"] or "(пустой ответ)"
            chunks = bot.split_by_paragraphs(text, 3800)
            for k, chunk in enumerate(chunks):
                head = f"— {row['index']}{letter} —\n" if k == 0 else ""
                tg_send(bot.TELEGRAM_BOT_TOKEN, chat_id, head + chunk)
    tg_send(bot.TELEGRAM_BOT_TOKEN, chat_id,
            "Готово. Ответь строкой, буквы от лучшего к худшему, например: 1BAC 2CAB 3ACB ...")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--user-id", type=int, required=True)
    ap.add_argument("--cases", type=Path)
    ap.add_argument("--models", default=",".join(ALIASES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--no-send", action="store_true", help="не слать тексты в Telegram")
    args = ap.parse_args()

    models = [m for m in args.models.split(",") if m]
    cases = load_cases(args.cases)
    run_dir = RUNS_DIR / datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir.mkdir(parents=True)

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(run_case, i, c, args.user_id, args.seed, models) for i, c in enumerate(cases)]
        rows = [f.result() for f in futures]

    run = {"seed": args.seed, "models": {a: bot.MODELS[a] for a in models}, "rows": rows}
    (run_dir / "run.json").write_text(json.dumps(run, ensure_ascii=False, indent=1), encoding="utf-8")
    (run_dir / "key.json").write_text(
        json.dumps({r["index"]: r["key"] for r in rows}, indent=1), encoding="utf-8")
    total = sum(r2["cost"] + r2["judge"].get("judge_cost", 0)
                for r in rows for r2 in r["results"].values())
    print(f"прогон сохранён: {run_dir.name}, стоимость ≈ ${total:.2f}")
    if not args.no_send:
        send_pack(rows, args.user_id)
        print("тексты отправлены в Telegram")


if __name__ == "__main__":
    main()
