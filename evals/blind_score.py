"""Сводка слепого теста: python evals/blind_score.py <папка_прогона> 1BAC 2CAB ...

Каждый токен: номер кейса и буквы от лучшего к худшему. Печатает только цифры,
текстов сцен в выводе нет.
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        run_dir = ROOT / "eval_runs" / sys.argv[1]
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    rows = {r["index"]: r for r in run["rows"]}
    models = list(run["models"])

    ranks = {m: [] for m in models}
    for tok in sys.argv[2:]:
        tok = tok.translate(str.maketrans("АВСавс", "ABCabc"))
        m = re.fullmatch(r"(\d+)([A-Ca-c]{2,3})", tok)
        if not m:
            sys.exit(f"не понял токен {tok!r}, нужен вид 1BAC")
        idx, order = int(m.group(1)), m.group(2).upper()
        row = rows.get(idx)
        if row is None or sorted(order) != sorted(row["key"]):
            sys.exit(f"кейс {idx}: буквы {order!r} не подходят")
        for place, letter in enumerate(order, start=1):
            ranks[row["key"][letter]].append(place)

    n = len(sys.argv) - 2
    out = [f"Слепой тест, кейсов оценено: {n}", ""]
    out.append("| модель | побед | средний ранг | противоречий канону / сцену | в объёме | $ / сцену |")
    out.append("|---|---|---|---|---|---|")
    stats = {}
    for m in models:
        res = [r["results"][m] for r in run["rows"]]
        viol = [x["judge"]["violations"] for x in res if "violations" in x["judge"]]
        bounds = [x["in_bounds"] for x in res if x["in_bounds"] is not None]
        failed = sum(1 for x in res if x["judge"].get("judge_failed"))
        mean_cost = sum(x["cost"] for x in res) / len(res)
        stats[m] = mean_cost
        wins = sum(1 for p in ranks[m] if p == 1)
        mean_rank = sum(ranks[m]) / len(ranks[m]) if ranks[m] else float("nan")
        v = f"{sum(viol) / len(viol):.2f}" if viol else "n/a"
        if failed:
            v += f" (судья упал {failed}×)"
        b = f"{100 * sum(bounds) / len(bounds):.0f}%" if bounds else "n/a"
        out.append(f"| {m} | {wins} из {len(ranks[m])} | {mean_rank:.2f} | {v} | {b} | ${mean_cost:.3f} |")
    cheapest = min(stats.values()) or 1e-9
    out += ["", "Во сколько раз дороже самой дешёвой: " +
            ", ".join(f"{m} ×{c / cheapest:.0f}" for m, c in stats.items())]
    text = "\n".join(out)
    print(text)
    (run_dir / "RESULTS.md").write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
