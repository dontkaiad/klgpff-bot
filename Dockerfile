FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY claude_tg_bot.py .

# Data dirs — mounted as volumes at runtime
RUN mkdir -p facts prompts models outputs

CMD ["python", "claude_tg_bot.py"]
