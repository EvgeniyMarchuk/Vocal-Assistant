from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from rg_utils import OLLAMA_MODEL, OLLAMA_CHAT_URL

FEEDBACK_SYSTEM = (
    "Не выводи внутренний анализ. Не пиши рассуждения перед ответом. "
    "Начинай сразу с '### 1. Общая картина'. "
    "Пиши по-русски, поддерживающим тоном, ясно. Без медицинских выводов."
)

FEEDBACK_USER_TEMPLATE = """\
На основе технического отчёта сформируй педагогический фидбэк для ученика вокала, \
занимающегося по методике Estill Voice Training.

Структура ответа:
### 1. Общая картина (2 предложения)
### 2. Что получилось хорошо (2–3 пункта)
### 3. Главные зоны роста (3–4 пункта; для каждой: что происходит → почему проблема → как звучит)
### 4. Что делать (3 конкретных упражнения с инструкцией)
### 5. План на неделю (3 шага)
### 6. Короткое резюме ученику (1–2 предложения)

Если в отчёте упоминаются Estill-признаки (CPP, H1-H2, alpha ratio, spectral tilt), \
объясни их смысл без жаргона.

Технический отчёт:
{report}
"""


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    hit = re.search(r"(#{1,3}\s*)?1\.\s*\**\s*Общая картина", text, flags=re.IGNORECASE)
    if hit:
        text = text[hit.start():]
    return text.strip()


def generate_feedback(
    text_report: str,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_CHAT_URL,
    max_tokens: int = 1200,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": FEEDBACK_SYSTEM},
            {"role": "user", "content": FEEDBACK_USER_TEMPLATE.format(report=text_report)},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.25,
            "top_p": 0.9,
            "num_ctx": 8192,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Не удалось подключиться к Ollama ({url}). "
            f"Убедись, что сервер запущен и модель установлена: ollama pull {model}"
        ) from exc

    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Ollama вернул пустой ответ: {data}")
    return _strip_thinking(content)
