from __future__ import annotations

import argparse
import json
import os
import re
import urllib.error
import urllib.request


MODEL_PATH = os.environ.get(
    "VOCAL_FEEDBACK_GGUF",
    "./Praat_PitchEstimation/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
)
OLLAMA_MODEL = os.environ.get("VOCAL_FEEDBACK_MODEL", "qwen3:4b")
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")
N_THREADS = min(8, os.cpu_count() or 8)
N_CTX = 4096
N_BATCH = 512
MAX_TOKENS = 1200


PROMPT_TEMPLATE = """[INST]

Ты профессиональный преподаватель вокала и аналитик акустики речи.
Твоя задача — на основе структурированного отчёта сформировать **точный, причинно-обоснованный и педагогически полезный фидбэк**.

---

# ЦЕЛЬ

Сформировать фидбэк, который:

* объясняет ученику, **что происходит в голосе**
* связывает **метрики → ощущение → действие**
* не перегружает числами, но использует их как доказательства

---

# ОСНОВНЫЕ ПРАВИЛА

1. НЕ добавляй новых проблем, которых нет в отчёте
2. НЕ делай медицинских выводов
3. ВСЕ утверждения должны опираться на метрики
4. Если метрика “в норме” — коротко отметь это как сильную сторону
5. Если метрика отклонена — объясни:

   * что это значит
   * как это звучит
   * что делать

---

# ЭТАП 1 — ВНУТРЕННИЙ АНАЛИЗ (НЕ ВЫВОДИТЬ)

Сначала проанализируй все 5 аспектов:

1. Интонация
2. Ритм
3. Атака
4. Дыхание
5. Смыкание

Для каждого определи:

* качество: good / ok / weak
* есть ли проблема
* насколько это критично (1–3)

Затем выбери **1–2 главные проблемы** (самые критичные)

---

# ЭТАП 2 — ФОРМИРОВАНИЕ ФИДБЭКА

## Структура:

### 1. Общая картина (2 предложения)

* кратко: что получилось + главный фокус

---

### 2. Что получилось хорошо (2–3 пункта)

* только реально подтверждённые вещи
* коротко
* без воды

---

### 3. Главные зоны роста (3–4 пункта)

Для каждой проблемы:

Формат:

* Что происходит
* Почему это проблема
* Как это звучит/ощущается

Пример:
"Ты выходишь на звук медленнее, чем учитель (≈45 ms против 35 ms). Это даёт ощущение запаздывающей атаки и снижает чёткость начала нот."

---

### 4. Что делать (3 упражнения)

Каждое упражнение:

* конкретное действие
* как выполнять
* на что обращать внимание

---

### 5. План на неделю (3 шага)

* коротко
* без повторений

---

### 6. Короткая фраза ученику (1–2 предложения)

---

# ВАЖНЫЕ ТРЕБОВАНИЯ К КАЧЕСТВУ

* Пиши простым языком (не академическим)
* Связывай:
  метрика → звук → ощущение → действие
* Не повторяй одни и те же идеи
* Не делай абстрактных советов

---

# ВХОДНЫЕ ДАННЫЕ

Отчёт:

{report_text}

---

# ВЫХОД

Сформируй фидбэк строго по структуре выше.

[/INST]"""


def load_report(path: str) -> str:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Report file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def strip_thinking_blocks(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    heading = re.search(r"(#{1,3}\s*)?1\.\s*\**\s*Общая картина", text, flags=re.IGNORECASE)
    if heading:
        text = text[heading.start():]
    return text.strip()


def build_prompt(report_text: str) -> str:
    return PROMPT_TEMPLATE.format(report_text=report_text.strip())


def generate_feedback_ollama(
    report_text: str,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_CHAT_URL,
    max_tokens: int = MAX_TOKENS,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Не выводи внутренний анализ. Не пиши рассуждения перед ответом. "
                    "Начинай сразу с заголовка '### 1. Общая картина'. "
                    "Если в инструкции есть этап 'ВНУТРЕННИЙ АНАЛИЗ', выполни его молча."
                ),
            },
            {
                "role": "user",
                "content": build_prompt(report_text),
            }
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
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=240) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Could not connect to Ollama. Start Ollama and install the model: "
            f"ollama pull {model}"
        ) from exc

    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Ollama returned an empty response: {data}")
    return strip_thinking_blocks(content)


def generate_feedback_llama_cpp(
    report_text: str,
    model_path: str = MODEL_PATH,
    n_threads: int = N_THREADS,
    n_ctx: int = N_CTX,
    max_tokens: int = MAX_TOKENS,
    stream: bool = True,
) -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            "llama-cpp backend requires a local GGUF model file. "
            "Set VOCAL_FEEDBACK_GGUF or pass --model-path with the path to the model."
        )

    from llama_cpp import Llama

    print("Loading model...")
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_batch=N_BATCH,
        verbose=False,
        n_gpu_layers=-1,
        use_mmap=True,
    )

    prompt = build_prompt(report_text)
    print("Generating feedback...\n")
    if stream:
        chunks = []
        for chunk in llm(
            prompt,
            max_tokens=max_tokens,
            temperature=0.25,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["</s>", "[/INST]"],
            stream=True,
        ):
            text = chunk["choices"][0].get("text", "")
            if text:
                print(text, end="", flush=True)
                chunks.append(text)
        print()
        return strip_thinking_blocks("".join(chunks))

    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.25,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["</s>", "[/INST]"],
    )
    return strip_thinking_blocks(output["choices"][0]["text"])


def generate_feedback(report_text: str, backend: str = "ollama", **kwargs) -> str:
    if backend == "ollama":
        return generate_feedback_ollama(
            report_text,
            model=kwargs.get("ollama_model", OLLAMA_MODEL),
            url=kwargs.get("ollama_url", OLLAMA_CHAT_URL),
            max_tokens=kwargs.get("max_tokens", MAX_TOKENS),
        )
    if backend == "llama-cpp":
        return generate_feedback_llama_cpp(
            report_text,
            model_path=kwargs.get("model_path", MODEL_PATH),
            n_threads=kwargs.get("n_threads", N_THREADS),
            n_ctx=kwargs.get("n_ctx", N_CTX),
            max_tokens=kwargs.get("max_tokens", MAX_TOKENS),
            stream=kwargs.get("stream", True),
        )
    raise ValueError(f"Unknown backend: {backend}")


def main():
    parser = argparse.ArgumentParser(description="Generate vocal feedback from analysis report.")
    parser.add_argument("--report_path", type=str, required=True, help="Path to analysis report file")
    parser.add_argument("--output", type=str, default=None, help="Optional output file")
    parser.add_argument("--backend", choices=["ollama", "llama-cpp"], default="ollama")
    parser.add_argument("--ollama-model", type=str, default=OLLAMA_MODEL)
    parser.add_argument("--ollama-url", type=str, default=OLLAMA_CHAT_URL)
    parser.add_argument("--model-path", type=str, default=MODEL_PATH)
    parser.add_argument("--n-threads", type=int, default=N_THREADS)
    parser.add_argument("--n-ctx", type=int, default=N_CTX)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()

    report_text = load_report(args.report_path)
    feedback = generate_feedback(
        report_text,
        backend=args.backend,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url,
        model_path=args.model_path,
        n_threads=args.n_threads,
        n_ctx=args.n_ctx,
        max_tokens=args.max_tokens,
        stream=not args.no_stream,
    )

    print("\n===== GENERATED FEEDBACK =====\n")
    print(feedback)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(feedback + "\n")
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
