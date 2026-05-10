# Vocal Assistant

Система анализа вокальных упражнений и генерации педагогической обратной связи. Разработана в рамках дипломной работы.

Проект состоит из трёх независимых компонентов:
- **Telegram-бот** — приём голосовых сообщений и конвертация аудио
- **Система анализа** — акустический анализ пары «учитель / ученик» и генерация фидбэка через LLM
- **ML-эксперименты** — исследование SSL-моделей (BEATs, Wav2Vec2, WavLM, Whisper) на датасете VocalSet

---

## Структура репозитория

```
Vocal-Assistant/
├── bot/                          # Telegram-бот
│   ├── bot.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── .env.example
├── Praat_PitchEstimation/        # Анализ и генерация фидбэка
│   ├── report_generator.py       # Основной скрипт: полный отчёт + LLM-фидбэк
│   ├── generate_feedback.py      # Генерация фидбэка по готовому отчёту
│   ├── pitch.py                  # Базовое извлечение Praat-признаков
│   ├── time_pitch.py             # Временной анализ питча + DTW
│   └── requirements.txt
├── MainExperiments/              # Jupyter-ноутбуки с ML-экспериментами
│   ├── BEATs/                    # Реализация модели BEATs (Microsoft)
│   ├── outputs/                  # Метрики обученных моделей (JSON, CSV)
│   ├── tsne_plots/               # t-SNE / PCA визуализации эмбеддингов
│   └── *.ipynb
└── reports/                      # Сгенерированные отчёты (создаётся автоматически)
```

---

## Компонент 1: Анализ голоса + фидбэк

### Требования

- Python 3.10+
- [Ollama](https://ollama.com) с установленной моделью
- ffmpeg (для конвертации аудио)

### Установка

```bash
cd Praat_PitchEstimation
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Установить Ollama и скачать модель:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b   # ~4.7 GB, рекомендуется для GPU ≥ 8 GB VRAM
```

### Подготовка аудио

Скрипт принимает файлы в формате WAV. Конвертация из других форматов:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -acodec pcm_s16le output.wav
```

### Запуск анализа

```bash
# Из корня репозитория (если используется корневой .venv)
python3 Praat_PitchEstimation/report_generator.py \
    --teacher path/to/teacher.wav \
    --student path/to/student.wav \
    --out ./reports \
    --model qwen2.5:7b
```

| Флаг | По умолчанию | Описание |
|------|-------------|----------|
| `--teacher` | — | Путь к WAV учителя (обязательно) |
| `--student` | — | Путь к WAV ученика (обязательно) |
| `--out` | `./reports` | Папка для сохранения отчётов |
| `--model` | `qwen3:4b` | Модель Ollama |
| `--ollama-url` | `http://127.0.0.1:11434/api/chat` | Адрес Ollama API |
| `--no-feedback` | — | Пропустить генерацию LLM-фидбэка |

### Структура отчёта

Для каждой пары создаётся отдельная папка `reports/<учитель>__vs__<ученик>/`:

```
reports/
└── Спич_да__vs__Спич_нет_смена_смыкания/
    ├── report.md        ← Markdown-отчёт с визуализациями и фидбэком
    ├── report.json      ← Сырые метрики в JSON
    └── img/             ← Визуализации (11 PNG)
        ├── pitch_contours.png
        ├── pitch_error.png
        ├── pitch_error_hist.png
        ├── scores_summary.png
        ├── logmel_teacher.png
        ├── logmel_student.png
        ├── mfcc_comparison.png
        ├── ltas.png
        ├── formants.png
        ├── intensity_dynamics.png
        └── cpp_alpha.png
```

### Что анализируется

| Категория | Метрики |
|-----------|---------|
| **Интонация** | DTW-выравнивание F0, средняя ошибка в центах, % нот в ±25/50/100¢ |
| **Ритм** | MAE вступлений (мс), MAE длительностей, расхождение темпа |
| **Атака** | Время нарастания (мс), прирост громкости (dB) |
| **Дыхание** | Voiced ratio, длина пауз |
| **Смыкание** | HNR, jitter, shimmer |
| **Estill-признаки** | CPP, H1−H2, Alpha ratio, Singer's formant, Spectral tilt |

### Только фидбэк по готовому отчёту

Если технический отчёт уже есть, можно запустить только генерацию фидбэка:

```bash
python3 Praat_PitchEstimation/generate_feedback.py \
    --report_path reports/my_pair/report.md \
    --output feedback.txt \
    --backend ollama \
    --ollama-model qwen2.5:7b
```

---

## Компонент 2: Telegram-бот

Принимает голосовые и аудио-сообщения, конвертирует в WAV, возвращает метаданные.

### Запуск через Docker (рекомендуется)

```bash
cd bot
cp .env.example .env
# Вставить реальный токен в .env
docker compose up -d --build
docker compose logs -f    # просмотр логов
docker compose down       # остановка
```

### Локальный запуск

```bash
cd bot
pip install -r requirements.txt
export TELEGRAM_BOT_TOKEN="<токен>"
python3 bot.py
```

Файлы сохраняются в `bot/storage/<chat_id>/`.

---

## Компонент 3: ML-эксперименты

Jupyter-ноутбуки для обучения и оценки моделей на датасете [VocalSet](https://zenodo.org/record/1442513).

```bash
pip install torch torchaudio transformers datasets jupyter
jupyter notebook MainExperiments/
```

| Ноутбук | Модель | Подход |
|---------|--------|--------|
| `DeformableCNN_VocalSet.ipynb` | DeformableCNN | Baseline |
| `Wav2Vec2_AnnotatedVocalSet.ipynb` | Wav2Vec2 | Full / LoRA fine-tuning |
| `WavLM_AnnotatedVocalSet.ipynb` | WavLM | Full / LoRA fine-tuning |
| `Whisper_AnnotatedVocalSet.ipynb` | Whisper | Full fine-tuning |
| `BEATs/BEATs_AnnotatedVocalSet_FineTuning.ipynb` | BEATs | Full fine-tuning |

Результаты обучения (метрики, confusion matrix) сохраняются в `MainExperiments/outputs/<model_name>/`.

---

## Переменные окружения

| Переменная | Компонент | Значение по умолчанию |
|-----------|-----------|----------------------|
| `TELEGRAM_BOT_TOKEN` | bot | — (обязательно) |
| `VOCAL_FEEDBACK_MODEL` | анализ | `qwen3:4b` |
| `OLLAMA_CHAT_URL` | анализ | `http://127.0.0.1:11434/api/chat` |
| `VOCAL_FEEDBACK_GGUF` | анализ | `./mistral-7b-instruct-v0.2.Q4_K_M.gguf` |

---

## Системные зависимости

- **ffmpeg** — конвертация аудио (обязательно для бота и анализа)
- **Ollama** — локальный LLM-сервер для генерации фидбэка

Проверка:

```bash
ffmpeg -version
ollama --version
```
