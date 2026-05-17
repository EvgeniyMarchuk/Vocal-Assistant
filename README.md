# Vocal Assistant

Система анализа вокальных упражнений и генерации педагогической обратной связи. Разработана в рамках дипломной работы.

Проект состоит из трёх независимых компонентов:

- **Telegram-бот** — приём голосовых сообщений и конвертация аудио
- **Анализ голоса** — акустический анализ пары «эталон / ученик» и генерация фидбэка через LLM
- **ML-эксперименты** — исследование SSL-моделей (BEATs, Wav2Vec2, WavLM, Whisper) на датасете VocalSet

---

## Структура репозитория

```
Vocal-Assistant/
├── analyze.py                    # Главный скрипт построения отчёта
├── vocal_analysis/               # Пакет анализа
│   ├── __init__.py               # Публичный API
│   ├── features.py               # Извлечение Praat / librosa-признаков
│   ├── alignment.py              # DTW-выравнивание по F0
│   ├── evaluation.py             # Скоры и текстовый отчёт
│   ├── feedback.py               # LLM-фидбэк через Ollama
│   ├── visualization.py          # PNG-графики
│   ├── plot_style.py             # Палитра, единый стиль, helpers
│   ├── report.py                 # Сборка markdown-отчётов
│   ├── utils.py                  # Константы и math-helpers
│   └── requirements.txt
├── bot/                          # Telegram-бот
│   ├── bot.py
│   ├── Dockerfile
│   ├── docker-compose.yml
│   ├── requirements.txt
│   └── .env.example
├── MainExperiments/              # Jupyter-ноутбуки с ML-экспериментами
│   ├── BEATs/                    # Реализация модели BEATs (Microsoft)
│   ├── outputs/                  # Метрики обученных моделей (JSON, CSV)
│   ├── tsne_plots/               # t-SNE / PCA визуализации эмбеддингов
│   └── *.ipynb
└── reports/                      # Сгенерированные отчёты (создаётся автоматически)
    └── <эталон>__vs__<ученик>/
        ├── report_student.md
        ├── analysis_data.md
        ├── report.json
        └── img/*.png
```

---

## Компонент 1: Анализ голоса + фидбэк

### Требования

- Python 3.10+
- [Ollama](https://ollama.com) с установленной моделью
- ffmpeg (для конвертации аудио в WAV, если на входе другой формат)

### Установка

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r vocal_analysis/requirements.txt
```

Установить Ollama и скачать модель:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull qwen2.5:7b   # ~4.7 GB, рекомендуется для GPU ≥ 8 GB VRAM
```

### Подготовка аудио

Скрипт принимает WAV напрямую и автоматически конвертирует другие форматы через ffmpeg. Если хочется сделать это вручную:

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -acodec pcm_s16le output.wav
```

### Запуск анализа

```bash
python3 analyze.py \
    --teacher path/to/teacher.wav \
    --student path/to/student.wav \
    --out ./reports \
    --model qwen2.5:7b
```

| Флаг            | По умолчанию                      | Описание                         |
| --------------- | --------------------------------- | -------------------------------- |
| `--teacher`     | —                                 | Путь к WAV эталона (обязательно) |
| `--student`     | —                                 | Путь к WAV ученика (обязательно) |
| `--out`         | `./reports`                       | Папка для сохранения отчётов     |
| `--model`       | `qwen3:4b`                        | Модель Ollama                    |
| `--ollama-url`  | `http://127.0.0.1:11434/api/chat` | Адрес Ollama API                 |
| `--no-feedback` | —                                 | Пропустить генерацию LLM-фидбэка |

### Структура отчёта

Для каждой пары создаётся отдельная папка `reports/<эталон>__vs__<ученик>/`:

```
reports/
└── Спич_да__vs__Спич_нет_смена_смыкания/
    ├── report_student.md   ← Отчёт для ученика (графики + LLM-фидбэк)
    ├── analysis_data.md    ← Полный технический отчёт со всеми метриками
    ├── report.json         ← Машиночитаемые метрики
    └── img/                ← Визуализации (11 PNG)
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

Чтобы поделиться отчётом — отправь всю папку пары (или zip).

### Что анализируется

| Категория           | Метрики                                                                 |
| ------------------- | ----------------------------------------------------------------------- |
| **Интонация**       | DTW-выравнивание F0, средняя ошибка в центах, % нот в ±25 / ±50 / ±100¢ |
| **Ритм**            | MAE вступлений (мс), MAE длительностей, расхождение темпа               |
| **Атака**           | Время нарастания (мс), прирост громкости (дБ)                           |
| **Дыхание**         | Voiced ratio, длина пауз                                                |
| **Смыкание**        | HNR, jitter, shimmer                                                    |
| **Estill-признаки** | CPP, H1−H2, Alpha ratio, Singer's formant, Spectral tilt                |

### Стиль графиков

Все PNG используют единую палитру и подписи на русском с единицами в скобках:

- **Эталон** — глубокий синий (`#2E5C9E`), **Ученик** — коралловый (`#E26A4F`).
- На графике pitch-контура ось Y подписана нотами (например, `A4`), фоном — полутоновая сетка.
- На графике pitch-error фон поделён на три зоны: зелёная ±25¢ («хорошо»), жёлтая ±50¢ («приемлемо»), красная ±100¢ («плохо»); линии MAE и медианы подписаны.

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

---

## Качество кода и MLOps-style workflow

В репозитории настроены единые инструменты форматирования и проверок:

- **Black** — автоформатирование Python-кода.
- **isort** — сортировка импортов в профиле Black.
- **Ruff** — быстрый линтинг и безопасные автофиксы.
- **Prettier** — форматирование Markdown / YAML / JSON.
- **pre-commit** — запуск всех проверок перед коммитом.

Установка dev-инструментов:

```bash
pip install -r requirements-dev.txt
pre-commit install
```

Ручной прогон всех хуков:

```bash
pre-commit run --all-files
```

Конфигурация инструментов находится в `pyproject.toml`, `.pre-commit-config.yaml`, `.prettierrc.json` и `.prettierignore`.

| Ноутбук                                          | Модель        | Подход                  |
| ------------------------------------------------ | ------------- | ----------------------- |
| `DeformableCNN_VocalSet.ipynb`                   | DeformableCNN | Baseline                |
| `Wav2Vec2_AnnotatedVocalSet.ipynb`               | Wav2Vec2      | Full / LoRA fine-tuning |
| `WavLM_AnnotatedVocalSet.ipynb`                  | WavLM         | Full / LoRA fine-tuning |
| `Whisper_AnnotatedVocalSet.ipynb`                | Whisper       | Full fine-tuning        |
| `BEATs/BEATs_AnnotatedVocalSet_FineTuning.ipynb` | BEATs         | Full fine-tuning        |

Результаты обучения (метрики, confusion matrix) сохраняются в `MainExperiments/outputs/<model_name>/`.

---

## Переменные окружения

| Переменная             | Компонент | Значение по умолчанию             |
| ---------------------- | --------- | --------------------------------- |
| `TELEGRAM_BOT_TOKEN`   | bot       | — (обязательно)                   |
| `VOCAL_FEEDBACK_MODEL` | анализ    | `qwen3:4b`                        |
| `OLLAMA_CHAT_URL`      | анализ    | `http://127.0.0.1:11434/api/chat` |

---

## Системные зависимости

- **ffmpeg** — конвертация аудио (нужен и боту, и анализу для не-WAV-входа)
- **Ollama** — локальный LLM-сервер для генерации фидбэка

Проверка:

```bash
ffmpeg -version
ollama --version
```
