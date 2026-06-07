#!/usr/bin/env python3
"""Главный CLI: построение отчёта по паре «эталон / ученик».

Пример:
    python3 analyze.py \\
        --teacher path/to/teacher.wav \\
        --student path/to/student.wav \\
        [--out ./reports] \\
        [--model qwen3:4b] \\
        [--no-feedback] \\
        [--beautiful-report] \\
        [--html-report]

Для каждой пары создаётся подпапка ``<out>/<teacher>__vs__<student>/`` с:
    - ``report_student.md``    — отчёт для ученика (графики + LLM-фидбэк)
    - ``beautiful_report_student.md`` — красиво оформленный отчёт для ученика
    - ``report_student.html``  — HTML отчёт для ученика с встроенными изображениями
    - ``analysis_data.md``     — полный технический отчёт со всеми метриками
    - ``report.json``          — машиночитаемые метрики
    - ``img/``                 — PNG-визуализации (общие для обоих .md)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import warnings
from pathlib import Path
from time import perf_counter

import numpy as np

# Установка сидов для детерминистичности
SEED = 123
random.seed(SEED)
np.random.seed(SEED)

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")
os.environ["PYTHONHASHSEED"] = str(SEED)

# Дополнительные настройки для детерминистичности библиотек
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Импорт библиотек после установки сидов
import librosa
import scipy

# Настройка детерминистичности для библиотек
# Для librosa DTW алгоритма - DTW в librosa детерминистичен по умолчанию
# Для scipy.signal.find_peaks - алгоритм детерминистичен по умолчанию

# Дополнительная настройка для обеспечения полной детерминистичности
# torch не используется в проекте, поэтому импорт убран

from vocal_analysis.html_report import md_file_to_html  # noqa: E402

from vocal_analysis import (  # noqa: E402  (после env-настройки)
    align_by_pitch,
    attack_metrics,
    build_analysis_data_md,
    build_flags,
    build_paths,
    build_pedagogical_brief,
    build_rule_based_feedback,
    build_student_md,
    build_text_report,
    evaluate,
    extract_features,
    generate_feedback,
    metrics_to_json_safe,
    save_all_plots,
)
from vocal_analysis.beautiful_report import build_beautiful_student_md  # noqa: E402
from vocal_analysis.html_report import build_html_student_report  # noqa: E402
from vocal_analysis.utils import OLLAMA_CHAT_URL, OLLAMA_MODEL  # noqa: E402


_PANDOC_PDF_ARGS = [
    "--pdf-engine=xelatex",
    "-V", "mainfont=PT Serif",
    "-V", "sansfont=PT Sans",
    "-V", "monofont=Courier New",
    "-V", "fontsize=12pt",
    "-V", "geometry:margin=2.5cm",
    "-V", "lang=ru",
    "-V", "colorlinks=true",
    "-V", "linkcolor=blue",
]


_EMOJI_MAP = {
    "✅": "[+]", "⚠️": "[!]", "❌": "[-]",
    "🎤": "", "📊": "", "💬": "", "📈": "",
}


def _preprocess_md_for_pdf(text: str) -> str:
    """Prepare markdown for pandoc: fix heading spacing and remove emoji."""
    # Replace emoji with ASCII equivalents
    for emoji, repl in _EMOJI_MAP.items():
        text = text.replace(emoji, repl)
    # Ensure blank line before every heading (##, ###)
    text = re.sub(r"(?<!\n)\n(#{2,})", r"\n\n\1", text)
    # Remove trailing spaces used as line breaks (they confuse pandoc inside feedback)
    text = re.sub(r"  \n", "\n\n", text)
    return text


def _md_to_pdf(md_path: Path) -> Path:
    """Convert a markdown report to PDF via pandoc + xelatex."""
    out_path = md_path.with_suffix(".pdf")
    # Preprocess to a temp file so we don't modify the original
    tmp_md = md_path.parent / "_tmp_pdf_input.md"
    raw = md_path.read_text(encoding="utf-8")
    tmp_md.write_text(_preprocess_md_for_pdf(raw), encoding="utf-8")
    try:
        cmd = [
            "pandoc", str(tmp_md),
            "-o", str(out_path),
            "--resource-path", str(md_path.parent),
            *_PANDOC_PDF_ARGS,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"pandoc завершился с ошибкой:\n{result.stderr[-800:]}"
            )
    finally:
        tmp_md.unlink(missing_ok=True)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построить отчёт сравнения «эталон / ученик» для вокального упражнения."
    )
    parser.add_argument(
        "--from-md",
        metavar="PATH",
        help="Конвертировать существующий report_student.md в PDF и выйти.",
    )
    parser.add_argument("--teacher", required=False, help="Путь к WAV эталона")
    parser.add_argument("--student", required=False, help="Путь к WAV ученика")
    parser.add_argument(
        "--out",
        default="./reports",
        help="Директория для отчётов (по умолчанию: ./reports)",
    )
    parser.add_argument(
        "--model",
        default=OLLAMA_MODEL,
        help=f"Модель Ollama (по умолчанию: {OLLAMA_MODEL})",
    )
    parser.add_argument(
        "--ollama-url", default=OLLAMA_CHAT_URL, help="Эндпоинт Ollama API"
    )
    parser.add_argument(
        "--no-feedback", action="store_true", help="Пропустить генерацию LLM-фидбэка"
    )
    parser.add_argument(
        "--crepe-pitch", action="store_true", help="Использовать CREPE для извлечения высоты тона вместо Praat"
    )
    parser.add_argument(
        "--beautiful-report", action="store_true", help="Создать красиво оформленный отчет вместо стандартного"
    )
    parser.add_argument(
        "--html-report", action="store_true", help="Создать HTML отчет вместо markdown"
    )
    parser.add_argument(
        "--pdf-report", action="store_true",
        help="Создать PDF отчет (через pandoc + xelatex)",
    )
    args = parser.parse_args()

    # Standalone md→pdf/html conversion: no audio analysis needed
    if args.from_md:
        md_path = Path(args.from_md)
        if not md_path.exists():
            print(f"Файл не найден: {md_path}", file=sys.stderr)
            sys.exit(1)
        if getattr(args, "html_report", False):
            html = md_file_to_html(md_path)
            out_path = md_path.with_suffix(".html")
            out_path.write_text(html, encoding="utf-8")
            print(f"✓ HTML отчёт: {out_path}")
        else:
            print("Генерация PDF (pandoc + xelatex) ...", flush=True)
            try:
                out_path = _md_to_pdf(md_path)
                print(f"✓ PDF отчёт:  {out_path}")
            except RuntimeError as exc:
                print(f"  ОШИБКА: {exc}", file=sys.stderr)
                sys.exit(1)
        return

    if not args.teacher or not args.student:
        parser.error("--teacher и --student обязательны (или используйте --from-md)")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = perf_counter()

    print(f"[1/6] Извлечение признаков: {Path(args.teacher).name} ...", flush=True)
    teacher = extract_features(args.teacher, use_crepe=args.crepe_pitch)

    print(f"[2/6] Извлечение признаков: {Path(args.student).name} ...", flush=True)
    student = extract_features(args.student, use_crepe=args.crepe_pitch)

    print("[3/6] DTW-выравнивание и оценка ...", flush=True)
    alignment = align_by_pitch(teacher, student)

    n = min(len(teacher["onsets"]), len(student["onsets"]))
    if n > 0:
        # Normalize to first onset so we compare *relative* timing within the
        # performance, not when each person started singing.
        t_rel = teacher["onsets"][:n] - teacher["onsets"][0]
        s_rel = student["onsets"][:n] - student["onsets"][0]
        onset_errors = s_rel - t_rel
    else:
        onset_errors = np.array([])
    # IOI (inter-onset intervals) = spacing between consecutive notes.
    # Duration errors measure whether the student holds notes for the same
    # duration as the teacher, regardless of absolute start time.
    duration_errors = (
        np.diff(student["onsets"][:n]) - np.diff(teacher["onsets"][:n])
        if n >= 2
        else np.array([])
    )

    teacher_attacks = attack_metrics(
        teacher["time"], teacher["intensity"], teacher["onsets"]
    )
    student_attacks = attack_metrics(
        student["time"], student["intensity"], student["onsets"]
    )

    metrics = evaluate(
        teacher,
        student,
        alignment,
        onset_errors,
        duration_errors,
        teacher_attacks,
        student_attacks,
    )
    metrics["dtw_distance"] = alignment["dtw_distance"]

    student_md_path, analysis_md_path, json_path, img_dir, _ = build_paths(
        args.teacher, args.student, out_dir
    )

    for audio_path in (args.teacher, args.student):
        src = Path(audio_path)
        dst = student_md_path.parent / src.name
        if not dst.exists() or dst.resolve() != src.resolve():
            shutil.copy2(src, dst)

    print("[4/6] Генерация визуализаций ...", flush=True)
    img_paths = save_all_plots(teacher, student, alignment, metrics, img_dir)

    print("[5/6] Сборка технического отчёта ...", flush=True)
    text_report = build_text_report(
        args.teacher, args.student, teacher, student, alignment, metrics
    )

    feedback = None
    flags = build_flags(teacher, student, metrics)
    pedagogical_brief = build_pedagogical_brief(metrics, flags)
    if not args.no_feedback:
        print(f"[6/6] LLM-фидбэк (model={args.model}) ...", flush=True)
        try:
            feedback = generate_feedback(
                text_report,
                model=args.model,
                url=args.ollama_url,
                pedagogical_brief=pedagogical_brief,
            )
        except RuntimeError as exc:
            print(f"  WARNING: {exc}", file=sys.stderr)
            print("  Использую детерминированный fallback-фидбэк.", file=sys.stderr)
            feedback = build_rule_based_feedback(metrics, flags)
    else:
        print("[6/6] Пропуск LLM-фидбэка (--no-feedback).", flush=True)

    print(f"Время формирования отчета: {perf_counter() - start:.1f}s")
    print("Запись отчётов ...", flush=True)

    # Всегда генерируем markdown — он нужен для PDF и как самостоятельный файл
    student_md_path.write_text(
        build_student_md(
            teacher_path=args.teacher,
            student_path=args.student,
            metrics=metrics,
            img_paths=img_paths,
            feedback=feedback,
        ),
        encoding="utf-8",
    )

    if args.pdf_report:
        print("Генерация PDF (pandoc + xelatex) ...", flush=True)
        try:
            pdf_path = _md_to_pdf(student_md_path)
            print(f"\n✓ PDF отчёт ученика:     {pdf_path}")
        except RuntimeError as exc:
            print(f"  WARNING: PDF не создан — {exc}", file=sys.stderr)
            print(f"\n✓ Отчёт ученика (md):   {student_md_path}")
    elif args.html_report:
        html_content = build_html_student_report(
            teacher_path=args.teacher,
            student_path=args.student,
            metrics=metrics,
            img_paths=img_paths,
            feedback=feedback,
        )
        html_report_path = student_md_path.parent / "report_student.html"
        html_report_path.write_text(html_content, encoding="utf-8")
        print(f"\n✓ HTML отчёт ученика:    {html_report_path}")
    elif args.beautiful_report:
        student_md_content = build_beautiful_student_md(
            teacher_path=args.teacher,
            student_path=args.student,
            metrics=metrics,
            img_paths=img_paths,
            feedback=feedback,
        )
        beautiful_student_md_path = student_md_path.parent / "beautiful_report_student.md"
        beautiful_student_md_path.write_text(student_md_content, encoding="utf-8")
        print(f"\n✓ Красивый отчёт ученика:    {beautiful_student_md_path}")
    else:
        print(f"\n✓ Отчёт ученика:    {student_md_path}")

    analysis_md_path.write_text(
        build_analysis_data_md(
            teacher_path=args.teacher,
            student_path=args.student,
            teacher=teacher,
            student=student,
            alignment=alignment,
            metrics=metrics,
            img_paths=img_paths,
            text_report=text_report,
            feedback=feedback,
        ),
        encoding="utf-8",
    )

    json_path.write_text(
        json.dumps(
            {
                "teacher": args.teacher,
                "student": args.student,
                "metrics": metrics_to_json_safe(metrics),
                "priority_flags": flags,
                "pedagogical_brief": pedagogical_brief,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"✓ Тех. анализ:      {analysis_md_path}")
    print(f"✓ JSON метрик:      {json_path}")
    print(f"✓ Визуализации:     {img_dir}/  ({len(img_paths)} PNG)")


if __name__ == "__main__":
    main()
