#!/usr/bin/env python3
"""Главный CLI: построение отчёта по паре «эталон / ученик».

Пример:
    python3 analyze.py \\
        --teacher path/to/teacher.wav \\
        --student path/to/student.wav \\
        [--out ./reports] \\
        [--model qwen3:4b] \\
        [--no-feedback]

Для каждой пары создаётся подпапка ``<out>/<teacher>__vs__<student>/`` с:
    - ``report_student.md``    — отчёт для ученика (графики + LLM-фидбэк)
    - ``analysis_data.md``     — полный технический отчёт со всеми метриками
    - ``report.json``          — машиночитаемые метрики
    - ``img/``                 — PNG-визуализации (общие для обоих .md)
"""

from __future__ import annotations

import argparse
import json
import os
import random
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
from vocal_analysis.utils import OLLAMA_CHAT_URL, OLLAMA_MODEL  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Построить отчёт сравнения «эталон / ученик» для вокального упражнения."
    )
    parser.add_argument("--teacher", required=True, help="Путь к WAV эталона")
    parser.add_argument("--student", required=True, help="Путь к WAV ученика")
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
    args = parser.parse_args()

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

    print(f"\n✓ Отчёт ученика:    {student_md_path}")
    print(f"✓ Тех. анализ:      {analysis_md_path}")
    print(f"✓ JSON метрик:      {json_path}")
    print(f"✓ Визуализации:     {img_dir}/  ({len(img_paths)} PNG)")


if __name__ == "__main__":
    main()
