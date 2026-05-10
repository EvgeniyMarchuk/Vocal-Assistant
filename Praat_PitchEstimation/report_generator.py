#!/usr/bin/env python3
"""
report_generator.py — Extended Markdown report for Estill Voice Training exercise analysis.

Usage:
    python3 report_generator.py \\
        --teacher /path/to/teacher.wav \\
        --student /path/to/student.wav \\
        [--out ./reports] \\
        [--model qwen3:4b] \\
        [--no-feedback]

Outputs into <out>/  :
    report.md             — full Markdown report with embedded images
    img/                  — PNG visualizations
    report.json           — raw metrics
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from pathlib import Path

import numpy as np

from rg_utils import OLLAMA_MODEL, OLLAMA_CHAT_URL, fmt
from rg_features import extract_features, attack_metrics
from rg_alignment import align_by_pitch
from rg_evaluation import evaluate, build_flags, build_text_report
from rg_visualization import save_all_plots
from rg_feedback import generate_feedback

warnings.filterwarnings("ignore")

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(name: str, maxlen: int = 60) -> str:
    name = re.sub(r"[^\wЀ-ӿ\s-]", "", name)
    name = re.sub(r"\s+", "_", name.strip())
    return name[:maxlen]


def build_paths(teacher_audio, student_audio, out_dir: Path):
    t_stem = _sanitize(Path(teacher_audio).stem)
    s_stem = _sanitize(Path(student_audio).stem)
    pair_dir = out_dir / f"{t_stem}__vs__{s_stem}"
    pair_dir.mkdir(parents=True, exist_ok=True)
    return (
        pair_dir / "report.md",
        pair_dir / "report.json",
        pair_dir / "img",
        f"{t_stem}__vs__{s_stem}",
    )


def metrics_to_json_safe(m: dict) -> dict:
    result = {}
    for k, v in m.items():
        if isinstance(v, (float, np.floating)):
            result[k] = None if not np.isfinite(float(v)) else float(v)
        elif isinstance(v, (int, np.integer)):
            result[k] = int(v)
        else:
            result[k] = v
    return result


# ─────────────────────────────────────────────────────────────────────────────
# MD report builder
# ─────────────────────────────────────────────────────────────────────────────

def _img(label, rel_path):
    return f"![{label}]({rel_path})"


def build_md_report(
    teacher_path, student_path,
    teacher, student,
    alignment, metrics,
    img_paths: dict,
    img_rel_prefix: str,
    text_report: str,
    feedback: str | None,
) -> str:
    m = metrics
    flags = build_flags(teacher, student, metrics)

    def img(label, name):
        return _img(label, f"{img_rel_prefix}/{name}")

    def score_badge(score):
        if not np.isfinite(score):
            return "—"
        if score >= 80:
            emoji = "✅"
        elif score >= 60:
            emoji = "⚠️"
        else:
            emoji = "❌"
        return f"{emoji} {fmt(score, 1)}/100"

    lines = []
    lines += [
        f"# Анализ упражнения: Teacher vs Student",
        "",
        f"> **Teacher:** `{Path(teacher_path).name}`  ",
        f"> **Student:** `{Path(student_path).name}`",
        "",
    ]

    lines += [
        "## Сводная оценка",
        "",
        "| Категория | Балл |",
        "|-----------|------|",
        f"| **Общий** | {score_badge(m['overall_score'])} |",
        f"| Интонация | {score_badge(m['intonation_score'])} |",
        f"| Ритм | {score_badge(m['rhythm_score'])} |",
        f"| Атака звука | {score_badge(m['attack_score'])} |",
        f"| Дыхание / поддержка | {score_badge(m['breath_score'])} |",
        f"| Смыкание / голосовой контроль | {score_badge(m['voice_closure_score'])} |",
        "",
        img("Scores summary", "scores_summary.png"),
        "",
    ]

    lines += [
        "## Базовые параметры",
        "",
        "| Параметр | Teacher | Student | Разница |",
        "|----------|---------|---------|---------|",
        f"| Длительность (s) | {fmt(teacher['duration'])} | {fmt(student['duration'])} | {fmt(student['duration'] - teacher['duration'])} |",
        f"| Tempo (BPM) | {fmt(teacher['tempo'], 1)} | {fmt(student['tempo'], 1)} | {fmt(m['tempo_diff_bpm'], 1)} |",
        f"| Onsets detected | {len(teacher['onsets'])} | {len(student['onsets'])} | — |",
        f"| Voiced ratio | {fmt(teacher['voiced_ratio'], 3)} | {fmt(student['voiced_ratio'], 3)} | {fmt(m['voiced_ratio_diff'], 3)} |",
        f"| Silent gaps (≥150 ms) | {m['silent_gaps_teacher_count']} | {m['silent_gaps_student_count']} | — |",
        f"| Mean silent gap (s) | {fmt(m['silent_gap_mean_teacher_s'])} | {fmt(m['silent_gap_mean_student_s'])} | — |",
        "",
    ]

    lines += [
        "## Интонация",
        "",
        img("Pitch contours", "pitch_contours.png"),
        "",
        img("DTW pitch error", "pitch_error.png"),
        "",
        img("Pitch error histogram", "pitch_error_hist.png"),
        "",
        "| Метрика | Значение |",
        "|---------|---------|",
        f"| Aligned voiced frames | {len(alignment['pitch_errors_cents'])} |",
        f"| DTW distance | {fmt(alignment['dtw_distance'], 1)} |",
        f"| Mean abs error (cents) | {fmt(m['pitch_mean_abs_cents'], 1)} |",
        f"| Median abs error (cents) | {fmt(m['pitch_median_abs_cents'], 1)} |",
        f"| Pitch bias student−teacher (cents) | {fmt(m['pitch_bias_cents'], 1)} |",
        f"| Pitch spread std (cents) | {fmt(m['pitch_std_cents'], 1)} |",
        f"| P90 abs error (cents) | {fmt(m['pitch_p90_abs_cents'], 1)} |",
        f"| In tune ±25c | {fmt(m['in_tune_25_pct'], 1)}% |",
        f"| In tune ±50c | {fmt(m['in_tune_50_pct'], 1)}% |",
        f"| In tune ±100c | {fmt(m['in_tune_100_pct'], 1)}% |",
        "",
    ]

    lines += [
        "## Спектральный анализ",
        "",
        "### Log Mel спектрограммы",
        "",
        img("Log Mel — Teacher", "logmel_teacher.png"),
        "",
        img("Log Mel — Student", "logmel_student.png"),
        "",
        "### MFCC (тембр и артикуляция)",
        "",
        img("MFCC comparison", "mfcc_comparison.png"),
        "",
        "### Долгосрочный средний спектр (LTAS)",
        "",
        img("LTAS", "ltas.png"),
        "",
    ]

    lines += [
        "## Форманты",
        "",
        img("Formants F1/F2", "formants.png"),
        "",
    ]

    lines += [
        "## Динамика громкости",
        "",
        img("Intensity dynamics", "intensity_dynamics.png"),
        "",
    ]

    lines += [
        "## Ритм и атака",
        "",
        "| Метрика | Teacher | Student | Разница |",
        "|---------|---------|---------|---------|",
        f"| Onset MAE (ms) | — | {fmt(m['onset_mae_ms'], 1)} | — |",
        f"| Duration MAE (ms) | — | {fmt(m['duration_mae_ms'], 1)} | — |",
        f"| Attack rise time (ms) | {fmt(m['attack_rise_teacher_ms'], 1)} | {fmt(m['attack_rise_student_ms'], 1)} | {fmt(m['attack_rise_student_ms'] - m['attack_rise_teacher_ms'] if np.isfinite(m['attack_rise_student_ms']) and np.isfinite(m['attack_rise_teacher_ms']) else np.nan, 1)} |",
        f"| Attack gain (dB) | {fmt(m['attack_gain_teacher_db'], 1)} | {fmt(m['attack_gain_student_db'], 1)} | {fmt(m['attack_gain_student_db'] - m['attack_gain_teacher_db'] if np.isfinite(m['attack_gain_student_db']) and np.isfinite(m['attack_gain_teacher_db']) else np.nan, 1)} |",
        "",
    ]

    lines += [
        "## Голосовой контроль (Praat)",
        "",
        "| Метрика | Teacher | Student | Разница |",
        "|---------|---------|---------|---------|",
        f"| HNR mean (dB) | {fmt(m['hnr_teacher_db'])} | {fmt(m['hnr_student_db'])} | {fmt(m['hnr_diff_db'])} |",
        f"| Jitter (local) | {fmt(m['jitter_teacher'], 4)} | {fmt(m['jitter_student'], 4)} | {fmt(m['jitter_student'] - m['jitter_teacher'], 4)} |",
        f"| Shimmer (local) | {fmt(m['shimmer_teacher'], 4)} | {fmt(m['shimmer_student'], 4)} | {fmt(m['shimmer_student'] - m['shimmer_teacher'], 4)} |",
        "",
    ]

    lines += [
        "## Вибрато",
        "",
        "| Метрика | Teacher | Student | Разница |",
        "|---------|---------|---------|---------|",
        f"| Vibrato rate (Hz) | {fmt(m['vibrato_rate_teacher_hz'])} | {fmt(m['vibrato_rate_student_hz'])} | {fmt(m['vibrato_rate_student_hz'] - m['vibrato_rate_teacher_hz'] if np.isfinite(m['vibrato_rate_student_hz']) and np.isfinite(m['vibrato_rate_teacher_hz']) else np.nan)} |",
        f"| Vibrato extent (Hz) | {fmt(m['vibrato_extent_teacher_hz'])} | {fmt(m['vibrato_extent_student_hz'])} | {fmt(m['vibrato_extent_student_hz'] - m['vibrato_extent_teacher_hz'] if np.isfinite(m['vibrato_extent_student_hz']) and np.isfinite(m['vibrato_extent_teacher_hz']) else np.nan)} |",
        "",
    ]

    lines += [
        "## Estill Voice Training — специфические признаки",
        "",
        img("CPP и Alpha ratio", "cpp_alpha.png"),
        "",
        "| Признак | Teacher | Student | Норма / Интерпретация |",
        "|---------|---------|---------|----------------------|",
        f"| CPP mean (dB) | {fmt(m['cpp_mean_teacher'], 1)} | {fmt(m['cpp_mean_student'], 1)} | ≥20: чистый; 15–20: ok; <15: придыхательный |",
        f"| H1−H2 mean (dB) | {fmt(m['h1h2_mean_teacher_db'], 1)} | {fmt(m['h1h2_mean_student_db'], 1)} | >6: открытое смыкание/sob; 0–6: balanced; <0: прессованное/belt |",
        f"| Alpha ratio (dB) | {fmt(m['alpha_ratio_teacher_db'], 1)} | {fmt(m['alpha_ratio_student_db'], 1)} | выше = twang/metal; ниже = sob/opera |",
        f"| Singer's formant (%) | {fmt(m['singer_formant_teacher_pct'], 1)} | {fmt(m['singer_formant_student_pct'], 1)} | профессиональный «звон» 2500–3500 Hz |",
        f"| Spectral tilt (dB/oct) | {fmt(m['spectral_tilt_teacher_db_oct'], 1)} | {fmt(m['spectral_tilt_student_db_oct'], 1)} | менее −5: belt; около −10: speech; ниже −15: sob |",
        "",
        "> **Интерпретация Estill:** CPP и H1−H2 вместе дают представление об открытости голосовых "
        "складок. Alpha ratio и spectral tilt — о тембровом «цвете» (twang vs. sob). "
        "Singer's formant — о проекции голоса.",
        "",
    ]

    lines += [
        "## Приоритетные выводы для педагога",
        "",
    ]
    if flags:
        for i, flag in enumerate(flags, 1):
            lines.append(f"{i}. {flag}")
    else:
        lines.append(
            "Критичных отклонений не обнаружено. Рекомендуется закрепить "
            "стабильное выполнение упражнения."
        )
    lines.append("")

    lines += ["## Фидбэк ученику (AI)", ""]
    if feedback:
        lines.append(feedback)
    else:
        lines.append(
            "_Генерация фидбэка пропущена. "
            "Запусти с `--model <ollama_model>` и убедись, что Ollama запущен._"
        )
    lines.append("")

    lines += [
        "---",
        "<details>",
        "<summary>Технический отчёт (raw, для LLM)</summary>",
        "",
        "```",
        text_report,
        "```",
        "",
        "</details>",
    ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate rich Markdown report for vocal exercise analysis."
    )
    parser.add_argument("--teacher", required=True, help="Path to teacher WAV")
    parser.add_argument("--student", required=True, help="Path to student WAV")
    parser.add_argument("--out", default="./reports", help="Output directory (default: ./reports)")
    parser.add_argument("--model", default=OLLAMA_MODEL, help=f"Ollama model (default: {OLLAMA_MODEL})")
    parser.add_argument("--ollama-url", default=OLLAMA_CHAT_URL, help="Ollama API URL")
    parser.add_argument("--no-feedback", action="store_true", help="Skip LLM feedback generation")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Extracting features: {Path(args.teacher).name} ...", flush=True)
    teacher = extract_features(args.teacher)

    print(f"[2/6] Extracting features: {Path(args.student).name} ...", flush=True)
    student = extract_features(args.student)

    print("[3/6] DTW alignment + evaluation ...", flush=True)
    alignment = align_by_pitch(teacher, student)

    n = min(len(teacher["onsets"]), len(student["onsets"]))
    onset_errors = student["onsets"][:n] - teacher["onsets"][:n] if n > 0 else np.array([])
    duration_errors = (np.diff(student["onsets"][:n]) - np.diff(teacher["onsets"][:n])
                       if n >= 2 else np.array([]))

    teacher_attacks = attack_metrics(teacher["time"], teacher["intensity"], teacher["onsets"])
    student_attacks = attack_metrics(student["time"], student["intensity"], student["onsets"])

    metrics = evaluate(teacher, student, alignment, onset_errors, duration_errors,
                       teacher_attacks, student_attacks)
    metrics["dtw_distance"] = alignment["dtw_distance"]

    print("[4/6] Generating visualizations ...", flush=True)
    md_path, json_path, img_dir, base_name = build_paths(args.teacher, args.student, out_dir)
    img_paths = save_all_plots(teacher, student, alignment, metrics, img_dir)

    print("[5/6] Building text report ...", flush=True)
    text_report = build_text_report(
        args.teacher, args.student, teacher, student, alignment, metrics
    )

    feedback = None
    if not args.no_feedback:
        print(f"[6/6] Generating LLM feedback (model={args.model}) ...", flush=True)
        try:
            feedback = generate_feedback(text_report, model=args.model, url=args.ollama_url)
        except RuntimeError as exc:
            print(f"  WARNING: {exc}", file=__import__("sys").stderr)
    else:
        print("[6/6] Skipping LLM feedback (--no-feedback).", flush=True)

    print("Building Markdown report ...", flush=True)
    md_content = build_md_report(
        teacher_path=args.teacher,
        student_path=args.student,
        teacher=teacher,
        student=student,
        alignment=alignment,
        metrics=metrics,
        img_paths=img_paths,
        img_rel_prefix="img",
        text_report=text_report,
        feedback=feedback,
    )

    md_path.write_text(md_content, encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "teacher": args.teacher,
                "student": args.student,
                "metrics": metrics_to_json_safe(metrics),
                "priority_flags": build_flags(teacher, student, metrics),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"\n✓ Report:  {md_path}")
    print(f"✓ Metrics: {json_path}")
    print(f"✓ Images:  {img_dir}/  ({len(img_paths)} files)")


if __name__ == "__main__":
    main()
