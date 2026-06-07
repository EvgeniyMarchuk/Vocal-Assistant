from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from .evaluation import build_flags
from .utils import fmt


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
        pair_dir / "report_student.md",
        pair_dir / "analysis_data.md",
        pair_dir / "report.json",
        pair_dir / "img",
        f"{t_stem}__vs__{s_stem}",
    )


def metrics_to_json_safe(m: dict) -> dict:
    result = {}
    for k, v in m.items():
        if isinstance(v, float | np.floating):
            result[k] = None if not np.isfinite(float(v)) else float(v)
        elif isinstance(v, int | np.integer):
            result[k] = int(v)
        else:
            result[k] = v
    return result


def _score_badge(score) -> str:
    if not np.isfinite(score):
        return "—"
    if score >= 80:
        emoji = "✅"
    elif score >= 60:
        emoji = "⚠️"
    else:
        emoji = "❌"
    return f"{emoji} {fmt(score, 1)}/100"


# ─────────────────────────────────────────────────────────────────────────────
# Student report — relative img/ links, distributable as a folder
# ─────────────────────────────────────────────────────────────────────────────


def build_student_md(
    teacher_path,
    student_path,
    metrics,
    img_paths: dict,
    feedback: str | None,
) -> str:
    m = metrics

    def img(name: str, label: str) -> str:
        return f"![{label}](img/{name})" if name in img_paths else ""

    lines = [
        "# Отчёт по упражнению",
        "",
        f"**Ученик:** {Path(student_path).stem}  ",
        f"**Эталон:** {Path(teacher_path).stem}",
        "",
        "## Итоговые оценки",
        "",
        img("scores_summary.png", "Итоговые оценки"),
        "",
        "## Обратная связь",
        "",
    ]
    if feedback:
        lines.append(feedback)
    else:
        lines.append(
            "_Фидбэк не был сгенерирован. "
            "Запусти без `--no-feedback` и убедись, что Ollama запущен._"
        )
    lines.append("")

    lines += ["## Визуальный анализ", ""]

    if "pitch_contours.png" in img_paths:
        lines += [
            "### Высота тона (интонация)",
            "",
            "> Синяя линия — эталон, коралловая — ученик. Чем ближе линии, "
            "тем точнее интонация. Серая сетка — полутоновая шкала, ось Y "
            "подписана нотами.",
            "",
            img("pitch_contours.png", "Сравнение высоты тона"),
            "",
        ]

    if "pitch_error.png" in img_paths:
        lines += [
            "### Отклонение высоты тона (центы)",
            "",
            "> Ноль — идеальное совпадение. Зелёная зона ±25¢ — «хорошо», "
            "жёлтая ±50¢ — «приемлемо», красная ±100¢ — «плохо».",
            "",
            img("pitch_error.png", "Ошибка интонации"),
            "",
        ]

    if "intensity_dynamics.png" in img_paths:
        lines += [
            "### Динамика громкости",
            "",
            img("intensity_dynamics.png", "Динамика громкости"),
            "",
        ]

    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Technical analysis report — full table set + LLM input
# ─────────────────────────────────────────────────────────────────────────────


def build_analysis_data_md(
    teacher_path,
    student_path,
    teacher,
    student,
    alignment,
    metrics,
    img_paths: dict,
    text_report: str,
    feedback: str | None,
) -> str:
    m = metrics
    flags = build_flags(teacher, student, metrics)

    def img(label, name):
        return f"![{label}](img/{name})"

    lines = [
        f"# Технический анализ: {Path(teacher_path).stem} vs {Path(student_path).stem}",
        "",
        f"> **Эталон:** `{Path(teacher_path).name}`  ",
        f"> **Ученик:** `{Path(student_path).name}`",
        "",
        "## Сводная оценка",
        "",
        "| Категория | Балл |",
        "|-----------|------|",
        f"| **Общий** | {_score_badge(m['overall_score'])} |",
        f"| Интонация | {_score_badge(m['intonation_score'])} |",
        f"| Ритм | {_score_badge(m['rhythm_score'])} |",
        f"| Атака звука | {_score_badge(m['attack_score'])} |",
        f"| Дыхание / поддержка | {_score_badge(m['breath_score'])} |",
        f"| Смыкание / голосовой контроль | {_score_badge(m['voice_closure_score'])} |",
        "",
        img("Сводка оценок", "scores_summary.png"),
        "",
        "## Базовые параметры",
        "",
        "| Параметр | Эталон | Ученик | Разница |",
        "|----------|--------|--------|---------|",
        f"| Длительность (с) | {fmt(teacher['duration'])} | {fmt(student['duration'])} | {fmt(student['duration'] - teacher['duration'])} |",
        f"| Темп (BPM) | {fmt(teacher['tempo'], 1)} | {fmt(student['tempo'], 1)} | {fmt(m['tempo_diff_bpm'], 1)} |",
        f"| Найдено онсетов | {len(teacher['onsets'])} | {len(student['onsets'])} | — |",
        f"| Voiced ratio | {fmt(teacher['voiced_ratio'], 3)} | {fmt(student['voiced_ratio'], 3)} | {fmt(m['voiced_ratio_diff'], 3)} |",
        f"| Паузы ≥150 мс | {m['silent_gaps_teacher_count']} | {m['silent_gaps_student_count']} | — |",
        f"| Средняя длина паузы (с) | {fmt(m['silent_gap_mean_teacher_s'])} | {fmt(m['silent_gap_mean_student_s'])} | — |",
        "",
        "## Интонация",
        "",
        img("Контуры F0", "pitch_contours.png"),
        "",
        img("DTW-выравненная ошибка интонации", "pitch_error.png"),
        "",
        img("Гистограмма ошибки интонации", "pitch_error_hist.png"),
        "",
        "| Метрика | Значение |",
        "|---------|---------|",
        f"| Aligned voiced frames | {len(alignment['pitch_errors_cents'])} |",
        f"| DTW distance | {fmt(alignment['dtw_distance'], 1)} |",
        f"| Mean abs error (центы) | {fmt(m['pitch_mean_abs_cents'], 1)} |",
        f"| Median abs error (центы) | {fmt(m['pitch_median_abs_cents'], 1)} |",
        f"| Pitch bias ученик−эталон (центы) | {fmt(m['pitch_bias_cents'], 1)} |",
        f"| Pitch spread std (центы) | {fmt(m['pitch_std_cents'], 1)} |",
        f"| P90 abs error (центы) | {fmt(m['pitch_p90_abs_cents'], 1)} |",
        f"| In tune ±25¢ | {fmt(m['in_tune_25_pct'], 1)}% |",
        f"| In tune ±50¢ | {fmt(m['in_tune_50_pct'], 1)}% |",
        f"| In tune ±100¢ | {fmt(m['in_tune_100_pct'], 1)}% |",
        "",
        "## Спектральный анализ",
        "",
        "### Лог-мел спектрограммы",
        "",
        img("Лог-мел — Эталон", "logmel_teacher.png"),
        "",
        img("Лог-мел — Ученик", "logmel_student.png"),
        "",
        "### MFCC (тембр и артикуляция)",
        "",
        img("Сравнение MFCC", "mfcc_comparison.png"),
        "",
        "### Долгосрочный средний спектр (LTAS)",
        "",
        img("LTAS", "ltas.png"),
        "",
        "## Форманты",
        "",
        img("Форманты F1/F2", "formants.png"),
        "",
        "## Динамика громкости",
        "",
        img("Интенсивность и RMS", "intensity_dynamics.png"),
        "",
        "## Ритм и атака",
        "",
        "| Метрика | Эталон | Ученик | Разница |",
        "|---------|--------|--------|---------|",
        f"| Onset MAE (мс) | — | {fmt(m['onset_mae_ms'], 1)} | — |",
        f"| Duration MAE (мс) | — | {fmt(m['duration_mae_ms'], 1)} | — |",
        f"| Время атаки (мс) | {fmt(m['attack_rise_teacher_ms'], 1)} | {fmt(m['attack_rise_student_ms'], 1)} | {fmt(m['attack_rise_student_ms'] - m['attack_rise_teacher_ms'] if np.isfinite(m['attack_rise_student_ms']) and np.isfinite(m['attack_rise_teacher_ms']) else np.nan, 1)} |",
        f"| Прирост громкости (дБ) | {fmt(m['attack_gain_teacher_db'], 1)} | {fmt(m['attack_gain_student_db'], 1)} | {fmt(m['attack_gain_student_db'] - m['attack_gain_teacher_db'] if np.isfinite(m['attack_gain_student_db']) and np.isfinite(m['attack_gain_teacher_db']) else np.nan, 1)} |",
        "",
        "## Голосовой контроль (Praat)",
        "",
        "| Метрика | Эталон | Ученик | Разница |",
        "|---------|--------|--------|---------|",
        f"| HNR mean (дБ) | {fmt(m['hnr_teacher_db'])} | {fmt(m['hnr_student_db'])} | {fmt(m['hnr_diff_db'])} |",
        f"| Jitter (local) | {fmt(m['jitter_teacher'], 4)} | {fmt(m['jitter_student'], 4)} | {fmt(m['jitter_student'] - m['jitter_teacher'], 4)} |",
        f"| Shimmer (local) | {fmt(m['shimmer_teacher'], 4)} | {fmt(m['shimmer_student'], 4)} | {fmt(m['shimmer_student'] - m['shimmer_teacher'], 4)} |",
        "",
        "## Вибрато",
        "",
        "| Метрика | Эталон | Ученик | Разница |",
        "|---------|--------|--------|---------|",
        f"| Vibrato rate (Гц) | {fmt(m['vibrato_rate_teacher_hz'])} | {fmt(m['vibrato_rate_student_hz'])} | {fmt(m['vibrato_rate_student_hz'] - m['vibrato_rate_teacher_hz'] if np.isfinite(m['vibrato_rate_student_hz']) and np.isfinite(m['vibrato_rate_teacher_hz']) else np.nan)} |",
        f"| Vibrato extent (Гц) | {fmt(m['vibrato_extent_teacher_hz'])} | {fmt(m['vibrato_extent_student_hz'])} | {fmt(m['vibrato_extent_student_hz'] - m['vibrato_extent_teacher_hz'] if np.isfinite(m['vibrato_extent_student_hz']) and np.isfinite(m['vibrato_extent_teacher_hz']) else np.nan)} |",
        "",
        "## Estill Voice Training — специфические признаки",
        "",
        img("CPP и Alpha ratio", "cpp_alpha.png"),
        "",
        "| Признак | Эталон | Ученик | Норма / интерпретация |",
        "|---------|--------|--------|----------------------|",
        f"| CPP mean (дБ) | {fmt(m['cpp_mean_teacher'], 1)} | {fmt(m['cpp_mean_student'], 1)} | ≥20: чистый; 15–20: ok; <15: придыхательный |",
        f"| H1−H2 mean (дБ) | {fmt(m['h1h2_mean_teacher_db'], 1)} | {fmt(m['h1h2_mean_student_db'], 1)} | >6: открытое смыкание/sob; 0–6: balanced; <0: прессованное/belt |",
        f"| Alpha ratio (дБ) | {fmt(m['alpha_ratio_teacher_db'], 1)} | {fmt(m['alpha_ratio_student_db'], 1)} | выше = twang/metal; ниже = sob/opera |",
        f"| Singer's formant (%) | {fmt(m['singer_formant_teacher_pct'], 1)} | {fmt(m['singer_formant_student_pct'], 1)} | профессиональный «звон» 2500–3500 Гц |",
        f"| Spectral tilt (дБ/окт) | {fmt(m['spectral_tilt_teacher_db_oct'], 1)} | {fmt(m['spectral_tilt_student_db_oct'], 1)} | менее −5: belt; около −10: speech; ниже −15: sob |",
        "",
        "> **Интерпретация Estill:** CPP и H1−H2 вместе дают представление об открытости голосовых "
        "складок. Alpha ratio и spectral tilt — о тембровом «цвете» (twang vs. sob). "
        "Singer's formant — о проекции голоса.",
        "",
        "## Приоритетные выводы",
        "",
    ]

    if flags:
        for i, flag in enumerate(flags, 1):
            lines.append(f"{i}. {flag}")
    else:
        lines.append(
            "Критичных отклонений не обнаружено. "
            "Рекомендуется закрепить стабильное выполнение упражнения."
        )
    lines.append("")

    lines += [
        "## Технический отчёт (входные данные для LLM)",
        "",
        "```",
        text_report,
        "```",
        "",
    ]

    if feedback:
        lines += [
            "## Сгенерированный фидбэк",
            "",
            feedback,
            "",
        ]

    return "\n".join(lines)
