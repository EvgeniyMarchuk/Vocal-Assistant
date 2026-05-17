from __future__ import annotations

import numpy as np

from .features import attack_summary, vibrato_metrics
from .utils import clamp, fmt, safe_mean, safe_median, safe_pct, safe_std

# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


def evaluate(
    teacher,
    student,
    alignment,
    onset_errors,
    duration_errors,
    teacher_attack_rows,
    student_attack_rows,
):
    pe = alignment["pitch_errors_cents"]
    ape = np.abs(pe)

    p_mean = safe_mean(ape)
    p_median = safe_median(ape)
    p_std = safe_std(pe)
    p_bias = safe_mean(pe)
    p_p90 = safe_pct(ape, 90)
    in_25 = 100.0 * float(np.mean(ape <= 25)) if ape.size else np.nan
    in_50 = 100.0 * float(np.mean(ape <= 50)) if ape.size else np.nan
    in_100 = 100.0 * float(np.mean(ape <= 100)) if ape.size else np.nan

    onset_mae = safe_mean(np.abs(onset_errors)) if len(onset_errors) else np.nan
    dur_mae = safe_mean(np.abs(duration_errors)) if len(duration_errors) else np.nan
    tempo_diff = (
        abs(student["tempo"] - teacher["tempo"])
        if (np.isfinite(student["tempo"]) and np.isfinite(teacher["tempo"]))
        else np.nan
    )

    hnr_t = safe_mean(teacher["hnr"])
    hnr_s = safe_mean(student["hnr"])
    hnr_diff = hnr_s - hnr_t

    vr_diff = student["voiced_ratio"] - teacher["voiced_ratio"]

    t_rise, t_gain = attack_summary(teacher_attack_rows)
    s_rise, s_gain = attack_summary(student_attack_rows)

    vib_t_rate, vib_t_ext = vibrato_metrics(teacher["f0"], teacher["time"])
    vib_s_rate, vib_s_ext = vibrato_metrics(student["f0"], student["time"])

    intonation_score = clamp(100.0 - (p_mean / 1.5 if np.isfinite(p_mean) else 100.0))
    rhythm_score = clamp(
        np.nanmean(
            [
                100.0 - onset_mae * 1000.0 / 5.0 if np.isfinite(onset_mae) else np.nan,
                100.0 - dur_mae * 1000.0 / 7.0 if np.isfinite(dur_mae) else np.nan,
                100.0 - tempo_diff * 2.0 if np.isfinite(tempo_diff) else np.nan,
            ]
        )
    )
    attack_score = clamp(
        np.nanmean(
            [
                (
                    100.0 - abs(s_rise - t_rise) * 1000.0 / 3.0
                    if np.isfinite(s_rise) and np.isfinite(t_rise)
                    else np.nan
                ),
                (
                    100.0 - abs(s_gain - t_gain) * 8.0
                    if np.isfinite(s_gain) and np.isfinite(t_gain)
                    else np.nan
                ),
            ]
        )
    )
    breath_score = clamp(
        np.nanmean(
            [
                100.0 - abs(vr_diff) * 250.0 if np.isfinite(vr_diff) else np.nan,
                100.0
                - abs(
                    safe_mean(student["silent_gaps"])
                    - safe_mean(teacher["silent_gaps"])
                )
                * 100.0,
                100.0 - max(0.0, -hnr_diff) * 4.0 if np.isfinite(hnr_diff) else np.nan,
            ]
        )
    )
    voice_closure_score = clamp(
        np.nanmean(
            [
                100.0 - abs(vr_diff) * 300.0 if np.isfinite(vr_diff) else np.nan,
                100.0 - abs(student["jitter"] - teacher["jitter"]) * 6000.0,
                100.0 - abs(student["shimmer"] - teacher["shimmer"]) * 900.0,
                100.0 - abs(hnr_diff) * 2.0 if np.isfinite(hnr_diff) else np.nan,
            ]
        )
    )
    overall_score = clamp(
        0.40 * intonation_score
        + 0.25 * rhythm_score
        + 0.15 * voice_closure_score
        + 0.10 * attack_score
        + 0.10 * breath_score
    )

    alpha_diff = student["alpha_ratio_db"] - teacher["alpha_ratio_db"]
    sf_diff = student["singer_formant_pct"] - teacher["singer_formant_pct"]
    cpp_t = safe_mean(teacher["cpp_contour"])
    cpp_s = safe_mean(student["cpp_contour"])
    h1h2_t = safe_mean(teacher["h1h2_contour"][np.isfinite(teacher["h1h2_contour"])])
    h1h2_s = safe_mean(student["h1h2_contour"][np.isfinite(student["h1h2_contour"])])

    return {
        "overall_score": overall_score,
        "intonation_score": intonation_score,
        "rhythm_score": rhythm_score,
        "attack_score": attack_score,
        "breath_score": breath_score,
        "voice_closure_score": voice_closure_score,
        "pitch_mean_abs_cents": p_mean,
        "pitch_median_abs_cents": p_median,
        "pitch_bias_cents": p_bias,
        "pitch_std_cents": p_std,
        "pitch_p90_abs_cents": p_p90,
        "in_tune_25_pct": in_25,
        "in_tune_50_pct": in_50,
        "in_tune_100_pct": in_100,
        "onset_mae_ms": onset_mae * 1000.0 if np.isfinite(onset_mae) else np.nan,
        "duration_mae_ms": dur_mae * 1000.0 if np.isfinite(dur_mae) else np.nan,
        "tempo_diff_bpm": tempo_diff,
        "hnr_teacher_db": hnr_t,
        "hnr_student_db": hnr_s,
        "hnr_diff_db": hnr_diff,
        "jitter_teacher": teacher["jitter"],
        "jitter_student": student["jitter"],
        "shimmer_teacher": teacher["shimmer"],
        "shimmer_student": student["shimmer"],
        "voiced_ratio_diff": vr_diff,
        "attack_rise_teacher_ms": t_rise * 1000.0 if np.isfinite(t_rise) else np.nan,
        "attack_rise_student_ms": s_rise * 1000.0 if np.isfinite(s_rise) else np.nan,
        "attack_gain_teacher_db": t_gain,
        "attack_gain_student_db": s_gain,
        "vibrato_rate_teacher_hz": vib_t_rate,
        "vibrato_rate_student_hz": vib_s_rate,
        "vibrato_extent_teacher_hz": vib_t_ext,
        "vibrato_extent_student_hz": vib_s_ext,
        "silent_gaps_teacher_count": int(len(teacher["silent_gaps"])),
        "silent_gaps_student_count": int(len(student["silent_gaps"])),
        "silent_gap_mean_teacher_s": safe_mean(teacher["silent_gaps"]),
        "silent_gap_mean_student_s": safe_mean(student["silent_gaps"]),
        "alpha_ratio_teacher_db": teacher["alpha_ratio_db"],
        "alpha_ratio_student_db": student["alpha_ratio_db"],
        "alpha_ratio_diff_db": alpha_diff,
        "singer_formant_teacher_pct": teacher["singer_formant_pct"],
        "singer_formant_student_pct": student["singer_formant_pct"],
        "singer_formant_diff_pct": sf_diff,
        "spectral_tilt_teacher_db_oct": teacher["spectral_tilt_db_oct"],
        "spectral_tilt_student_db_oct": student["spectral_tilt_db_oct"],
        "cpp_mean_teacher": cpp_t,
        "cpp_mean_student": cpp_s,
        "h1h2_mean_teacher_db": h1h2_t,
        "h1h2_mean_student_db": h1h2_s,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Recommendation flags
# ─────────────────────────────────────────────────────────────────────────────


def build_flags(teacher, student, metrics):
    flags = []
    m = metrics
    if np.isfinite(m["pitch_mean_abs_cents"]) and m["pitch_mean_abs_cents"] > 50:
        d = "выше" if m["pitch_bias_cents"] > 0 else "ниже"
        flags.append(
            f"Интонация: средняя ошибка {fmt(m['pitch_mean_abs_cents'], 1)} cents, "
            f"общий сдвиг {d} эталона."
        )
    if np.isfinite(m["onset_mae_ms"]) and m["onset_mae_ms"] > 120:
        flags.append(
            f"Ритм: расхождение вступлений {fmt(m['onset_mae_ms'], 0)} ms; "
            "нужно точнее попадать во входы."
        )
    if np.isfinite(m["duration_mae_ms"]) and m["duration_mae_ms"] > 150:
        flags.append(
            f"Длительности: расхождение {fmt(m['duration_mae_ms'], 0)} ms; "
            "ноты удерживаются не как в эталоне."
        )
    if np.isfinite(m["attack_score"]) and m["attack_score"] < 70:
        flags.append(
            "Атака звука: характер начала нот отличается от учителя; "
            "проверить мягкость/резкость входа."
        )
    if np.isfinite(m["hnr_diff_db"]) and m["hnr_diff_db"] < -2:
        flags.append(
            f"Смыкание: HNR ученика ниже на {fmt(abs(m['hnr_diff_db']), 1)} dB — "
            "возможно больше придыхания."
        )
    if np.isfinite(m["cpp_mean_student"]) and m["cpp_mean_student"] < 15:
        flags.append(
            f"CPP: {fmt(m['cpp_mean_student'], 1)} dB — низкое значение, "
            "голос может звучать придыхательно или нечётко."
        )
    if np.isfinite(m["h1h2_mean_student_db"]):
        h = m["h1h2_mean_student_db"]
        if h > 8:
            flags.append(
                f"H1-H2 = {fmt(h, 1)} dB — открытое смыкание, возможно придыхание "
                "(характерно для sob / speechlike quality в Estill)."
            )
        elif h < -4:
            flags.append(
                f"H1-H2 = {fmt(h, 1)} dB — плотное смыкание / прессованный голос "
                "(belt-like качество)."
            )
    if np.isfinite(m["alpha_ratio_diff_db"]) and abs(m["alpha_ratio_diff_db"]) > 4:
        d = (
            "ярче (больше twang/мetal)"
            if m["alpha_ratio_diff_db"] > 0
            else "темнее (меньше яркости)"
        )
        flags.append(f"Тембр (alpha ratio): голос ученика {d}, чем у учителя.")
    if (
        np.isfinite(m["singer_formant_diff_pct"])
        and abs(m["singer_formant_diff_pct"]) > 1.5
    ):
        d = "больше" if m["singer_formant_diff_pct"] > 0 else "меньше"
        flags.append(
            f"Singer's formant: у ученика {d} энергии в зоне 2500–3500 Hz "
            f"({fmt(m['singer_formant_student_pct'], 1)} vs {fmt(m['singer_formant_teacher_pct'], 1)}%)."
        )
    return flags


# ─────────────────────────────────────────────────────────────────────────────
# Text report (for LLM input)
# ─────────────────────────────────────────────────────────────────────────────


def build_text_report(teacher_path, student_path, teacher, student, alignment, metrics):
    flags = build_flags(teacher, student, metrics)
    m = metrics
    lines = []
    lines += [
        "===== ТЕХНИЧЕСКИЙ ОТЧЁТ: АНАЛИЗ ГОЛОСОВОГО УПРАЖНЕНИЯ =====",
        f"Учитель: {teacher_path}",
        f"Ученик: {student_path}",
        f"Длительность: учитель {fmt(teacher['duration'])} s | ученик {fmt(student['duration'])} s",
        f"Tempo: учитель {fmt(teacher['tempo'], 1)} BPM | ученик {fmt(student['tempo'], 1)} BPM | diff {fmt(m['tempo_diff_bpm'], 1)} BPM",
        "",
        "----- Сводная оценка -----",
        f"Общий балл: {fmt(m['overall_score'], 1)}/100",
        f"Интонация: {fmt(m['intonation_score'], 1)}/100",
        f"Ритм: {fmt(m['rhythm_score'], 1)}/100",
        f"Атака звука: {fmt(m['attack_score'], 1)}/100",
        f"Дыхание/поддержка: {fmt(m['breath_score'], 1)}/100",
        f"Смыкание/голосовой контроль: {fmt(m['voice_closure_score'], 1)}/100",
        "",
        "----- Интонация (DTW) -----",
        f"Aligned voiced frames: {len(alignment['pitch_errors_cents'])}",
        f"DTW distance: {fmt(m.get('dtw_distance', alignment['dtw_distance']), 1)}",
        f"Mean abs pitch error: {fmt(m['pitch_mean_abs_cents'], 1)} cents",
        f"Median abs pitch error: {fmt(m['pitch_median_abs_cents'], 1)} cents",
        f"Pitch bias (student-teacher): {fmt(m['pitch_bias_cents'], 1)} cents",
        f"Pitch spread std: {fmt(m['pitch_std_cents'], 1)} cents",
        f"P90 abs error: {fmt(m['pitch_p90_abs_cents'], 1)} cents",
        f"In tune: ±25c {fmt(m['in_tune_25_pct'], 1)}% | ±50c {fmt(m['in_tune_50_pct'], 1)}% | ±100c {fmt(m['in_tune_100_pct'], 1)}%",
        "",
        "----- Ритм -----",
        f"Onsets: учитель {len(teacher['onsets'])} | ученик {len(student['onsets'])}",
        f"Onset MAE: {fmt(m['onset_mae_ms'], 1)} ms",
        f"Duration MAE: {fmt(m['duration_mae_ms'], 1)} ms",
        "",
        "----- Атака -----",
        f"Attack rise time: учитель {fmt(m['attack_rise_teacher_ms'], 1)} ms | ученик {fmt(m['attack_rise_student_ms'], 1)} ms",
        f"Attack gain: учитель {fmt(m['attack_gain_teacher_db'], 1)} dB | ученик {fmt(m['attack_gain_student_db'], 1)} dB",
        "",
        "----- Дыхание и поддержка -----",
        f"Voiced ratio: учитель {fmt(teacher['voiced_ratio'], 3)} | ученик {fmt(student['voiced_ratio'], 3)} | diff {fmt(m['voiced_ratio_diff'], 3)}",
        f"Silent gaps: учитель {m['silent_gaps_teacher_count']} | ученик {m['silent_gaps_student_count']}",
        f"Mean silent gap: учитель {fmt(m['silent_gap_mean_teacher_s'])} s | ученик {fmt(m['silent_gap_mean_student_s'])} s",
        "",
        "----- Смыкание голосовых складок (Praat) -----",
        f"HNR mean: учитель {fmt(m['hnr_teacher_db'])} dB | ученик {fmt(m['hnr_student_db'])} dB | diff {fmt(m['hnr_diff_db'])} dB",
        f"Jitter: учитель {fmt(m['jitter_teacher'], 4)} | ученик {fmt(m['jitter_student'], 4)}",
        f"Shimmer: учитель {fmt(m['shimmer_teacher'], 4)} | ученик {fmt(m['shimmer_student'], 4)}",
        "",
        "----- Вибрато -----",
        f"Vibrato rate: учитель {fmt(m['vibrato_rate_teacher_hz'])} Hz | ученик {fmt(m['vibrato_rate_student_hz'])} Hz",
        f"Vibrato extent: учитель {fmt(m['vibrato_extent_teacher_hz'])} Hz | ученик {fmt(m['vibrato_extent_student_hz'])} Hz",
        "",
        "----- Estill-признаки -----",
        f"CPP mean: учитель {fmt(m['cpp_mean_teacher'], 1)} dB | ученик {fmt(m['cpp_mean_student'], 1)} dB",
        "  (≥20 dB — чистый голос; <15 dB — возможно придыхательный)",
        f"H1-H2 mean: учитель {fmt(m['h1h2_mean_teacher_db'], 1)} dB | ученик {fmt(m['h1h2_mean_student_db'], 1)} dB",
        "  (>6 dB — открытое/придыхательное смыкание; <0 dB — плотное/прессованное)",
        f"Alpha ratio: учитель {fmt(m['alpha_ratio_teacher_db'], 1)} dB | ученик {fmt(m['alpha_ratio_student_db'], 1)} dB",
        "  (выше = ярче/twang; ниже = темнее/softer)",
        f"Singer's formant (2500–3500 Hz): учитель {fmt(m['singer_formant_teacher_pct'], 1)}% | ученик {fmt(m['singer_formant_student_pct'], 1)}%",
        f"Spectral tilt: учитель {fmt(m['spectral_tilt_teacher_db_oct'], 1)} dB/oct | ученик {fmt(m['spectral_tilt_student_db_oct'], 1)} dB/oct",
        "  (менее отрицательный = ярче/belt; более отрицательный = мягче/sob)",
        "",
        "----- Приоритетные выводы для LLM -----",
    ]
    if flags:
        for i, flag in enumerate(flags, 1):
            lines.append(f"{i}. {flag}")
    else:
        lines.append(
            "Критичных отклонений не найдено; фидбэк можно строить вокруг "
            "закрепления стабильного выполнения."
        )
    lines += [
        "",
        "----- Инструкция LLM -----",
        "Сформируй обратную связь на русском языке. "
        "Структура: общая картина → что хорошо → главные зоны роста (с объяснением) → "
        "3 упражнения → план на неделю → короткое резюме ученику. "
        "Используй Estill-терминологию, если уместно.",
    ]
    return "\n".join(lines)
