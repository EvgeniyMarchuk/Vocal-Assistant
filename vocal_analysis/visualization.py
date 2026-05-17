"""Сборка PNG-визуализаций отчёта.

Все подписи на русском. Цвета и стиль фиксированы в plot_style.py:
эталон — синий, ученик — коралловый. Зоны точности интонации:
зелёный (±25¢), жёлтый (±50¢), красный (±100¢).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from . import plot_style as ps
from .plot_style import (
    BAD_COLOR,
    GOOD_COLOR,
    OKAY_COLOR,
    STUDENT_COLOR,
    STUDENT_LABEL,
    TEACHER_COLOR,
    TEACHER_LABEL,
    apply_style,
    note_axis_ticks,
    save,
    semitone_grid,
    voiced_only,
)


def save_all_plots(
    teacher, student, alignment, metrics, img_dir: Path
) -> dict[str, str]:
    apply_style()
    img_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}

    def out(name: str) -> Path:
        p = img_dir / name
        paths[name] = str(p)
        return p

    _plot_pitch_contours(teacher, student, out("pitch_contours.png"))
    _plot_pitch_error(teacher, alignment, out("pitch_error.png"))
    _plot_pitch_error_hist(alignment, metrics, out("pitch_error_hist.png"))
    _plot_logmel(teacher, student, "Эталон", out("logmel_teacher.png"))
    _plot_logmel(
        student, teacher, "Ученик", out("logmel_student.png"), pair_for_range=True
    )
    _plot_mfcc_comparison(teacher, student, out("mfcc_comparison.png"))
    _plot_ltas(teacher, student, out("ltas.png"))
    _plot_formants(teacher, student, out("formants.png"))
    _plot_intensity_dynamics(teacher, student, out("intensity_dynamics.png"))
    _plot_cpp_alpha(teacher, student, out("cpp_alpha.png"))
    _plot_scores_summary(metrics, out("scores_summary.png"))

    return paths


# ─────────────────────────────────────────────────────────────────────────────
# 1. Контуры F0 с нотной осью
# ─────────────────────────────────────────────────────────────────────────────


def _plot_pitch_contours(teacher, student, path: Path) -> None:
    t_f0 = voiced_only(teacher["f0"])
    s_f0 = voiced_only(student["f0"])

    finite = np.concatenate(
        [
            t_f0[np.isfinite(t_f0)],
            s_f0[np.isfinite(s_f0)],
        ]
    )
    if finite.size:
        f_lo = max(50.0, float(np.nanpercentile(finite, 1)) * 0.92)
        f_hi = float(np.nanpercentile(finite, 99)) * 1.10
    else:
        f_lo, f_hi = 100.0, 500.0
    f_lo = max(50.0, f_lo)
    f_hi = max(f_lo * 1.5, f_hi)

    fig, ax = plt.subplots(figsize=(14, 4.5))

    # Полутоновая сетка как лёгкая подложка
    for hz in semitone_grid(f_lo, f_hi):
        ax.axhline(hz, color="#E8E8EE", lw=0.5, zorder=0)

    ax.plot(
        teacher["time"],
        t_f0,
        color=TEACHER_COLOR,
        lw=1.6,
        label=f"{TEACHER_LABEL} F0",
        zorder=3,
    )
    ax.plot(
        student["time"],
        s_f0,
        color=STUDENT_COLOR,
        lw=1.4,
        alpha=0.9,
        label=f"{STUDENT_LABEL} F0",
        zorder=3,
    )

    # Онсеты как маркеры по краям оси (не вертикальные стены)
    trans = ax.get_xaxis_transform()
    if len(teacher["onsets"]) > 0:
        ax.plot(
            teacher["onsets"],
            np.full_like(teacher["onsets"], 1.0),
            marker="v",
            linestyle="",
            color=TEACHER_COLOR,
            markersize=7,
            transform=trans,
            clip_on=False,
            label=f"Онсеты ({TEACHER_LABEL.lower()})",
        )
    if len(student["onsets"]) > 0:
        ax.plot(
            student["onsets"],
            np.full_like(student["onsets"], 0.0),
            marker="^",
            linestyle="",
            color=STUDENT_COLOR,
            markersize=7,
            transform=trans,
            clip_on=False,
            label=f"Онсеты ({STUDENT_LABEL.lower()})",
        )

    ax.set_yscale("log")
    ax.set_ylim(f_lo, f_hi)
    ticks, labels = note_axis_ticks(f_lo, f_hi, step_semitones=3)
    ax.set_yticks(ticks)
    ax.set_yticklabels(
        [
            f"{name} ({int(round(hz))} Гц)"
            for name, hz in zip(labels, ticks, strict=False)
        ]
    )
    ax.minorticks_off()

    ax.set_title("Высота тона (F0): эталон vs ученик")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Нота / частота")
    ax.legend(loc="upper right", ncol=2)
    ax.grid(axis="x", alpha=0.5)
    save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Pitch error c цветными зонами и аннотацией mean/median
# ─────────────────────────────────────────────────────────────────────────────


def _add_cents_zones(ax, x_label_pos: float = 0.985):
    """Раскрашивает фон оси Y по зонам ±25/±50/±100¢ и добавляет ярлыки справа."""
    ax.axhspan(-25, 25, color=GOOD_COLOR, alpha=0.13, zorder=0)
    ax.axhspan(25, 50, color=OKAY_COLOR, alpha=0.13, zorder=0)
    ax.axhspan(-50, -25, color=OKAY_COLOR, alpha=0.13, zorder=0)
    ax.axhspan(50, 100, color=BAD_COLOR, alpha=0.10, zorder=0)
    ax.axhspan(-100, -50, color=BAD_COLOR, alpha=0.10, zorder=0)

    ax.axhline(0, color="#2A2A30", lw=1.0, zorder=1)
    for y, color in [
        (25, GOOD_COLOR),
        (-25, GOOD_COLOR),
        (50, OKAY_COLOR),
        (-50, OKAY_COLOR),
        (100, BAD_COLOR),
        (-100, BAD_COLOR),
    ]:
        ax.axhline(y, color=color, ls="--", lw=0.8, alpha=0.7, zorder=1)

    label_kwargs = {
        "transform": ax.get_yaxis_transform(),
        "ha": "right",
        "va": "center",
        "fontsize": 9,
        "fontweight": "bold",
    }
    ax.text(
        x_label_pos,
        0,
        "Хорошо ±25¢",
        color="#1F7E4F",
        bbox={
            "facecolor": "white",
            "edgecolor": GOOD_COLOR,
            "boxstyle": "round,pad=0.25",
            "alpha": 0.85,
        },
        **label_kwargs,
    )
    ax.text(
        x_label_pos,
        37.5,
        "Приемлемо ±50¢",
        color="#9F6B17",
        bbox={
            "facecolor": "white",
            "edgecolor": OKAY_COLOR,
            "boxstyle": "round,pad=0.25",
            "alpha": 0.85,
        },
        **label_kwargs,
    )
    ax.text(
        x_label_pos,
        75,
        "Плохо ±100¢",
        color="#9F2C2C",
        bbox={
            "facecolor": "white",
            "edgecolor": BAD_COLOR,
            "boxstyle": "round,pad=0.25",
            "alpha": 0.85,
        },
        **label_kwargs,
    )


def _plot_pitch_error(teacher, alignment, path: Path) -> None:
    err = alignment["pitch_errors_cents"]
    t = teacher["time"]
    fig, ax = plt.subplots(figsize=(14, 4.2))

    if err.size and alignment["teacher_idx"].size:
        x = t[alignment["teacher_idx"]]
        ax.plot(x, err, color=ps.ACCENT_COLOR, lw=1.0, alpha=0.9, zorder=3)

        mean_e = float(np.mean(err))
        median_e = float(np.median(err))
        mae = float(np.mean(np.abs(err)))

        ax.axhline(
            mean_e,
            color=ps.ACCENT_COLOR,
            ls="--",
            lw=1.0,
            alpha=0.85,
            label=f"Среднее = {mean_e:+.1f}¢",
        )
        ax.axhline(
            median_e,
            color="#3A3A45",
            ls=":",
            lw=1.0,
            alpha=0.85,
            label=f"Медиана = {median_e:+.1f}¢",
        )
        ax.legend(loc="upper left")

        ax.text(
            0.012,
            0.97,
            f"MAE = {mae:.1f}¢   |   N = {err.size} кадров",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=ps.TEXT_MUTED,
            bbox={
                "facecolor": "white",
                "edgecolor": ps.NEUTRAL_GRID,
                "boxstyle": "round,pad=0.35",
            },
        )

    _add_cents_zones(ax)

    y_max = 130
    if err.size:
        y_max = max(130.0, float(np.percentile(np.abs(err), 99)) * 1.05)
    ax.set_ylim(-y_max, y_max)

    ax.set_title("DTW-выравненная ошибка интонации (ученик − эталон)")
    ax.set_xlabel("Время эталона (с)")
    ax.set_ylabel("Отклонение (центы)")
    ax.grid(axis="x", alpha=0.5)
    save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Гистограмма ошибки с зонами и MAE/median
# ─────────────────────────────────────────────────────────────────────────────


def _plot_pitch_error_hist(alignment, metrics, path: Path) -> None:
    err = alignment["pitch_errors_cents"]
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Цветные зоны как фон
    ax.axvspan(-25, 25, color=GOOD_COLOR, alpha=0.13, zorder=0)
    ax.axvspan(25, 50, color=OKAY_COLOR, alpha=0.13, zorder=0)
    ax.axvspan(-50, -25, color=OKAY_COLOR, alpha=0.13, zorder=0)
    ax.axvspan(50, 100, color=BAD_COLOR, alpha=0.10, zorder=0)
    ax.axvspan(-100, -50, color=BAD_COLOR, alpha=0.10, zorder=0)

    if err.size:
        clip = float(max(150.0, np.percentile(np.abs(err), 99) * 1.1))
        clipped = np.clip(err, -clip, clip)
        ax.hist(
            clipped,
            bins=50,
            color=ps.ACCENT_COLOR,
            alpha=0.78,
            edgecolor="white",
            zorder=2,
        )
        ax.set_xlim(-clip, clip)

        mae = float(np.mean(np.abs(err)))
        median = float(np.median(err))
        ax.axvline(0, color="#2A2A30", lw=1.0, zorder=3)
        ax.axvline(
            median,
            color="#3A3A45",
            ls=":",
            lw=1.4,
            label=f"Медиана = {median:+.1f}¢",
            zorder=3,
        )
        ax.axvline(
            mae, color=BAD_COLOR, ls="--", lw=1.4, label=f"MAE = {mae:.1f}¢", zorder=3
        )
        ax.axvline(-mae, color=BAD_COLOR, ls="--", lw=1.4, zorder=3)

        # Подпись % в зонах
        in25 = metrics.get("in_tune_25_pct")
        in50 = metrics.get("in_tune_50_pct")
        in100 = metrics.get("in_tune_100_pct")
        info_lines = [f"N = {err.size} кадров"]
        if in25 is not None and np.isfinite(in25):
            info_lines.append(f"В ±25¢: {in25:.1f}%")
        if in50 is not None and np.isfinite(in50):
            info_lines.append(f"В ±50¢: {in50:.1f}%")
        if in100 is not None and np.isfinite(in100):
            info_lines.append(f"В ±100¢: {in100:.1f}%")
        ax.text(
            0.985,
            0.97,
            "\n".join(info_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9.5,
            fontweight="bold",
            color=ps.TEXT_MUTED,
            bbox={
                "facecolor": "white",
                "edgecolor": ps.NEUTRAL_GRID,
                "boxstyle": "round,pad=0.35",
            },
        )
        ax.legend(loc="upper left")

    ax.set_title("Распределение ошибки интонации")
    ax.set_xlabel("Отклонение (центы)")
    ax.set_ylabel("Кол-во кадров")
    save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 4–5. Лог-мел спектрограммы
# ─────────────────────────────────────────────────────────────────────────────


def _plot_logmel(
    primary, other, role: str, path: Path, pair_for_range: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    if pair_for_range:
        vmin = min(primary["logmel"].min(), other["logmel"].min())
        vmax = max(primary["logmel"].max(), other["logmel"].max())
    else:
        vmin = min(primary["logmel"].min(), other["logmel"].min())
        vmax = max(primary["logmel"].max(), other["logmel"].max())

    img = ax.imshow(
        primary["logmel"],
        aspect="auto",
        origin="lower",
        extent=[primary["logmel_times"][0], primary["logmel_times"][-1], 0, 128],
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
    )
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Уровень (дБ)")
    ax.set_title(f"Лог-мел спектрограмма — {role}")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Мел-канал")
    ax.grid(False)
    save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 6. MFCC сравнение
# ─────────────────────────────────────────────────────────────────────────────


def _plot_mfcc_comparison(teacher, student, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)
    vmin = min(teacher["mfcc"].min(), student["mfcc"].min())
    vmax = max(teacher["mfcc"].max(), student["mfcc"].max())

    for ax, feat, label in [
        (axes[0], teacher, TEACHER_LABEL),
        (axes[1], student, STUDENT_LABEL),
    ]:
        img = ax.imshow(
            feat["mfcc"],
            aspect="auto",
            origin="lower",
            extent=[feat["mfcc_times"][0], feat["mfcc_times"][-1], 0, 20],
            cmap="coolwarm",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = plt.colorbar(img, ax=ax)
        cbar.set_label("Амплитуда")
        ax.set_title(f"MFCC — {label}")
        ax.set_xlabel("Время (с)")
        ax.set_ylabel("Коэффициент MFCC")
        ax.grid(False)
    save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 7. LTAS
# ─────────────────────────────────────────────────────────────────────────────


def _plot_ltas(teacher, student, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 4.2))
    f_t, l_t = teacher["ltas_freqs"], teacher["ltas_db"]
    f_s, l_s = student["ltas_freqs"], student["ltas_db"]
    valid_t = (f_t > 50) & (f_t < 8000)
    valid_s = (f_s > 50) & (f_s < 8000)

    ax.axvspan(
        2500, 3500, alpha=0.10, color="#2EA56A", label="Singer's formant (2.5–3.5 кГц)"
    )
    ax.axvspan(
        1000,
        5000,
        alpha=0.06,
        color="#E0A53B",
        label="Alpha ratio: верхняя полоса (1–5 кГц)",
    )

    ax.plot(
        f_t[valid_t],
        l_t[valid_t],
        color=TEACHER_COLOR,
        lw=1.5,
        label=f"{TEACHER_LABEL}",
    )
    ax.plot(
        f_s[valid_s],
        l_s[valid_s],
        color=STUDENT_COLOR,
        lw=1.4,
        alpha=0.9,
        label=f"{STUDENT_LABEL}",
    )
    ax.set_xscale("log")
    ax.set_xlim(60, 8000)
    ax.set_title("Долгосрочный средний спектр (LTAS)")
    ax.set_xlabel("Частота (Гц, лог)")
    ax.set_ylabel("Уровень (дБ)")
    ax.legend(loc="lower left", fontsize=9)
    save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Форманты F1 / F2
# ─────────────────────────────────────────────────────────────────────────────


def _plot_formants(teacher, student, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        teacher["time"],
        voiced_only(teacher["f1"]),
        color=TEACHER_COLOR,
        lw=1.2,
        label=f"{TEACHER_LABEL} F1",
    )
    ax.plot(
        teacher["time"],
        voiced_only(teacher["f2"]),
        color=TEACHER_COLOR,
        lw=1.2,
        ls="--",
        alpha=0.85,
        label=f"{TEACHER_LABEL} F2",
    )
    ax.plot(
        student["time"],
        voiced_only(student["f1"]),
        color=STUDENT_COLOR,
        lw=1.2,
        alpha=0.95,
        label=f"{STUDENT_LABEL} F1",
    )
    ax.plot(
        student["time"],
        voiced_only(student["f2"]),
        color=STUDENT_COLOR,
        lw=1.2,
        ls="--",
        alpha=0.85,
        label=f"{STUDENT_LABEL} F2",
    )
    ax.set_title("Форманты F1 / F2 во времени")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Частота форманты (Гц)")
    ax.legend(ncol=2, loc="upper right")
    save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Динамика интенсивности и RMS
# ─────────────────────────────────────────────────────────────────────────────


def _plot_intensity_dynamics(teacher, student, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(
        teacher["time"],
        teacher["intensity"],
        color=TEACHER_COLOR,
        lw=1.4,
        label=f"{TEACHER_LABEL} — интенсивность",
    )
    ax.plot(
        student["time"],
        student["intensity"],
        color=STUDENT_COLOR,
        lw=1.3,
        alpha=0.9,
        label=f"{STUDENT_LABEL} — интенсивность",
    )

    ax2 = ax.twinx()
    ax2.plot(
        teacher["time"],
        teacher["rms"],
        color=TEACHER_COLOR,
        lw=0.9,
        alpha=0.45,
        ls=":",
        label=f"{TEACHER_LABEL} — RMS",
    )
    ax2.plot(
        student["time"],
        student["rms"],
        color=STUDENT_COLOR,
        lw=0.9,
        alpha=0.45,
        ls=":",
        label=f"{STUDENT_LABEL} — RMS",
    )
    ax2.set_ylabel("RMS (отн. ед.)")
    ax2.grid(False)

    ax.set_title("Динамика громкости: интенсивность (дБ) и RMS")
    ax.set_xlabel("Время (с)")
    ax.set_ylabel("Интенсивность (дБ)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", ncol=2, fontsize=9)
    save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 10. CPP + Alpha ratio
# ─────────────────────────────────────────────────────────────────────────────


def _plot_cpp_alpha(teacher, student, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 6.5), sharex=False)
    ax_cpp, ax_alpha = axes

    t_cpp = np.interp(teacher["time"], teacher["cpp_times"], teacher["cpp_contour"])
    s_cpp = np.interp(student["time"], student["cpp_times"], student["cpp_contour"])
    ax_cpp.plot(
        teacher["time"],
        t_cpp,
        color=TEACHER_COLOR,
        lw=1.4,
        label=f"{TEACHER_LABEL} CPP",
    )
    ax_cpp.plot(
        student["time"],
        s_cpp,
        color=STUDENT_COLOR,
        lw=1.3,
        alpha=0.9,
        label=f"{STUDENT_LABEL} CPP",
    )
    ax_cpp.axhline(
        15, color=OKAY_COLOR, ls="--", lw=1.0, label="15 дБ — порог чёткости"
    )
    ax_cpp.axhline(20, color=GOOD_COLOR, ls="--", lw=1.0, label="20 дБ — чистый голос")
    ax_cpp.text(
        0.985,
        0.05,
        "ниже — придыхательный  /  выше — чёткий",
        transform=ax_cpp.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=ps.TEXT_MUTED,
        style="italic",
    )
    ax_cpp.set_title("CPP — чёткость голоса (Cepstral Peak Prominence)")
    ax_cpp.set_ylabel("CPP (дБ)")
    ax_cpp.set_xlabel("Время (с)")
    ax_cpp.legend(loc="upper right", ncol=2, fontsize=9)

    t_alpha = np.interp(
        teacher["time"], teacher["alpha_times"], teacher["alpha_contour"]
    )
    s_alpha = np.interp(
        student["time"], student["alpha_times"], student["alpha_contour"]
    )
    ax_alpha.plot(
        teacher["time"],
        t_alpha,
        color=TEACHER_COLOR,
        lw=1.4,
        label=f"{TEACHER_LABEL} Alpha ratio",
    )
    ax_alpha.plot(
        student["time"],
        s_alpha,
        color=STUDENT_COLOR,
        lw=1.3,
        alpha=0.9,
        label=f"{STUDENT_LABEL} Alpha ratio",
    )
    ax_alpha.text(
        0.985,
        0.05,
        "выше — twang / ярче  /  ниже — sob / темнее",
        transform=ax_alpha.transAxes,
        ha="right",
        va="bottom",
        fontsize=9,
        color=ps.TEXT_MUTED,
        style="italic",
    )
    ax_alpha.set_title("Alpha ratio — яркость тембра (lg E[1–5 кГц] / E[50 Гц–1 кГц])")
    ax_alpha.set_xlabel("Время (с)")
    ax_alpha.set_ylabel("Alpha ratio (дБ)")
    ax_alpha.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    save(fig, path)


# ─────────────────────────────────────────────────────────────────────────────
# 11. Сводная диаграмма оценок
# ─────────────────────────────────────────────────────────────────────────────


def _plot_scores_summary(metrics, path: Path) -> None:
    labels = ["Общий", "Интонация", "Ритм", "Атака", "Дыхание", "Смыкание"]
    values = [
        metrics["overall_score"],
        metrics["intonation_score"],
        metrics["rhythm_score"],
        metrics["attack_score"],
        metrics["breath_score"],
        metrics["voice_closure_score"],
    ]

    def color_for(v: float) -> str:
        if not np.isfinite(v):
            return ps.NEUTRAL_GRID
        if v >= 80:
            return GOOD_COLOR
        if v >= 60:
            return OKAY_COLOR
        return BAD_COLOR

    colors = [color_for(v) for v in values]

    fig, ax = plt.subplots(figsize=(9, 4.6))

    ax.axvspan(0, 60, color=BAD_COLOR, alpha=0.07, zorder=0)
    ax.axvspan(60, 80, color=OKAY_COLOR, alpha=0.08, zorder=0)
    ax.axvspan(80, 100, color=GOOD_COLOR, alpha=0.08, zorder=0)

    y_pos = np.arange(len(labels))[::-1]  # сверху вниз: «Общий» вверху
    plot_vals = [v if np.isfinite(v) else 0 for v in values]

    bars = ax.barh(
        y_pos, plot_vals, color=colors, edgecolor="white", height=0.65, zorder=2
    )

    for bar, val in zip(bars, values, strict=False):
        if np.isfinite(val):
            ax.text(
                min(val + 1.5, 100.5),
                bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}",
                va="center",
                ha="left",
                fontsize=11,
                fontweight="bold",
                color="#2A2A30",
            )
        else:
            ax.text(
                2,
                bar.get_y() + bar.get_height() / 2,
                "n/a",
                va="center",
                ha="left",
                fontsize=10,
                color=ps.TEXT_MUTED,
            )

    ax.axvline(60, color=OKAY_COLOR, ls="--", lw=0.9, alpha=0.7, zorder=1)
    ax.axvline(80, color=GOOD_COLOR, ls="--", lw=0.9, alpha=0.7, zorder=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlim(0, 110)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("Балл (из 100)")
    ax.set_title("Итоговые оценки по категориям")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.45)

    ax.text(
        60,
        len(labels) - 0.3,
        "≥60 приемлемо",
        color="#9F6B17",
        fontsize=8.5,
        ha="center",
        fontweight="bold",
    )
    ax.text(
        80,
        len(labels) - 0.3,
        "≥80 хорошо",
        color="#1F7E4F",
        fontsize=8.5,
        ha="center",
        fontweight="bold",
    )

    save(fig, path)
