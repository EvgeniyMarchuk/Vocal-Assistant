from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from rg_utils import fmt


def _save(fig, path):
    fig.savefig(str(path), dpi=130, bbox_inches="tight")
    plt.close(fig)


def save_all_plots(teacher, student, alignment, metrics, img_dir: Path) -> dict[str, str]:
    img_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    def rel(name):
        p = img_dir / name
        paths[name] = str(p)
        return p

    te, st = teacher, student
    ti = te["time"]
    si = st["time"]
    aln = alignment

    # 1. Pitch contours + onsets
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ti, te["f0"], color="tab:blue", lw=1.2, label="Teacher F0")
    ax.plot(si, st["f0"], color="tab:red", lw=1.1, alpha=0.8, label="Student F0")
    for t in te["onsets"][:50]:
        ax.axvline(t, color="tab:blue", alpha=0.15, lw=0.8)
    for t in st["onsets"][:50]:
        ax.axvline(t, color="tab:red", alpha=0.15, lw=0.8)
    ax.set_title("Pitch contours (F0) + onsets")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("F0 (Hz)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, rel("pitch_contours.png"))

    # 2. DTW pitch error over time
    fig, ax = plt.subplots(figsize=(14, 3))
    err = aln["pitch_errors_cents"]
    if aln["teacher_idx"].size:
        ax.plot(ti[aln["teacher_idx"]], err, color="tab:purple", lw=0.9)
        for val, col in [(25, "#2ca02c"), (-25, "#2ca02c"), (50, "#ff7f0e"), (-50, "#ff7f0e"),
                         (100, "#d62728"), (-100, "#d62728")]:
            ax.axhline(val, color=col, ls="--", lw=0.7, alpha=0.7)
    ax.axhline(0, color="black", lw=1.0)
    ax.set_title("DTW-aligned pitch error (student − teacher, cents)")
    ax.set_xlabel("Teacher time (s)")
    ax.set_ylabel("Error (cents)")
    ax.grid(alpha=0.25)
    _save(fig, rel("pitch_error.png"))

    # 3. Pitch error histogram
    fig, ax = plt.subplots(figsize=(7, 4))
    if err.size:
        ax.hist(err, bins=50, color="tab:purple", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="black", lw=1.0)
    ax.set_title("Pitch error distribution")
    ax.set_xlabel("Cents")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.25)
    _save(fig, rel("pitch_error_hist.png"))

    # 4. Log Mel spectrogram — teacher
    fig, ax = plt.subplots(figsize=(14, 4))
    img = ax.imshow(
        te["logmel"], aspect="auto", origin="lower",
        extent=[te["logmel_times"][0], te["logmel_times"][-1], 0, 128],
        cmap="magma",
    )
    plt.colorbar(img, ax=ax, label="dB")
    ax.set_title("Log Mel Spectrogram — Teacher")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    _save(fig, rel("logmel_teacher.png"))

    # 5. Log Mel spectrogram — student
    fig, ax = plt.subplots(figsize=(14, 4))
    vmin = min(te["logmel"].min(), st["logmel"].min())
    vmax = max(te["logmel"].max(), st["logmel"].max())
    img = ax.imshow(
        st["logmel"], aspect="auto", origin="lower",
        extent=[st["logmel_times"][0], st["logmel_times"][-1], 0, 128],
        cmap="magma", vmin=vmin, vmax=vmax,
    )
    plt.colorbar(img, ax=ax, label="dB")
    ax.set_title("Log Mel Spectrogram — Student")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mel bin")
    _save(fig, rel("logmel_student.png"))

    # 6. MFCC comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 4), sharey=True)
    for ax, feat, label, cmap in [
        (axes[0], te, "Teacher", "coolwarm"),
        (axes[1], st, "Student", "coolwarm"),
    ]:
        img = ax.imshow(
            feat["mfcc"], aspect="auto", origin="lower",
            extent=[feat["mfcc_times"][0], feat["mfcc_times"][-1], 0, 20],
            cmap=cmap,
        )
        plt.colorbar(img, ax=ax)
        ax.set_title(f"MFCC — {label}")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("MFCC coefficient")
    _save(fig, rel("mfcc_comparison.png"))

    # 7. Long-term average spectrum (LTAS)
    fig, ax = plt.subplots(figsize=(10, 4))
    f_t, l_t = te["ltas_freqs"], te["ltas_db"]
    f_s, l_s = st["ltas_freqs"], st["ltas_db"]
    valid_t = (f_t > 50) & (f_t < 8000)
    valid_s = (f_s > 50) & (f_s < 8000)
    ax.plot(f_t[valid_t], l_t[valid_t], color="tab:blue", lw=1.2, label="Teacher LTAS")
    ax.plot(f_s[valid_s], l_s[valid_s], color="tab:red", lw=1.1, alpha=0.85, label="Student LTAS")
    ax.axvspan(2500, 3500, alpha=0.08, color="green", label="Singer's formant (2.5–3.5 kHz)")
    ax.axvspan(1000, 5000, alpha=0.04, color="orange", label="Alpha ratio hi-band (1–5 kHz)")
    ax.set_title("Long-term Average Spectrum (LTAS)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("dB")
    ax.set_xscale("log")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, rel("ltas.png"))

    # 8. Formants F1/F2
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ti, te["f1"], color="#1b9e77", lw=1.0, label="Teacher F1")
    ax.plot(ti, te["f2"], color="#66a61e", lw=1.0, label="Teacher F2")
    ax.plot(si, st["f1"], color="#d95f02", lw=1.0, alpha=0.8, label="Student F1")
    ax.plot(si, st["f2"], color="#e7298a", lw=1.0, alpha=0.8, label="Student F2")
    ax.set_title("Formants F1 / F2")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    _save(fig, rel("formants.png"))

    # 9. Intensity dynamics
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(ti, te["intensity"], color="tab:blue", lw=1.1, label="Teacher intensity")
    ax.plot(si, st["intensity"], color="tab:red", lw=1.0, alpha=0.85, label="Student intensity")
    ax2 = ax.twinx()
    ax2.plot(ti, te["rms"], color="tab:cyan", lw=0.8, alpha=0.6, label="Teacher RMS")
    ax2.plot(si, st["rms"], color="tab:pink", lw=0.8, alpha=0.6, label="Student RMS")
    ax2.set_ylabel("RMS (linear)")
    ax.set_title("Intensity (dB) and RMS dynamics")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Intensity (dB)")
    lines1, lab1 = ax.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, lab1 + lab2, fontsize=8)
    ax.grid(alpha=0.25)
    _save(fig, rel("intensity_dynamics.png"))

    # 10. CPP contour + alpha ratio
    fig, axes = plt.subplots(2, 1, figsize=(14, 6), sharex=False)
    ax_cpp, ax_alpha = axes

    t_cpp_interp = np.interp(ti, te["cpp_times"], te["cpp_contour"])
    s_cpp_interp = np.interp(si, st["cpp_times"], st["cpp_contour"])
    ax_cpp.plot(ti, t_cpp_interp, color="tab:blue", lw=1.1, label="Teacher CPP")
    ax_cpp.plot(si, s_cpp_interp, color="tab:red", lw=1.0, alpha=0.85, label="Student CPP")
    ax_cpp.axhline(15, color="tab:orange", ls="--", lw=0.9, label="15 dB (clarity threshold)")
    ax_cpp.axhline(20, color="tab:green", ls="--", lw=0.9, label="20 dB (clear voice)")
    ax_cpp.set_title("CPP (Cepstral Peak Prominence) — голосовая чёткость")
    ax_cpp.set_ylabel("CPP (dB)")
    ax_cpp.legend(fontsize=8)
    ax_cpp.grid(alpha=0.25)

    t_alpha_interp = np.interp(ti, te["alpha_times"], te["alpha_contour"])
    s_alpha_interp = np.interp(si, st["alpha_times"], st["alpha_contour"])
    ax_alpha.plot(ti, t_alpha_interp, color="tab:blue", lw=1.1, label="Teacher Alpha ratio")
    ax_alpha.plot(si, s_alpha_interp, color="tab:red", lw=1.0, alpha=0.85, label="Student Alpha ratio")
    ax_alpha.set_title("Alpha ratio (log E[1–5 kHz] / E[50–1 kHz]) — яркость/twang")
    ax_alpha.set_xlabel("Time (s)")
    ax_alpha.set_ylabel("Alpha ratio (dB)")
    ax_alpha.legend(fontsize=8)
    ax_alpha.grid(alpha=0.25)
    _save(fig, rel("cpp_alpha.png"))

    # 11. Scores summary
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ["Общий", "Интонация", "Ритм", "Атака", "Дыхание", "Смыкание"]
    values = [
        metrics["overall_score"], metrics["intonation_score"], metrics["rhythm_score"],
        metrics["attack_score"], metrics["breath_score"], metrics["voice_closure_score"],
    ]
    colors = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#b07aa1"]
    bars = ax.bar(labels, values, color=colors, width=0.6)
    for bar, val in zip(bars, values):
        if np.isfinite(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.axhline(80, color="green", ls="--", lw=0.8, alpha=0.6, label="80 (хорошо)")
    ax.axhline(60, color="orange", ls="--", lw=0.8, alpha=0.6, label="60 (умеренно)")
    ax.set_title("Итоговые оценки по категориям")
    ax.set_ylabel("Балл / 100")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    _save(fig, rel("scores_summary.png"))

    return paths
