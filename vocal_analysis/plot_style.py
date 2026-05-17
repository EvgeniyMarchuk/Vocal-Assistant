"""Единый стиль и палитра для всех графиков отчёта."""

from __future__ import annotations

import math

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Цвета пары
TEACHER_COLOR = "#2E5C9E"  # глубокий синий
STUDENT_COLOR = "#E26A4F"  # коралловый
TEACHER_LABEL = "Эталон"
STUDENT_LABEL = "Ученик"

# Цвета зон точности интонации (центы)
GOOD_COLOR = "#2EA56A"  # ±25¢ — хорошо
OKAY_COLOR = "#E0A53B"  # ±50¢ — приемлемо
BAD_COLOR = "#D24D4D"  # ±100¢ — плохо

# Дополнительные цвета
ACCENT_COLOR = "#7C4DFF"  # фиолетовый — линии ошибок
NEUTRAL_GRID = "#D8D8E0"
TEXT_MUTED = "#5A5A65"

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def apply_style() -> None:
    """Применить унифицированный matplotlib-стиль."""
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "#2A2A30",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#2A2A30",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.titlepad": 10,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": NEUTRAL_GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "xtick.color": "#2A2A30",
            "ytick.color": "#2A2A30",
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.fancybox": True,
            "legend.edgecolor": NEUTRAL_GRID,
            "legend.fontsize": 10,
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "lines.linewidth": 1.4,
            "lines.solid_capstyle": "round",
        }
    )


def save(fig, path) -> None:
    fig.savefig(str(path), dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: pitch (Hz) ↔ ноты
# ─────────────────────────────────────────────────────────────────────────────


def hz_to_midi(hz: float) -> float:
    return 69.0 + 12.0 * math.log2(max(hz, 1e-6) / 440.0)


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69.0) / 12.0))


def midi_to_note_name(midi: int) -> str:
    name = NOTE_NAMES[midi % 12]
    octave = midi // 12 - 1
    return f"{name}{octave}"


def note_axis_ticks(fmin_hz: float, fmax_hz: float, step_semitones: int = 3):
    """Возвращает (positions_hz, labels) для оси Y с нотными подписями."""
    midi_lo = int(math.floor(hz_to_midi(fmin_hz)))
    midi_hi = int(math.ceil(hz_to_midi(fmax_hz)))
    # Округляем нижнюю границу вверх до ближайшего "C"-кратного шага
    midi_lo = midi_lo - (midi_lo % step_semitones)
    midis = list(range(midi_lo, midi_hi + 1, step_semitones))
    positions = [midi_to_hz(m) for m in midis]
    labels = [midi_to_note_name(m) for m in midis]
    return positions, labels


def semitone_grid(fmin_hz: float, fmax_hz: float):
    """Все полутона в диапазоне — для светлой подложки на pitch-графике."""
    midi_lo = int(math.floor(hz_to_midi(fmin_hz)))
    midi_hi = int(math.ceil(hz_to_midi(fmax_hz)))
    return [midi_to_hz(m) for m in range(midi_lo, midi_hi + 1)]


def voiced_only(values: np.ndarray) -> np.ndarray:
    """Заменить нули/отрицательные значения на NaN (чтобы plot не рисовал линию через паузы)."""
    arr = np.asarray(values, dtype=float).copy()
    arr[(arr <= 0) | ~np.isfinite(arr)] = np.nan
    return arr
