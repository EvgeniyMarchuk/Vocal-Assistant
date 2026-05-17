from __future__ import annotations

import librosa
import numpy as np

from .utils import hz_to_cents

# Penalty (cents) for aligning a voiced frame against an unvoiced one.
# Large enough to discourage mismatches but finite so DTW can still route around
# long unvoiced stretches.  500 cents = 5 semitones.
_UNVOICED_PENALTY = 500.0


def _build_cost_matrix(
    t_cents: np.ndarray,
    s_cents: np.ndarray,
    t_voiced: np.ndarray,
    s_voiced: np.ndarray,
    intensity_weight: float = 0.0,
    t_intensity: np.ndarray | None = None,
    s_intensity: np.ndarray | None = None,
) -> np.ndarray:
    """Build DTW cost matrix with proper voiced/unvoiced handling.

    voiced–voiced  : |cents_t - cents_s|  (+  intensity term if requested)
    voiced–unvoiced: _UNVOICED_PENALTY
    unvoiced–voiced: _UNVOICED_PENALTY
    unvoiced–unvoiced: 0   (both silent → perfect match)
    """
    N, M = len(t_cents), len(s_cents)
    cost = np.empty((N, M), dtype=np.float32)

    both_voiced = np.outer(t_voiced, s_voiced).astype(bool)
    t_only = np.outer(t_voiced, ~s_voiced).astype(bool)
    s_only = np.outer(~t_voiced, s_voiced).astype(bool)

    # voiced–voiced: |Δcents|
    tc = t_cents[:, None] * np.ones(M, dtype=np.float32)
    sc = s_cents[None, :] * np.ones(N, dtype=np.float32)[:, None]
    pitch_diff = np.abs(tc - sc)

    if intensity_weight > 0 and t_intensity is not None and s_intensity is not None:
        ti = t_intensity[:, None] * np.ones(M, dtype=np.float32)
        si = s_intensity[None, :] * np.ones(N, dtype=np.float32)[:, None]
        int_diff = np.abs(ti - si)
        cost = (1.0 - intensity_weight) * pitch_diff + intensity_weight * int_diff
    else:
        cost = pitch_diff

    cost[t_only | s_only] = _UNVOICED_PENALTY
    cost[~both_voiced & ~t_only & ~s_only] = 0.0  # both unvoiced

    return cost.astype(np.float64)


def align_by_pitch(
    teacher: dict,
    student: dict,
    intensity_weight: float = 0.0,
) -> dict:
    """DTW alignment on pitch contours with proper voiced/unvoiced handling.

    Args:
        teacher: feature dict from extract_features().
        student: feature dict from extract_features().
        intensity_weight: 0 = pitch-only; 0.3 = 70 % pitch + 30 % intensity.

    Returns:
        dict with path, teacher_idx, student_idx, pitch_errors_cents, dtw_distance.
    """
    t_f0 = teacher["f0"].astype(float)
    s_f0 = student["f0"].astype(float)
    t_voiced = np.isfinite(t_f0) & (t_f0 > 0)
    s_voiced = np.isfinite(s_f0) & (s_f0 > 0)

    t_cents = np.where(t_voiced, hz_to_cents(np.where(t_voiced, t_f0, 440.0)), 0.0)
    s_cents = np.where(s_voiced, hz_to_cents(np.where(s_voiced, s_f0, 440.0)), 0.0)

    # Normalise intensity to [0, 100] range for comparability with cents scale.
    t_int = s_int = None
    if intensity_weight > 0:
        ti = np.nan_to_num(teacher["intensity"].astype(float))
        si = np.nan_to_num(student["intensity"].astype(float))
        lo = min(ti.min(), si.min())
        hi = max(ti.max(), si.max()) + 1e-9
        t_int = (ti - lo) / (hi - lo) * 100.0
        s_int = (si - lo) / (hi - lo) * 100.0

    cost = _build_cost_matrix(
        t_cents, s_cents, t_voiced, s_voiced,
        intensity_weight=intensity_weight,
        t_intensity=t_int, s_intensity=s_int,
    )

    _, wp = librosa.sequence.dtw(C=cost)
    path = wp[::-1]

    t_idx, s_idx, errors = [], [], []
    for i, j in path:
        if t_voiced[i] and s_voiced[j]:
            t_idx.append(i)
            s_idx.append(j)
            errors.append(float(hz_to_cents(s_f0[j]) - hz_to_cents(t_f0[i])))

    dtw_dist = float(np.sum(np.abs(errors))) if errors else np.nan
    return {
        "path": path,
        "teacher_idx": np.array(t_idx, dtype=int),
        "student_idx": np.array(s_idx, dtype=int),
        "pitch_errors_cents": np.array(errors, dtype=float),
        "dtw_distance": dtw_dist,
    }
