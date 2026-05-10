from __future__ import annotations

import librosa
import numpy as np
from scipy.spatial.distance import cdist

from rg_utils import hz_to_cents


def align_by_pitch(teacher: dict, student: dict) -> dict:
    tc = np.nan_to_num(hz_to_cents(teacher["f0"]), nan=0.0).reshape(1, -1)
    sc = np.nan_to_num(hz_to_cents(student["f0"]), nan=0.0).reshape(1, -1)
    cost = cdist(tc.T, sc.T, metric="euclidean")
    _, wp = librosa.sequence.dtw(C=cost)
    path = wp[::-1]

    t_idx, s_idx, errors = [], [], []
    for i, j in path:
        tf = teacher["f0"][i]
        sf = student["f0"][j]
        if np.isfinite(tf) and tf > 0 and np.isfinite(sf) and sf > 0:
            t_idx.append(i)
            s_idx.append(j)
            errors.append(float(hz_to_cents(sf) - hz_to_cents(tf)))

    dtw_dist = float(np.sum(np.abs(errors))) if errors else np.nan
    return {
        "path": path,
        "teacher_idx": np.array(t_idx, dtype=int),
        "student_idx": np.array(s_idx, dtype=int),
        "pitch_errors_cents": np.array(errors, dtype=float),
        "dtw_distance": dtw_dist,
    }
