from __future__ import annotations

import os

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
PITCH_FLOOR = 75
PITCH_CEILING = 600
TIME_STEP = 0.01
REF_HZ = 440.0

OLLAMA_MODEL = os.environ.get("VOCAL_FEEDBACK_MODEL", "qwen3:4b")
OLLAMA_CHAT_URL = os.environ.get("OLLAMA_CHAT_URL", "http://127.0.0.1:11434/api/chat")

# ─────────────────────────────────────────────────────────────────────────────
# Math helpers
# ─────────────────────────────────────────────────────────────────────────────

def safe_mean(v):
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.mean(a)) if a.size else np.nan


def safe_median(v):
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else np.nan


def safe_std(v):
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.std(a)) if a.size else np.nan


def safe_pct(v, q):
    a = np.asarray(v, dtype=float)
    a = a[np.isfinite(a)]
    return float(np.percentile(a, q)) if a.size else np.nan


def clamp(v, lo=0.0, hi=100.0):
    return float(np.clip(v, lo, hi))


def fmt(v, d=2, empty="n/a"):
    if v is None or not np.isfinite(float(v)):
        return empty
    return f"{float(v):.{d}f}"


def hz_to_cents(f0, ref=REF_HZ):
    f0 = np.asarray(f0, dtype=float)
    return 1200.0 * np.log2(np.maximum(f0, 1e-6) / ref)


def moving_average(v, w):
    v = np.asarray(v, dtype=float)
    w = max(1, int(w))
    if w <= 1:
        return v.copy()
    k = np.ones(w) / w
    left = w // 2
    right = w - 1 - left
    return np.convolve(np.pad(v, (left, right), mode="edge"), k, mode="valid")


def estimate_tempo(onsets, lo=0.18, hi=8.0):
    o = np.asarray(onsets, dtype=float)
    if o.size < 2:
        return np.nan
    iv = np.diff(o)
    iv = iv[(iv >= lo) & (iv <= hi)]
    return float(60.0 / np.median(iv)) if iv.size else np.nan
