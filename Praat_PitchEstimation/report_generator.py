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
    report_<hash>.md         — full Markdown report with embedded images
    img/                     — PNG visualizations
    report_<hash>.json       — raw metrics
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import urllib.error
import urllib.request
import warnings
from pathlib import Path

import subprocess
import tempfile

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from parselmouth.praat import call as praat_call
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.stats import linregress

warnings.filterwarnings("ignore")

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

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


# ─────────────────────────────────────────────────────────────────────────────
# Onset detection (from notebook — robust Praat intensity rises)
# ─────────────────────────────────────────────────────────────────────────────

def detect_intensity_onsets(
    intensity_time,
    intensity_values,
    min_interval_s=0.4,
    smooth_s=0.04,
    pre_window_s=0.16,
    post_window_s=0.18,
    min_gain_db=5.0,
    min_active_db=None,
):
    t = np.asarray(intensity_time, dtype=float)
    x = np.asarray(intensity_values, dtype=float)
    valid = np.isfinite(t) & np.isfinite(x)
    t, x = t[valid], x[valid]
    if t.size < 3:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(t)
    t, x = t[order], x[order]
    dt = float(np.median(np.diff(t))) if t.size > 1 else TIME_STEP
    if not np.isfinite(dt) or dt <= 0:
        dt = TIME_STEP

    sf = max(1, int(round(smooth_s / dt)))
    if sf % 2 == 0:
        sf += 1
    xs = moving_average(x, sf)

    lo = safe_pct(xs, 10)
    hi = safe_pct(xs, 90)
    dr = hi - lo if np.isfinite(hi - lo) else 0.0
    gain_thr = max(min_gain_db, min(12.0, 0.22 * dr))
    act_thr = min_active_db if min_active_db is not None else lo + max(8.0, 0.28 * dr)

    pos_diff = np.maximum(np.diff(xs, prepend=xs[0]), 0.0)
    novelty = moving_average(pos_diff, max(1, int(round(0.03 / dt))))
    noise = safe_median(novelty)
    mad = safe_median(np.abs(novelty - noise))
    pk_h = max(noise + 3.0 * mad, safe_pct(novelty, 75))
    pk_p = max(0.08, 3.0 * mad)
    min_dist = max(1, int(round(min_interval_s / dt)))

    peaks, _ = find_peaks(novelty, height=pk_h, prominence=pk_p, distance=min_dist)

    onsets = []
    pre_f = max(1, int(round(pre_window_s / dt)))
    post_f = max(1, int(round(post_window_s / dt)))
    for peak in peaks:
        pre0 = max(0, peak - pre_f)
        post1 = min(xs.size, peak + post_f + 1)
        if post1 <= peak or peak <= pre0:
            continue
        base = safe_pct(xs[pre0:peak], 20)
        pk_val = safe_pct(xs[peak:post1], 90)
        if not np.isfinite(base) or not np.isfinite(pk_val):
            continue
        if pk_val - base < gain_thr or pk_val < act_thr:
            continue
        thr = base + 0.25 * (pk_val - base)
        local = xs[pre0:peak + 1]
        above = np.where(local >= thr)[0]
        oi = pre0 + int(above[0]) if above.size else peak
        if onsets and t[oi] - onsets[-1] < min_interval_s:
            if novelty[oi] > novelty[np.searchsorted(t, onsets[-1])]:
                onsets[-1] = float(t[oi])
            continue
        onsets.append(float(t[oi]))

    return np.asarray(onsets, dtype=float), novelty


# ─────────────────────────────────────────────────────────────────────────────
# Attack metrics (from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def attack_metrics(time, intensity, onsets, window_sec=0.18):
    rows = []
    for onset in onsets:
        pre = (time >= onset - 0.05) & (time < onset)
        post = (time >= onset) & (time <= onset + window_sec)
        if pre.sum() < 2 or post.sum() < 3:
            continue
        base = safe_pct(intensity[pre], 20)
        peak = safe_pct(intensity[post], 90)
        target = base + 0.8 * (peak - base)
        pt = time[post]
        pi = intensity[post]
        reached = np.where(pi >= target)[0]
        rt = float(pt[reached[0]] - onset) if reached.size else np.nan
        rows.append({
            "onset_s": float(onset),
            "attack_gain_db": float(peak - base),
            "attack_rise_time_s": rt,
        })
    return rows


def attack_summary(rows):
    if not rows:
        return np.nan, np.nan
    gains = [r["attack_gain_db"] for r in rows if np.isfinite(r["attack_gain_db"])]
    rises = [r["attack_rise_time_s"] for r in rows if np.isfinite(r["attack_rise_time_s"])]
    return safe_median(rises), safe_median(gains)


# ─────────────────────────────────────────────────────────────────────────────
# Vibrato (from notebook)
# ─────────────────────────────────────────────────────────────────────────────

def vibrato_metrics(f0, time):
    valid = np.isfinite(f0)
    if valid.sum() < 10:
        return np.nan, np.nan
    frame_rate = 1.0 / np.median(np.diff(time)) if time.size > 1 else 100.0
    fc = np.nan_to_num(f0, nan=safe_median(f0))
    peaks, _ = find_peaks(fc, distance=max(1, int(frame_rate / 12)))
    if len(peaks) < 2:
        return np.nan, safe_std(f0[valid])
    periods = np.diff(time[peaks])
    periods = periods[periods > 0]
    rate = float(1.0 / np.mean(periods)) if periods.size else np.nan
    return rate, safe_std(f0[valid])


# ─────────────────────────────────────────────────────────────────────────────
# Estill-specific features
# ─────────────────────────────────────────────────────────────────────────────

def compute_cpp_contour(y, sr, fmin=75, fmax=600, n_fft=2048, hop_length=None):
    """Frame-wise Cepstral Peak Prominence via cepstrum (librosa-based)."""
    if hop_length is None:
        hop_length = max(1, int(TIME_STEP * sr))
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length, window="hann")
    S_pow = np.abs(D) ** 2
    log_S = 10.0 * np.log10(S_pow + 1e-10)
    cep = np.real(np.fft.ifft(log_S, axis=0))
    n_bins = cep.shape[0]
    q_min = max(1, int(sr / fmax))
    q_max = min(n_bins // 2, int(sr / fmin))
    cpp_vals = []
    for fi in range(cep.shape[1]):
        frame = cep[:, fi]
        region = frame[q_min:q_max]
        if region.size < 2:
            cpp_vals.append(np.nan)
            continue
        peak_rel = int(np.argmax(region))
        peak_abs = q_min + peak_rel
        peak_val = frame[peak_abs]
        qs = np.arange(1, q_max + 1)
        fs = frame[1:q_max + 1]
        if qs.size >= 2:
            slope, intercept = np.polyfit(qs, fs, 1)
            trend = slope * peak_abs + intercept
            cpp_vals.append(float(max(0.0, peak_val - trend)))
        else:
            cpp_vals.append(float(max(0.0, peak_val)))
    times = librosa.frames_to_time(np.arange(len(cpp_vals)), sr=sr, hop_length=hop_length)
    return times, np.array(cpp_vals, dtype=float)


def compute_h1h2_contour(y, sr, f0, f0_times, n_fft=2048, hop_length=None):
    """Frame-wise H1-H2 (first minus second harmonic, dB). Higher = breathier."""
    if hop_length is None:
        hop_length = max(1, int(TIME_STEP * sr))
    D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
    S = np.abs(D)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    frame_times = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop_length)
    h1h2 = []
    for fi, t in enumerate(frame_times):
        f0_here = float(np.interp(t, f0_times, np.nan_to_num(f0, nan=0.0)))
        if f0_here < PITCH_FLOOR or f0_here > PITCH_CEILING:
            h1h2.append(np.nan)
            continue
        h1_idx = int(np.argmin(np.abs(freqs - f0_here)))
        h2_idx = int(np.argmin(np.abs(freqs - 2.0 * f0_here)))
        h1_a = S[h1_idx, fi]
        h2_a = S[h2_idx, fi]
        if h1_a > 0 and h2_a > 0:
            h1h2.append(20.0 * np.log10(h1_a / h2_a))
        else:
            h1h2.append(np.nan)
    return frame_times, np.array(h1h2, dtype=float)


def compute_spectral_features(y, sr, n_fft=4096, hop_length=512):
    """
    Returns per-file scalars:
      alpha_ratio_db    — 10*log10(E[1k-5kHz] / E[50-1kHz])
      singer_formant_pct — % energy in 2500-3500 Hz
      spectral_tilt_db_oct — slope of LTAS in dB/octave
    And time-varying arrays:
      alpha_times, alpha_contour
    """
    D = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    # Long-term average spectrum (mean amplitude per bin)
    ltas = np.mean(D, axis=1)
    ltas_db = 20.0 * np.log10(ltas + 1e-10)

    # Alpha ratio (scalar)
    lo_mask = (freqs >= 50) & (freqs <= 1000)
    hi_mask = (freqs >= 1000) & (freqs <= 5000)
    e_lo = np.sum(D[lo_mask] ** 2, axis=None)
    e_hi = np.sum(D[hi_mask] ** 2, axis=None)
    alpha_ratio = float(10.0 * np.log10(e_hi / e_lo)) if e_lo > 0 else np.nan

    # Singer's formant energy (scalar)
    sf_mask = (freqs >= 2500) & (freqs <= 3500)
    total_e = np.sum(D ** 2)
    sf_e = np.sum(D[sf_mask] ** 2)
    singer_formant_pct = float(100.0 * sf_e / total_e) if total_e > 0 else np.nan

    # Spectral tilt via linear regression on LTAS (dB vs log2-frequency)
    valid = (freqs > 50) & np.isfinite(ltas_db)
    if valid.sum() >= 5:
        slope, *_ = linregress(np.log2(freqs[valid]), ltas_db[valid])
        spectral_tilt = float(slope)
    else:
        spectral_tilt = np.nan

    # Time-varying alpha ratio
    e_lo_t = np.sum(D[lo_mask, :] ** 2, axis=0) + 1e-20
    e_hi_t = np.sum(D[hi_mask, :] ** 2, axis=0) + 1e-20
    alpha_contour = 10.0 * np.log10(e_hi_t / e_lo_t)
    alpha_times = librosa.frames_to_time(np.arange(D.shape[1]), sr=sr, hop_length=hop_length)

    return {
        "alpha_ratio_db": alpha_ratio,
        "singer_formant_pct": singer_formant_pct,
        "spectral_tilt_db_oct": spectral_tilt,
        "ltas_freqs": freqs,
        "ltas_db": ltas_db,
        "alpha_times": alpha_times,
        "alpha_contour": alpha_contour,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Full feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def _ensure_wav(audio_path: str) -> tuple[str, bool]:
    """Return (wav_path, is_tmp). Converts non-WAV files via ffmpeg."""
    if Path(audio_path).suffix.lower() == ".wav":
        return audio_path, False
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", audio_path, "-ar", "16000", "-ac", "1",
         "-sample_fmt", "s16", tmp.name],
        check=True, capture_output=True,
    )
    return tmp.name, True


def extract_features(audio_path: str) -> dict:
    audio_path = str(audio_path)
    wav_path, is_tmp = _ensure_wav(audio_path)
    try:
        snd = parselmouth.Sound(wav_path)
        y, sr = librosa.load(wav_path, sr=None, mono=True)
        duration = float(snd.get_total_duration())

        # Pitch / F0
        pitch = snd.to_pitch(time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        time = pitch.xs()
        f0 = pitch.selected_array["frequency"].astype(float)
        f0[f0 <= 0] = np.nan
        voiced_mask = np.isfinite(f0)

        # Intensity
        intensity = snd.to_intensity(time_step=TIME_STEP, minimum_pitch=PITCH_FLOOR)
        int_time = intensity.xs()
        int_vals = intensity.values[0].astype(float)
        int_interp = np.interp(time, int_time, int_vals)

        # HNR
        harmonicity = snd.to_harmonicity_cc(time_step=TIME_STEP, minimum_pitch=PITCH_FLOOR)
        hnr_time = harmonicity.xs()
        hnr_vals = harmonicity.values[0].astype(float)
        hnr_vals[hnr_vals <= -200] = np.nan
        vh = np.isfinite(hnr_vals)
        hnr_interp = np.interp(time, hnr_time[vh], hnr_vals[vh]) if vh.sum() >= 2 else np.full_like(time, np.nan)

        # Formants
        formant = snd.to_formant_burg(time_step=TIME_STEP)
        f1 = np.array([formant.get_value_at_time(1, t) for t in time], dtype=float)
        f2 = np.array([formant.get_value_at_time(2, t) for t in time], dtype=float)
        f3 = np.array([formant.get_value_at_time(3, t) for t in time], dtype=float)

        # Jitter / Shimmer
        pp = praat_call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        jitter = float(praat_call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        shimmer = float(praat_call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

        # Onsets (intensity-based)
        onsets, onset_env = detect_intensity_onsets(int_time, int_vals)
        tempo = estimate_tempo(onsets)

        # RMS
        rms = librosa.feature.rms(y=y)[0]
        rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        rms_interp = np.interp(time, rms_time, rms)

        # Silent gaps
        silent_gaps = []
        start = None
        for i, uv in enumerate(~voiced_mask):
            if uv and start is None:
                start = i
            if (not uv or i == len(voiced_mask) - 1) and start is not None:
                end = i if uv else i + 1
                gap = time[min(end - 1, len(time) - 1)] - time[start]
                if gap >= 0.15:
                    silent_gaps.append(float(gap))
                start = None

        # Spectral features (Estill)
        spectral = compute_spectral_features(y, sr)

        # CPP contour
        cpp_times, cpp_contour = compute_cpp_contour(y, sr)

        # H1-H2 contour
        h1h2_times, h1h2_contour = compute_h1h2_contour(y, sr, f0, time)

        # MFCC (20 coefficients)
        hop = max(1, int(TIME_STEP * sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop, n_fft=2048)
        mfcc_times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop)

        # Log mel spectrogram
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, hop_length=hop, n_fft=2048)
        logmel = librosa.power_to_db(mel, ref=np.max)
        logmel_times = librosa.frames_to_time(np.arange(logmel.shape[1]), sr=sr, hop_length=hop)

        return {
            "path": audio_path,
            "sr": sr,
            "y": y,
            "duration": duration,
            "time": time,
            "f0": f0,
            "voiced_mask": voiced_mask,
            "voiced_ratio": float(np.mean(voiced_mask)) if voiced_mask.size else np.nan,
            "intensity": int_interp,
            "int_time": int_time,
            "int_vals": int_vals,
            "hnr": hnr_interp,
            "f1": f1,
            "f2": f2,
            "f3": f3,
            "jitter": jitter,
            "shimmer": shimmer,
            "onsets": onsets,
            "onset_env": onset_env,
            "tempo": tempo,
            "rms": rms_interp,
            "silent_gaps": np.array(silent_gaps, dtype=float),
            # Estill
            "alpha_ratio_db": spectral["alpha_ratio_db"],
            "singer_formant_pct": spectral["singer_formant_pct"],
            "spectral_tilt_db_oct": spectral["spectral_tilt_db_oct"],
            "ltas_freqs": spectral["ltas_freqs"],
            "ltas_db": spectral["ltas_db"],
            "alpha_times": spectral["alpha_times"],
            "alpha_contour": spectral["alpha_contour"],
            "cpp_times": cpp_times,
            "cpp_contour": cpp_contour,
            "h1h2_times": h1h2_times,
            "h1h2_contour": h1h2_contour,
            "mfcc": mfcc,
            "mfcc_times": mfcc_times,
            "logmel": logmel,
            "logmel_times": logmel_times,
        }
    finally:
        if is_tmp:
            Path(wav_path).unlink(missing_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# DTW alignment
# ─────────────────────────────────────────────────────────────────────────────

def align_by_pitch(teacher, student):
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


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation / scoring
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(teacher, student, alignment, onset_errors, duration_errors,
             teacher_attack_rows, student_attack_rows):
    pe = alignment["pitch_errors_cents"]
    ape = np.abs(pe)

    # Pitch metrics
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
    tempo_diff = abs(student["tempo"] - teacher["tempo"]) if (
        np.isfinite(student["tempo"]) and np.isfinite(teacher["tempo"])) else np.nan

    hnr_t = safe_mean(teacher["hnr"])
    hnr_s = safe_mean(student["hnr"])
    hnr_diff = hnr_s - hnr_t

    vr_diff = student["voiced_ratio"] - teacher["voiced_ratio"]

    t_rise, t_gain = attack_summary(teacher_attack_rows)
    s_rise, s_gain = attack_summary(student_attack_rows)

    vib_t_rate, vib_t_ext = vibrato_metrics(teacher["f0"], teacher["time"])
    vib_s_rate, vib_s_ext = vibrato_metrics(student["f0"], student["time"])

    # Scores
    intonation_score = clamp(100.0 - (p_mean / 1.5 if np.isfinite(p_mean) else 100.0))
    rhythm_score = clamp(np.nanmean([
        100.0 - onset_mae * 1000.0 / 5.0 if np.isfinite(onset_mae) else np.nan,
        100.0 - dur_mae * 1000.0 / 7.0 if np.isfinite(dur_mae) else np.nan,
        100.0 - tempo_diff * 2.0 if np.isfinite(tempo_diff) else np.nan,
    ]))
    attack_score = clamp(np.nanmean([
        100.0 - abs(s_rise - t_rise) * 1000.0 / 3.0 if np.isfinite(s_rise) and np.isfinite(t_rise) else np.nan,
        100.0 - abs(s_gain - t_gain) * 8.0 if np.isfinite(s_gain) and np.isfinite(t_gain) else np.nan,
    ]))
    breath_score = clamp(np.nanmean([
        100.0 - abs(vr_diff) * 250.0 if np.isfinite(vr_diff) else np.nan,
        100.0 - abs(safe_mean(student["silent_gaps"]) - safe_mean(teacher["silent_gaps"])) * 100.0,
        100.0 - max(0.0, -hnr_diff) * 4.0 if np.isfinite(hnr_diff) else np.nan,
    ]))
    voice_closure_score = clamp(np.nanmean([
        100.0 - abs(vr_diff) * 300.0 if np.isfinite(vr_diff) else np.nan,
        100.0 - abs(student["jitter"] - teacher["jitter"]) * 6000.0,
        100.0 - abs(student["shimmer"] - teacher["shimmer"]) * 900.0,
        100.0 - abs(hnr_diff) * 2.0 if np.isfinite(hnr_diff) else np.nan,
    ]))
    overall_score = clamp(
        0.40 * intonation_score + 0.25 * rhythm_score +
        0.15 * voice_closure_score + 0.10 * attack_score + 0.10 * breath_score
    )

    # Estill diff metrics
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
        # Pitch
        "pitch_mean_abs_cents": p_mean,
        "pitch_median_abs_cents": p_median,
        "pitch_bias_cents": p_bias,
        "pitch_std_cents": p_std,
        "pitch_p90_abs_cents": p_p90,
        "in_tune_25_pct": in_25,
        "in_tune_50_pct": in_50,
        "in_tune_100_pct": in_100,
        # Rhythm
        "onset_mae_ms": onset_mae * 1000.0 if np.isfinite(onset_mae) else np.nan,
        "duration_mae_ms": dur_mae * 1000.0 if np.isfinite(dur_mae) else np.nan,
        "tempo_diff_bpm": tempo_diff,
        # Voice control
        "hnr_teacher_db": hnr_t,
        "hnr_student_db": hnr_s,
        "hnr_diff_db": hnr_diff,
        "jitter_teacher": teacher["jitter"],
        "jitter_student": student["jitter"],
        "shimmer_teacher": teacher["shimmer"],
        "shimmer_student": student["shimmer"],
        "voiced_ratio_diff": vr_diff,
        # Attack
        "attack_rise_teacher_ms": t_rise * 1000.0 if np.isfinite(t_rise) else np.nan,
        "attack_rise_student_ms": s_rise * 1000.0 if np.isfinite(s_rise) else np.nan,
        "attack_gain_teacher_db": t_gain,
        "attack_gain_student_db": s_gain,
        # Vibrato
        "vibrato_rate_teacher_hz": vib_t_rate,
        "vibrato_rate_student_hz": vib_s_rate,
        "vibrato_extent_teacher_hz": vib_t_ext,
        "vibrato_extent_student_hz": vib_s_ext,
        # Breath / pauses
        "silent_gaps_teacher_count": int(len(teacher["silent_gaps"])),
        "silent_gaps_student_count": int(len(student["silent_gaps"])),
        "silent_gap_mean_teacher_s": safe_mean(teacher["silent_gaps"]),
        "silent_gap_mean_student_s": safe_mean(student["silent_gaps"]),
        # Estill
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
        d = "ярче (больше twang/мetal)" if m["alpha_ratio_diff_db"] > 0 else "темнее (меньше яркости)"
        flags.append(f"Тембр (alpha ratio): голос ученика {d}, чем у учителя.")
    if np.isfinite(m["singer_formant_diff_pct"]) and abs(m["singer_formant_diff_pct"]) > 1.5:
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


# ─────────────────────────────────────────────────────────────────────────────
# Visualizations
# ─────────────────────────────────────────────────────────────────────────────

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

    # Scores table
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

    # Basic parameters
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

    # Intonation
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

    # Spectral analysis
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

    # Formants
    lines += [
        "## Форманты",
        "",
        img("Formants F1/F2", "formants.png"),
        "",
    ]

    # Intensity
    lines += [
        "## Динамика громкости",
        "",
        img("Intensity dynamics", "intensity_dynamics.png"),
        "",
    ]

    # Rhythm & attack
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

    # Voice control
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

    # Vibrato
    lines += [
        "## Вибрато",
        "",
        "| Метрика | Teacher | Student | Разница |",
        "|---------|---------|---------|---------|",
        f"| Vibrato rate (Hz) | {fmt(m['vibrato_rate_teacher_hz'])} | {fmt(m['vibrato_rate_student_hz'])} | {fmt(m['vibrato_rate_student_hz'] - m['vibrato_rate_teacher_hz'] if np.isfinite(m['vibrato_rate_student_hz']) and np.isfinite(m['vibrato_rate_teacher_hz']) else np.nan)} |",
        f"| Vibrato extent (Hz) | {fmt(m['vibrato_extent_teacher_hz'])} | {fmt(m['vibrato_extent_student_hz'])} | {fmt(m['vibrato_extent_student_hz'] - m['vibrato_extent_teacher_hz'] if np.isfinite(m['vibrato_extent_student_hz']) and np.isfinite(m['vibrato_extent_teacher_hz']) else np.nan)} |",
        "",
    ]

    # Estill features
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

    # Priority findings
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

    # Feedback
    lines += ["## Фидбэк ученику (AI)", ""]
    if feedback:
        lines.append(feedback)
    else:
        lines.append(
            "_Генерация фидбэка пропущена. "
            "Запусти с `--model <ollama_model>` и убедись, что Ollama запущен._"
        )
    lines.append("")

    # Technical appendix
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
# LLM feedback
# ─────────────────────────────────────────────────────────────────────────────

FEEDBACK_SYSTEM = (
    "Не выводи внутренний анализ. Не пиши рассуждения перед ответом. "
    "Начинай сразу с '### 1. Общая картина'. "
    "Пиши по-русски, поддерживающим тоном, ясно. Без медицинских выводов."
)

FEEDBACK_USER_TEMPLATE = """\
На основе технического отчёта сформируй педагогический фидбэк для ученика вокала, \
занимающегося по методике Estill Voice Training.

Структура ответа:
### 1. Общая картина (2 предложения)
### 2. Что получилось хорошо (2–3 пункта)
### 3. Главные зоны роста (3–4 пункта; для каждой: что происходит → почему проблема → как звучит)
### 4. Что делать (3 конкретных упражнения с инструкцией)
### 5. План на неделю (3 шага)
### 6. Короткое резюме ученику (1–2 предложения)

Если в отчёте упоминаются Estill-признаки (CPP, H1-H2, alpha ratio, spectral tilt), \
объясни их смысл без жаргона.

Технический отчёт:
{report}
"""


def _strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    hit = re.search(r"(#{1,3}\s*)?1\.\s*\**\s*Общая картина", text, flags=re.IGNORECASE)
    if hit:
        text = text[hit.start():]
    return text.strip()


def generate_feedback(
    text_report: str,
    model: str = OLLAMA_MODEL,
    url: str = OLLAMA_CHAT_URL,
    max_tokens: int = 1200,
) -> str:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": FEEDBACK_SYSTEM},
            {"role": "user", "content": FEEDBACK_USER_TEMPLATE.format(report=text_report)},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": 0.25,
            "top_p": 0.9,
            "num_ctx": 8192,
            "num_predict": max_tokens,
        },
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            f"Не удалось подключиться к Ollama ({url}). "
            f"Убедись, что сервер запущен и модель установлена: ollama pull {model}"
        ) from exc

    content = data.get("message", {}).get("content", "")
    if not content:
        raise RuntimeError(f"Ollama вернул пустой ответ: {data}")
    return _strip_thinking(content)


# ─────────────────────────────────────────────────────────────────────────────
# I/O helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sanitize(name: str, maxlen: int = 60) -> str:
    import re
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
