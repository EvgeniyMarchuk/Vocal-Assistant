from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call as praat_call
from scipy.signal import find_peaks
from scipy.stats import linregress

from .utils import (
    PITCH_CEILING,
    PITCH_FLOOR,
    TIME_STEP,
    estimate_tempo,
    moving_average,
    safe_median,
    safe_pct,
    safe_std,
)

# ─────────────────────────────────────────────────────────────────────────────
# Onset detection
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
# Attack metrics
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
# Vibrato
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

    ltas = np.mean(D, axis=1)
    ltas_db = 20.0 * np.log10(ltas + 1e-10)

    lo_mask = (freqs >= 50) & (freqs <= 1000)
    hi_mask = (freqs >= 1000) & (freqs <= 5000)
    e_lo = np.sum(D[lo_mask] ** 2, axis=None)
    e_hi = np.sum(D[hi_mask] ** 2, axis=None)
    alpha_ratio = float(10.0 * np.log10(e_hi / e_lo)) if e_lo > 0 else np.nan

    sf_mask = (freqs >= 2500) & (freqs <= 3500)
    total_e = np.sum(D ** 2)
    sf_e = np.sum(D[sf_mask] ** 2)
    singer_formant_pct = float(100.0 * sf_e / total_e) if total_e > 0 else np.nan

    valid = (freqs > 50) & np.isfinite(ltas_db)
    if valid.sum() >= 5:
        slope, *_ = linregress(np.log2(freqs[valid]), ltas_db[valid])
        spectral_tilt = float(slope)
    else:
        spectral_tilt = np.nan

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

        pitch = snd.to_pitch(time_step=TIME_STEP, pitch_floor=PITCH_FLOOR, pitch_ceiling=PITCH_CEILING)
        time = pitch.xs()
        f0 = pitch.selected_array["frequency"].astype(float)
        f0[f0 <= 0] = np.nan
        voiced_mask = np.isfinite(f0)

        intensity = snd.to_intensity(time_step=TIME_STEP, minimum_pitch=PITCH_FLOOR)
        int_time = intensity.xs()
        int_vals = intensity.values[0].astype(float)
        int_interp = np.interp(time, int_time, int_vals)

        harmonicity = snd.to_harmonicity_cc(time_step=TIME_STEP, minimum_pitch=PITCH_FLOOR)
        hnr_time = harmonicity.xs()
        hnr_vals = harmonicity.values[0].astype(float)
        hnr_vals[hnr_vals <= -200] = np.nan
        vh = np.isfinite(hnr_vals)
        hnr_interp = np.interp(time, hnr_time[vh], hnr_vals[vh]) if vh.sum() >= 2 else np.full_like(time, np.nan)

        formant = snd.to_formant_burg(time_step=TIME_STEP)
        f1 = np.array([formant.get_value_at_time(1, t) for t in time], dtype=float)
        f2 = np.array([formant.get_value_at_time(2, t) for t in time], dtype=float)
        f3 = np.array([formant.get_value_at_time(3, t) for t in time], dtype=float)

        pp = praat_call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
        jitter = float(praat_call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3))
        shimmer = float(praat_call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6))

        onsets, onset_env = detect_intensity_onsets(int_time, int_vals)
        tempo = estimate_tempo(onsets)

        rms = librosa.feature.rms(y=y)[0]
        rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
        rms_interp = np.interp(time, rms_time, rms)

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

        spectral = compute_spectral_features(y, sr)
        cpp_times, cpp_contour = compute_cpp_contour(y, sr)
        h1h2_times, h1h2_contour = compute_h1h2_contour(y, sr, f0, time)

        hop = max(1, int(TIME_STEP * sr))
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=hop, n_fft=2048)
        mfcc_times = librosa.frames_to_time(np.arange(mfcc.shape[1]), sr=sr, hop_length=hop)

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
