from pathlib import Path
import hashlib

import librosa
import matplotlib.pyplot as plt
import numpy as np
import parselmouth
from fastdtw import fastdtw
from scipy.signal import find_peaks
from scipy.spatial.distance import euclidean


# =====================================================
# HELPERS
# =====================================================

REF_HZ = 440.0


def safe_mean(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmean(arr)) if arr.size else np.nan


def safe_std(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanstd(arr)) if arr.size else np.nan


def safe_median(values):
    arr = np.asarray(values, dtype=float)
    return float(np.nanmedian(arr)) if arr.size else np.nan


def safe_percentile(values, q):
    arr = np.asarray(values, dtype=float)
    return float(np.nanpercentile(arr, q)) if arr.size else np.nan


def clamp(value, low=0.0, high=100.0):
    return float(np.clip(value, low, high))


def fmt(value, digits=2, nan_value="n/a"):
    if value is None or not np.isfinite(value):
        return nan_value
    return f"{value:.{digits}f}"


def hz_to_cents(frequency_hz):
    return 1200 * np.log2(frequency_hz / REF_HZ)


# =====================================================
# FEATURE EXTRACTION
# =====================================================

def extract_all_features(audio_path, time_step=0.01):
    snd = parselmouth.Sound(audio_path)
    y, sr = librosa.load(audio_path, sr=None)
    duration = snd.get_total_duration()

    # ---------------- Pitch ----------------
    pitch = snd.to_pitch(time_step=time_step, pitch_floor=75, pitch_ceiling=600)
    f0 = pitch.selected_array["frequency"].astype(float)
    f0[f0 <= 0] = np.nan
    time = pitch.xs()
    voiced_mask = np.isfinite(f0)
    voiced_ratio = float(np.mean(voiced_mask)) if f0.size else 0.0

    # ---------------- Intensity ----------------
    intensity = snd.to_intensity(time_step=time_step)
    intensity_time = intensity.xs()
    intensity_values = intensity.values[0].astype(float)
    intensity_interp = np.interp(time, intensity_time, intensity_values)

    # ---------------- Harmonicity / HNR ----------------
    harmonicity = snd.to_harmonicity_cc(time_step=time_step, minimum_pitch=75)
    hnr_time = harmonicity.xs()
    hnr_values = harmonicity.values[0].astype(float)
    hnr_values[hnr_values <= -200] = np.nan
    valid_hnr = np.isfinite(hnr_values)
    if np.sum(valid_hnr) >= 2:
        hnr_interp = np.interp(time, hnr_time[valid_hnr], hnr_values[valid_hnr])
    else:
        hnr_interp = np.full_like(time, np.nan, dtype=float)

    # ---------------- Formants ----------------
    formant = snd.to_formant_burg(time_step=time_step)
    f1, f2, f3 = [], [], []
    for t in time:
        f1.append(formant.get_value_at_time(1, t))
        f2.append(formant.get_value_at_time(2, t))
        f3.append(formant.get_value_at_time(3, t))
    f1 = np.asarray(f1, dtype=float)
    f2 = np.asarray(f2, dtype=float)
    f3 = np.asarray(f3, dtype=float)

    # ---------------- Jitter / Shimmer ----------------
    point_process = parselmouth.praat.call(
        snd,
        "To PointProcess (periodic, cc)",
        75,
        600,
    )
    jitter = parselmouth.praat.call(
        point_process,
        "Get jitter (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
    )
    shimmer = parselmouth.praat.call(
        [snd, point_process],
        "Get shimmer (local)",
        0,
        0,
        0.0001,
        0.02,
        1.3,
        1.6,
    )

    # ---------------- Onsets & Rhythm ----------------
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    onset_frames = librosa.onset.onset_detect(onset_envelope=onset_env, sr=sr)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    tempo = float(np.ravel(tempo)[0]) if np.size(tempo) else np.nan

    # ---------------- Energy dynamics ----------------
    rms = librosa.feature.rms(y=y)[0]
    rms_time = librosa.frames_to_time(np.arange(len(rms)), sr=sr)
    rms_interp = np.interp(time, rms_time, rms)

    return {
        "audio_path": audio_path,
        "time": time,
        "f0": f0,
        "voiced_mask": voiced_mask,
        "voiced_ratio": voiced_ratio,
        "intensity": intensity_interp,
        "intensity_raw": intensity_values,
        "intensity_time": intensity_time,
        "hnr": hnr_interp,
        "f1": f1,
        "f2": f2,
        "f3": f3,
        "jitter": float(jitter),
        "shimmer": float(shimmer),
        "onsets": onset_times,
        "tempo": tempo,
        "beats": beat_times,
        "rms": rms_interp,
        "duration": float(duration),
    }


# =====================================================
# ALIGNMENT
# =====================================================

def align_pitch(teacher_f0, student_f0):
    t = np.nan_to_num(teacher_f0, nan=0.0)
    s = np.nan_to_num(student_f0, nan=0.0)
    dtw_distance, path = fastdtw(
        t.reshape(-1, 1),
        s.reshape(-1, 1),
        dist=euclidean,
    )

    teacher_idx = []
    student_idx = []
    pitch_errors = []

    for i, j in path:
        tf = teacher_f0[i]
        sf = student_f0[j]
        if np.isfinite(tf) and tf > 0 and np.isfinite(sf) and sf > 0:
            error_cents = hz_to_cents(sf) - hz_to_cents(tf)
            teacher_idx.append(i)
            student_idx.append(j)
            pitch_errors.append(error_cents)

    return {
        "dtw_distance": float(dtw_distance),
        "path": path,
        "teacher_idx": np.asarray(teacher_idx, dtype=int),
        "student_idx": np.asarray(student_idx, dtype=int),
        "pitch_errors_cents": np.asarray(pitch_errors, dtype=float),
    }


# =====================================================
# VIBRATO ANALYSIS
# =====================================================

def analyze_vibrato(f0, frame_rate_hz=100):
    if f0.size == 0:
        return np.nan, np.nan

    f0_clean = np.nan_to_num(f0, nan=0.0)
    peaks, _ = find_peaks(f0_clean, distance=max(1, int(frame_rate_hz / 12)))

    if len(peaks) < 2:
        return np.nan, np.nan

    peak_times = peaks / frame_rate_hz
    peak_diffs = np.diff(peak_times)
    valid_diffs = peak_diffs[peak_diffs > 0]
    if valid_diffs.size == 0:
        return np.nan, np.nan

    vibrato_rate = 1.0 / np.mean(valid_diffs)
    vibrato_extent = safe_std(f0[np.isfinite(f0)])

    return float(vibrato_rate), float(vibrato_extent)


def estimate_frame_rate(time_axis):
    if len(time_axis) < 2:
        return 100
    dt = time_axis[1] - time_axis[0]
    if dt <= 0:
        return 100
    return int(round(1.0 / dt))


# =====================================================
# RHYTHM COMPARISON
# =====================================================

def compare_onsets(teacher_onsets, student_onsets):
    min_len = min(len(teacher_onsets), len(student_onsets))
    if min_len == 0:
        return np.array([])
    return student_onsets[:min_len] - teacher_onsets[:min_len]


def compare_note_durations(teacher_onsets, student_onsets):
    min_len = min(len(teacher_onsets), len(student_onsets))
    if min_len < 2:
        return np.array([])
    teacher_dur = np.diff(teacher_onsets[:min_len])
    student_dur = np.diff(student_onsets[:min_len])
    return student_dur - teacher_dur


# =====================================================
# SCORING FOR EXERCISE CORRECTNESS
# =====================================================

def evaluate_performance(
    teacher,
    student,
    alignment,
    onset_error,
    duration_error,
    vib_teacher,
    vib_student,
):
    errors = alignment["pitch_errors_cents"]
    abs_errors = np.abs(errors)

    mean_abs_pitch = safe_mean(abs_errors)
    median_abs_pitch = safe_median(abs_errors)
    pitch_std = safe_std(errors)
    in_tune_25 = 100.0 * np.mean(abs_errors <= 25) if abs_errors.size else np.nan
    in_tune_50 = 100.0 * np.mean(abs_errors <= 50) if abs_errors.size else np.nan
    in_tune_100 = 100.0 * np.mean(abs_errors <= 100) if abs_errors.size else np.nan

    onset_mae = safe_mean(np.abs(onset_error))
    dur_mae = safe_mean(np.abs(duration_error))
    tempo_diff = abs(student["tempo"] - teacher["tempo"])
    voiced_ratio_diff = abs(student["voiced_ratio"] - teacher["voiced_ratio"])

    jitter_diff = abs(student["jitter"] - teacher["jitter"])
    shimmer_diff = abs(student["shimmer"] - teacher["shimmer"])

    hnr_teacher = safe_mean(teacher["hnr"])
    hnr_student = safe_mean(student["hnr"])
    hnr_diff = abs(hnr_student - hnr_teacher)

    vib_rate_t, vib_ext_t = vib_teacher
    vib_rate_s, vib_ext_s = vib_student
    vib_rate_diff = abs(vib_rate_s - vib_rate_t) if np.isfinite(vib_rate_t) and np.isfinite(vib_rate_s) else np.nan
    vib_ext_diff = abs(vib_ext_s - vib_ext_t) if np.isfinite(vib_ext_t) and np.isfinite(vib_ext_s) else np.nan

    # Heuristic scores in [0, 100] to rank student performance.
    intonation_score = clamp(100.0 - (mean_abs_pitch / 1.5 if np.isfinite(mean_abs_pitch) else 100.0))
    rhythm_score = clamp(
        np.nanmean(
            [
                100.0 - (onset_mae * 1000.0) / 5.0 if np.isfinite(onset_mae) else np.nan,
                100.0 - (dur_mae * 1000.0) / 7.0 if np.isfinite(dur_mae) else np.nan,
                100.0 - tempo_diff * 2.0 if np.isfinite(tempo_diff) else np.nan,
            ]
        )
    )
    voice_control_score = clamp(
        np.nanmean(
            [
                100.0 - voiced_ratio_diff * 300.0 if np.isfinite(voiced_ratio_diff) else np.nan,
                100.0 - jitter_diff * 6000.0 if np.isfinite(jitter_diff) else np.nan,
                100.0 - shimmer_diff * 900.0 if np.isfinite(shimmer_diff) else np.nan,
                100.0 - hnr_diff * 2.0 if np.isfinite(hnr_diff) else np.nan,
            ]
        )
    )

    overall_score = clamp(
        0.50 * intonation_score + 0.35 * rhythm_score + 0.15 * voice_control_score
    )

    return {
        "pitch_mean_abs_cents": mean_abs_pitch,
        "pitch_median_abs_cents": median_abs_pitch,
        "pitch_std_cents": pitch_std,
        "pitch_bias_cents": safe_mean(errors),
        "pitch_p90_abs_cents": safe_percentile(abs_errors, 90),
        "in_tune_25_percent": in_tune_25,
        "in_tune_50_percent": in_tune_50,
        "in_tune_100_percent": in_tune_100,
        "onset_mae_sec": onset_mae,
        "duration_mae_sec": dur_mae,
        "tempo_diff_bpm": tempo_diff,
        "voiced_ratio_diff": voiced_ratio_diff,
        "jitter_diff": jitter_diff,
        "shimmer_diff": shimmer_diff,
        "hnr_teacher_db": hnr_teacher,
        "hnr_student_db": hnr_student,
        "hnr_diff_db": hnr_diff,
        "vibrato_rate_teacher": vib_rate_t,
        "vibrato_rate_student": vib_rate_s,
        "vibrato_rate_diff": vib_rate_diff,
        "vibrato_extent_teacher": vib_ext_t,
        "vibrato_extent_student": vib_ext_s,
        "vibrato_extent_diff": vib_ext_diff,
        "intonation_score": intonation_score,
        "rhythm_score": rhythm_score,
        "voice_control_score": voice_control_score,
        "overall_score": overall_score,
    }


# =====================================================
# REPORT
# =====================================================

def generate_full_report(teacher, student, alignment, onset_error, duration_error, metrics):
    errors = alignment["pitch_errors_cents"]
    report = []

    report.append("===== ОЦЕНКА ПОВТОРА УПРАЖНЕНИЯ (Teacher vs Student) =====")
    report.append(
        f"Overall score: {fmt(metrics['overall_score'], 1)}/100 "
        f"(Intonation: {fmt(metrics['intonation_score'], 1)}, "
        f"Rhythm: {fmt(metrics['rhythm_score'], 1)}, "
        f"Voice control: {fmt(metrics['voice_control_score'], 1)})"
    )
    report.append("")
    report.append("----- Базовые параметры сигнала -----")
    report.append(f"Duration: teacher {fmt(teacher['duration'], 2)} s | student {fmt(student['duration'], 2)} s")
    report.append(f"Tempo: teacher {fmt(teacher['tempo'], 1)} BPM | student {fmt(student['tempo'], 1)} BPM | diff {fmt(metrics['tempo_diff_bpm'], 1)}")
    report.append(f"Onsets detected: teacher {len(teacher['onsets'])} | student {len(student['onsets'])}")
    report.append(f"Voiced ratio: teacher {fmt(teacher['voiced_ratio'], 3)} | student {fmt(student['voiced_ratio'], 3)} | diff {fmt(metrics['voiced_ratio_diff'], 3)}")
    report.append("")
    report.append("----- Интонация (через DTW-выравнивание) -----")
    report.append(f"Aligned voiced frames: {len(errors)}")
    report.append(f"DTW distance: {fmt(alignment['dtw_distance'], 1)}")
    report.append(f"Pitch mean abs error: {fmt(metrics['pitch_mean_abs_cents'], 1)} cents")
    report.append(f"Pitch median abs error: {fmt(metrics['pitch_median_abs_cents'], 1)} cents")
    report.append(f"Pitch bias (student-teacher): {fmt(metrics['pitch_bias_cents'], 1)} cents")
    report.append(f"Pitch spread std: {fmt(metrics['pitch_std_cents'], 1)} cents")
    report.append(f"Pitch P90 abs error: {fmt(metrics['pitch_p90_abs_cents'], 1)} cents")
    report.append(
        f"In tune ratio: ±25c {fmt(metrics['in_tune_25_percent'], 1)}% | "
        f"±50c {fmt(metrics['in_tune_50_percent'], 1)}% | "
        f"±100c {fmt(metrics['in_tune_100_percent'], 1)}%"
    )
    report.append("")
    report.append("----- Ритм -----")
    report.append(f"Onset MAE: {fmt(metrics['onset_mae_sec'] * 1000 if np.isfinite(metrics['onset_mae_sec']) else np.nan, 1)} ms")
    report.append(f"Duration MAE: {fmt(metrics['duration_mae_sec'] * 1000 if np.isfinite(metrics['duration_mae_sec']) else np.nan, 1)} ms")
    if onset_error.size:
        report.append(
            f"Onset error median / std: "
            f"{fmt(safe_median(onset_error) * 1000, 1)} / {fmt(safe_std(onset_error) * 1000, 1)} ms"
        )
    if duration_error.size:
        report.append(
            f"Duration error median / std: "
            f"{fmt(safe_median(duration_error) * 1000, 1)} / {fmt(safe_std(duration_error) * 1000, 1)} ms"
        )
    report.append("")
    report.append("----- Голосовой контроль (Praat) -----")
    report.append(f"Jitter: teacher {fmt(teacher['jitter'], 4)} | student {fmt(student['jitter'], 4)} | diff {fmt(metrics['jitter_diff'], 4)}")
    report.append(f"Shimmer: teacher {fmt(teacher['shimmer'], 4)} | student {fmt(student['shimmer'], 4)} | diff {fmt(metrics['shimmer_diff'], 4)}")
    report.append(f"HNR mean: teacher {fmt(metrics['hnr_teacher_db'], 2)} dB | student {fmt(metrics['hnr_student_db'], 2)} dB | diff {fmt(metrics['hnr_diff_db'], 2)} dB")
    report.append("")
    report.append("----- Вибрато (грубая оценка) -----")
    report.append(
        f"Vibrato rate: teacher {fmt(metrics['vibrato_rate_teacher'], 2)} Hz | "
        f"student {fmt(metrics['vibrato_rate_student'], 2)} Hz | "
        f"diff {fmt(metrics['vibrato_rate_diff'], 2)}"
    )
    report.append(
        f"Vibrato extent: teacher {fmt(metrics['vibrato_extent_teacher'], 2)} Hz | "
        f"student {fmt(metrics['vibrato_extent_student'], 2)} Hz | "
        f"diff {fmt(metrics['vibrato_extent_diff'], 2)}"
    )

    # Action-oriented conclusions for vocal exercise correction.
    recommendations = []
    # if np.isfinite(metrics["in_tune_50_percent"]) and metrics["in_tune_50_percent"] < 65:
    #     recommendations.append("Интонация: мало попаданий в ±50 cents, полезно замедлить упражнение и петь под референсные ноты.")
    # if np.isfinite(metrics["onset_mae_sec"]) and metrics["onset_mae_sec"] > 0.10:
    #     recommendations.append("Ритм: заметное смещение вступлений, стоит тренировать входы под метроном/клик.")
    # if np.isfinite(metrics["duration_mae_sec"]) and metrics["duration_mae_sec"] > 0.12:
    #     recommendations.append("Ритм: длительности нот сильно отличаются, важно контролировать удержание каждой ноты.")
    # if np.isfinite(metrics["pitch_std_cents"]) and metrics["pitch_std_cents"] > 90:
    #     recommendations.append("Стабильность: большой разброс высоты, стоит добавить упражнения на ровное держание тона.")
    # if np.isfinite(metrics["voiced_ratio_diff"]) and metrics["voiced_ratio_diff"] > 0.15:
    #     recommendations.append("Фразировка/дыхание: доля озвученных участков отличается от учителя, проверьте паузы и поддержку дыханием.")

    if recommendations:
        report.append("")
        report.append("----- Что исправлять в первую очередь -----")
        for idx, rec in enumerate(recommendations, start=1):
            report.append(f"{idx}. {rec}")

    return "\n".join(report)


# =====================================================
# VISUALIZATION
# =====================================================

def _plot_onsets(ax, onset_times, color, label):
    max_markers = 40
    for idx, t in enumerate(onset_times[:max_markers]):
        ax.axvline(t, color=color, alpha=0.15, linewidth=0.8, label=label if idx == 0 else None)


def plot_diagnostics(teacher, student, alignment, metrics, output_path):
    teacher_idx = alignment["teacher_idx"]
    student_idx = alignment["student_idx"]
    errors = alignment["pitch_errors_cents"]

    fig, axes = plt.subplots(3, 2, figsize=(16, 13))
    ax1, ax2, ax3, ax4, ax5, ax6 = axes.flatten()

    # 1) Raw pitch contours
    ax1.plot(teacher["time"], teacher["f0"], linewidth=1.1, label="Teacher F0", color="tab:blue")
    ax1.plot(student["time"], student["f0"], linewidth=1.1, alpha=0.85, label="Student F0", color="tab:red")
    _plot_onsets(ax1, teacher["onsets"], "tab:blue", "Teacher onsets")
    _plot_onsets(ax1, student["onsets"], "tab:red", "Student onsets")
    ax1.set_title("Pitch contour + onsets")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("F0 (Hz)")
    ax1.grid(alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    # 2) Aligned pitch error over teacher time
    if teacher_idx.size:
        ax2.plot(teacher["time"][teacher_idx], errors, color="tab:purple", linewidth=1.0)
        ax2.axhline(0, color="black", linewidth=1.0)
        ax2.axhline(50, color="tab:green", linestyle="--", linewidth=0.8)
        ax2.axhline(-50, color="tab:green", linestyle="--", linewidth=0.8)
        ax2.axhline(100, color="tab:orange", linestyle="--", linewidth=0.8)
        ax2.axhline(-100, color="tab:orange", linestyle="--", linewidth=0.8)
    ax2.set_title("Aligned pitch error (student - teacher, cents)")
    ax2.set_xlabel("Teacher time (s)")
    ax2.set_ylabel("Error (cents)")
    ax2.grid(alpha=0.3)

    # 3) Error distribution
    if errors.size:
        ax3.hist(errors, bins=40, color="tab:purple", alpha=0.75, edgecolor="white")
        ax3.axvline(0, color="black", linewidth=1.0)
    ax3.set_title("Pitch error distribution")
    ax3.set_xlabel("Error (cents)")
    ax3.set_ylabel("Count")
    ax3.grid(alpha=0.25)

    # 4) Intensity and RMS dynamics
    ax4.plot(teacher["time"], teacher["intensity"], label="Teacher intensity", color="tab:blue", linewidth=1.0)
    ax4.plot(student["time"], student["intensity"], label="Student intensity", color="tab:red", linewidth=1.0, alpha=0.85)
    ax4.plot(teacher["time"], teacher["rms"] * 100, label="Teacher RMS x100", color="tab:cyan", alpha=0.5)
    ax4.plot(student["time"], student["rms"] * 100, label="Student RMS x100", color="tab:pink", alpha=0.5)
    ax4.set_title("Loudness / energy dynamics")
    ax4.set_xlabel("Time (s)")
    ax4.set_ylabel("Intensity (dB) and RMS*100")
    ax4.grid(alpha=0.3)
    ax4.legend(loc="upper right", fontsize=8)

    # 5) Formants F1/F2 for vocal tract behavior
    ax5.plot(teacher["time"], teacher["f1"], label="Teacher F1", color="#1b9e77", linewidth=1.0)
    ax5.plot(teacher["time"], teacher["f2"], label="Teacher F2", color="#66a61e", linewidth=1.0)
    ax5.plot(student["time"], student["f1"], label="Student F1", color="#d95f02", linewidth=1.0, alpha=0.85)
    ax5.plot(student["time"], student["f2"], label="Student F2", color="#e7298a", linewidth=1.0, alpha=0.85)
    ax5.set_title("Formants (F1/F2)")
    ax5.set_xlabel("Time (s)")
    ax5.set_ylabel("Frequency (Hz)")
    ax5.grid(alpha=0.3)
    ax5.legend(loc="upper right", fontsize=8)

    # 6) DTW path + score box
    if alignment["path"]:
        path_arr = np.array(alignment["path"])
        ax6.plot(path_arr[:, 0], path_arr[:, 1], color="tab:gray", linewidth=1.0)
    ax6.set_title("DTW alignment path")
    ax6.set_xlabel("Teacher frame index")
    ax6.set_ylabel("Student frame index")
    ax6.grid(alpha=0.3)
    text = (
        f"Overall: {fmt(metrics['overall_score'], 1)}/100\n"
        f"Intonation: {fmt(metrics['intonation_score'], 1)}\n"
        f"Rhythm: {fmt(metrics['rhythm_score'], 1)}\n"
        f"Voice ctrl: {fmt(metrics['voice_control_score'], 1)}\n"
        f"Pitch MAE: {fmt(metrics['pitch_mean_abs_cents'], 1)}c\n"
        f"Onset MAE: {fmt(metrics['onset_mae_sec'] * 1000 if np.isfinite(metrics['onset_mae_sec']) else np.nan, 1)} ms"
    )
    ax6.text(
        0.02,
        0.98,
        text,
        transform=ax6.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    fig.suptitle("Teacher vs Student Exercise Diagnostics (Praat + DTW)", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def build_unique_artifact_paths(teacher_audio, student_audio, output_dir):
    teacher_stem = Path(teacher_audio).stem
    student_stem = Path(student_audio).stem

    pair_key = f"{Path(teacher_audio).resolve()}||{Path(student_audio).resolve()}"
    pair_hash = hashlib.sha1(pair_key.encode("utf-8")).hexdigest()[:10]
    base_name = f"{teacher_stem}__vs__{student_stem}__{pair_hash}"

    out_dir = Path(output_dir)
    return (
        out_dir / f"{base_name}.png",
        out_dir / f"{base_name}.txt",
    )


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    teacher_audio = "/home/evgeniy/Projects/VocalAssistant/datasets/M3/001/teacher/20220330100654.wav"
    student_audio = "/home/evgeniy/Projects/VocalAssistant/datasets/M3/001/student/20220407120212.wav"
    output_dir = "/home/evgeniy/Projects/VocalAssistant/Praat_PitchEstimation/analysis"
    output_plot, output_report = build_unique_artifact_paths(
        teacher_audio=teacher_audio,
        student_audio=student_audio,
        output_dir=output_dir,
    )

    teacher = extract_all_features(teacher_audio)
    student = extract_all_features(student_audio)

    alignment = align_pitch(teacher["f0"], student["f0"])
    onset_error = compare_onsets(teacher["onsets"], student["onsets"])
    duration_error = compare_note_durations(teacher["onsets"], student["onsets"])
    vib_teacher = analyze_vibrato(teacher["f0"], frame_rate_hz=estimate_frame_rate(teacher["time"]))
    vib_student = analyze_vibrato(student["f0"], frame_rate_hz=estimate_frame_rate(student["time"]))

    metrics = evaluate_performance(
        teacher=teacher,
        student=student,
        alignment=alignment,
        onset_error=onset_error,
        duration_error=duration_error,
        vib_teacher=vib_teacher,
        vib_student=vib_student,
    )

    report = generate_full_report(
        teacher=teacher,
        student=student,
        alignment=alignment,
        onset_error=onset_error,
        duration_error=duration_error,
        metrics=metrics,
    )
    print(report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report + "\n", encoding="utf-8")

    plot_diagnostics(
        teacher=teacher,
        student=student,
        alignment=alignment,
        metrics=metrics,
        output_path=output_plot,
    )
    print(f"\nДиагностический график сохранен: {output_plot}")
    print(f"Текстовый отчет сохранен: {output_report}")
