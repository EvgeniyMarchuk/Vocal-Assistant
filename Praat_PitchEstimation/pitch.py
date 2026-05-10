import os
import numpy as np
import librosa
import matplotlib.pyplot as plt
import parselmouth

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


# ==========================
# 2. Praat feature extraction
# ==========================

def extract_praat_features(path):

    snd = parselmouth.Sound(path)

    pitch = snd.to_pitch()
    intensity = snd.to_intensity()
    harmonicity = snd.to_harmonicity()

    point_process = parselmouth.praat.call(
        snd, "To PointProcess (periodic, cc)", 75, 500
    )

    jitter = parselmouth.praat.call(
        point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3
    )

    shimmer = parselmouth.praat.call(
        [snd, point_process],
        "Get shimmer (local)",
        0, 0, 0.0001, 0.02, 1.3, 1.6
    )

    hnr = harmonicity.values[harmonicity.values != -200]
    mean_hnr = np.mean(hnr) if len(hnr) > 0 else 0

    formant = snd.to_formant_burg()

    f1 = parselmouth.praat.call(formant, "Get mean", 1, 0, 0, "Hertz")
    f2 = parselmouth.praat.call(formant, "Get mean", 2, 0, 0, "Hertz")

    mean_intensity = np.mean(intensity.values)

    return {
        "jitter": jitter,
        "shimmer": shimmer,
        "hnr": mean_hnr,
        "mean_intensity": mean_intensity,
        "f1": f1,
        "f2": f2,
    }


# ==========================
# 3. Convert Hz → cents
# ==========================

def hz_to_cents(f0, ref=440.0):
    f0 = np.maximum(f0, 1e-6)
    return 1200 * np.log2(f0 / ref)


# ==========================
# 4. DTW alignment
# ==========================

def align_sequences(seq1, seq2):
    _, path = fastdtw(
        seq1.reshape(-1, 1),
        seq2.reshape(-1, 1),
        dist=euclidean
    )
    return path


# ==========================
# 5. Note segmentation
# ==========================

def segment_notes(cents, confidence,
                  min_frames=20,
                  pitch_jump=50):

    notes = []
    start = None

    for i in range(len(cents)):
        if confidence[i] == 0:
            if start is not None and i - start > min_frames:
                notes.append((start, i))
            start = None
            continue

        if start is None:
            start = i
            continue

        if abs(cents[i] - cents[i - 1]) > pitch_jump:
            if i - start > min_frames:
                notes.append((start, i))
            start = i

    if start is not None and len(cents) - start > min_frames:
        notes.append((start, len(cents)))

    return notes


# ==========================
# 6. Diagnostic comparison
# ==========================

def analyze_notes(
    teacher_notes,
    teacher_cents,
    teacher_time,
    student_cents,
    student_time,
    dtw_path
):
    alignment = {}
    for t_idx, s_idx in dtw_path:
        alignment.setdefault(t_idx, []).append(s_idx)

    for i, (t_start, t_end) in enumerate(teacher_notes):

        student_indices = []
        for t in range(t_start, t_end):
            if t in alignment:
                student_indices.extend(alignment[t])

        if len(student_indices) == 0:
            continue

        s_start = min(student_indices)
        s_end = max(student_indices)

        teacher_pitch = np.mean(teacher_cents[t_start:t_end])
        student_pitch = np.mean(student_cents[s_start:s_end])
        pitch_diff = student_pitch - teacher_pitch

        teacher_dur = teacher_time[t_end - 1] - teacher_time[t_start]
        student_dur = student_time[s_end - 1] - student_time[s_start]

        if teacher_dur <= 0:
            continue

        dur_ratio = (student_dur - teacher_dur) / teacher_dur * 100
        onset_diff = (student_time[s_start] -
                      teacher_time[t_start]) * 1000

        print(f"\nНота {i + 1}:")
        print(f"— Pitch deviation: {pitch_diff:.1f} cents")
        print(f"— Duration {'longer' if dur_ratio > 0 else 'shorter'} "
              f"by {abs(dur_ratio):.1f}%")
        print(f"— Onset {'later' if onset_diff > 0 else 'earlier'} "
              f"by {abs(onset_diff):.0f} ms")


# ==========================
# MAIN
# ==========================

def main():

    teacher_path = "/home/evgeniy/Projects/VocalAssistant/datasets/M3/001/teacher/20220330100654.wav"
    student_path = "/home/evgeniy/Projects/VocalAssistant/datasets/M3/001/student/20220407120212.wav"

    print("\n=== PRAAT FEATURES ===")

    teacher_praat = extract_praat_features(teacher_path)
    student_praat = extract_praat_features(student_path)

    print("\nTeacher Praat features:", teacher_praat)
    print("\nStudent Praat features:", student_praat)


if __name__ == "__main__":
    main()