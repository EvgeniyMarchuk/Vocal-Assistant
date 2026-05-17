#!/usr/bin/env python3
"""Batch experiment on exercises_unpacked: compare all 'нет' vs 'да' pairs.

For each exercise type, runs analyze.py logic (no LLM) and prints a
structured metrics table. Also compares original alignment vs fixed alignment.
"""

from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba_cache")

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from vocal_analysis import (  # noqa: E402
    align_by_pitch,
    attack_metrics,
    build_flags,
    build_pedagogical_brief,
    build_rule_based_feedback,
    build_text_report,
    evaluate,
    extract_features,
    generate_feedback,
)
from vocal_analysis.utils import OLLAMA_CHAT_URL, OLLAMA_MODEL, fmt  # noqa: E402

EXERCISES_DIR = Path(__file__).parent / "exercises_unpacked" / "Упражнения примеры "
REPORTS_DIR = Path(__file__).parent / "reports_exercises"
REPORTS_DIR.mkdir(exist_ok=True)

# Map exercise name → teacher file
EXERCISES = {
    "Фальцет": "Фальцет да.m4a",
    "Спич": "Спич да.m4a",
    "СОБ": "СОБ да.m4a",
}


def run_pair(
    teacher_path: Path,
    student_path: Path,
    label: str,
    with_llm: bool = False,
    llm_model: str = OLLAMA_MODEL,
    llm_url: str = OLLAMA_CHAT_URL,
) -> dict:
    teacher = extract_features(str(teacher_path))
    student = extract_features(str(student_path))

    alignment = align_by_pitch(teacher, student)

    n = min(len(teacher["onsets"]), len(student["onsets"]))
    if n > 0:
        # Normalize to first onset — compare relative timing within performance.
        t_rel = teacher["onsets"][:n] - teacher["onsets"][0]
        s_rel = student["onsets"][:n] - student["onsets"][0]
        onset_errors = s_rel - t_rel
    else:
        onset_errors = np.array([])
    duration_errors = (
        np.diff(student["onsets"][:n]) - np.diff(teacher["onsets"][:n])
        if n >= 2
        else np.array([])
    )

    teacher_attacks = attack_metrics(
        teacher["time"], teacher["intensity"], teacher["onsets"]
    )
    student_attacks = attack_metrics(
        student["time"], student["intensity"], student["onsets"]
    )

    metrics = evaluate(
        teacher,
        student,
        alignment,
        onset_errors,
        duration_errors,
        teacher_attacks,
        student_attacks,
    )
    metrics["dtw_distance"] = alignment["dtw_distance"]

    flags = build_flags(teacher, student, metrics)

    feedback = None
    if with_llm:
        text_report = build_text_report(
            str(teacher_path),
            str(student_path),
            teacher,
            student,
            alignment,
            metrics,
        )
        pedagogical_brief = build_pedagogical_brief(metrics, flags)
        try:
            feedback = generate_feedback(
                text_report,
                model=llm_model,
                url=llm_url,
                pedagogical_brief=pedagogical_brief,
            )
        except RuntimeError as exc:
            fallback = build_rule_based_feedback(metrics, flags)
            feedback = f"[LLM unavailable: {exc}]\n\n{fallback}"

    return {
        "label": label,
        "teacher": teacher_path.name,
        "student": student_path.name,
        "duration_teacher": teacher["duration"],
        "duration_student": student["duration"],
        "metrics": metrics,
        "flags": flags,
        "n_voiced_aligned": len(alignment["pitch_errors_cents"]),
        "feedback": feedback,
    }


def print_table(results: list[dict]) -> None:
    header = (
        f"{'Label':<55} {'Ovr':>5} {'Inton':>5} {'Rtm':>5} "
        f"{'Atk':>5} {'Brth':>5} {'Voc':>5} "
        f"{'PitchMAE':>8} {'HNR_d':>6} {'CPP_s':>6} {'H1H2_s':>7} "
        f"{'AlphaD':>7} {'Flags':>5}"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        m = r["metrics"]
        label = r["label"][:54]
        print(
            f"{label:<55} "
            f"{fmt(m['overall_score'],1):>5} "
            f"{fmt(m['intonation_score'],1):>5} "
            f"{fmt(m['rhythm_score'],1):>5} "
            f"{fmt(m['attack_score'],1):>5} "
            f"{fmt(m['breath_score'],1):>5} "
            f"{fmt(m['voice_closure_score'],1):>5} "
            f"{fmt(m['pitch_mean_abs_cents'],1):>8} "
            f"{fmt(m['hnr_diff_db'],1):>6} "
            f"{fmt(m['cpp_mean_student'],1):>6} "
            f"{fmt(m['h1h2_mean_student_db'],1):>7} "
            f"{fmt(m['alpha_ratio_diff_db'],1):>7} "
            f"{len(r['flags']):>5}"
        )


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--llm", action="store_true", help="Generate LLM feedback for each pair"
    )
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--ollama-url", default=OLLAMA_CHAT_URL)
    args = ap.parse_args()

    all_results: list[dict] = []

    for ex_name, ref_file in EXERCISES.items():
        teacher_path = EXERCISES_DIR / ref_file
        if not teacher_path.exists():
            print(f"WARNING: reference not found: {teacher_path}", file=sys.stderr)
            continue

        # Find all 'нет' files for this exercise
        error_files = sorted(
            p
            for p in EXERCISES_DIR.iterdir()
            if p.name.lower().startswith(ex_name.lower()) and "нет" in p.name
        )

        print(f"\n{'='*80}")
        print(f"Exercise: {ex_name}  |  Teacher: {ref_file}")
        print(f"{'='*80}")

        for student_path in error_files:
            error_label = student_path.stem  # filename without ext
            label = f"{ex_name} | {error_label}"
            print(f"\n  Running: {student_path.name} ...", end=" ", flush=True)

            try:
                result = run_pair(
                    teacher_path,
                    student_path,
                    label,
                    with_llm=args.llm,
                    llm_model=args.model,
                    llm_url=args.ollama_url,
                )
                all_results.append(result)
                print(f"OK  (overall={fmt(result['metrics']['overall_score'],1)})")
                print(f"    Flags: {result['flags']}")
                if result.get("feedback"):
                    print(f"    --- LLM feedback ---\n{result['feedback']}\n")
            except Exception as exc:
                print(f"FAILED: {exc}")

    print(f"\n\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print_table(all_results)

    # Save JSON
    out_json = REPORTS_DIR / "exercises_metrics.json"
    safe = []
    for r in all_results:
        safe_r = {k: v for k, v in r.items() if k not in ("metrics",)}
        safe_r["metrics"] = {
            k: (
                float(v)
                if isinstance(v, float | np.floating) and np.isfinite(v)
                else (
                    None
                    if isinstance(v, float | np.floating)
                    else int(v) if isinstance(v, int | np.integer) else v
                )
            )
            for k, v in r["metrics"].items()
            if not isinstance(v, np.ndarray)
        }
        safe.append(safe_r)
    out_json.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\nSaved metrics to {out_json}")


if __name__ == "__main__":
    main()
