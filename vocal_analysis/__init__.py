"""vocal_analysis — acoustic analysis of teacher / student vocal pairs."""

from .alignment import align_by_pitch
from .evaluation import build_flags, build_text_report, evaluate
from .features import attack_metrics, extract_features
from .feedback import (
    build_pedagogical_brief,
    build_rule_based_feedback,
    generate_feedback,
)
from .report import (
    build_analysis_data_md,
    build_paths,
    build_student_md,
    metrics_to_json_safe,
)
from .beautiful_report import build_beautiful_student_md
from .html_report import build_html_student_report, md_file_to_html
from .visualization import save_all_plots

__all__ = [
    "align_by_pitch",
    "attack_metrics",
    "build_analysis_data_md",
    "build_beautiful_student_md",
    "build_flags",
    "build_html_student_report",
    "md_file_to_html",
    "build_paths",
    "build_pedagogical_brief",
    "build_rule_based_feedback",
    "build_student_md",
    "build_text_report",
    "evaluate",
    "extract_features",
    "generate_feedback",
    "metrics_to_json_safe",
    "save_all_plots",
]
