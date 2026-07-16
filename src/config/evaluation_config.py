"""Evaluation configuration for the model evaluation stage.

Provides a configuration dataclass to control all evaluation
parameters and visualization settings.
"""

from dataclasses import dataclass, field

from src.config.settings import EVALUATION_REPORTS_DIR


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation and reporting.

    Attributes:
        average: Averaging method for multi-class metrics ('macro', 'weighted').
        output_directory: Base directory for evaluation reports.
        save_figures: Whether to save visualization figures to disk.
        save_reports: Whether to save text/csv reports to disk.
        figure_format: File format for saved figures (e.g., 'png').
        dpi: Dots per inch for saved figures.
        class_labels: List of class label strings in consistent order.
    """

    average: str = "macro"
    output_directory: str = EVALUATION_REPORTS_DIR
    save_figures: bool = True
    save_reports: bool = True
    figure_format: str = "png"
    dpi: int = 300
    class_labels: list[str] = field(
        default_factory=lambda: [
            "normal",
            "hate_speech",
            "insult",
            "harassment",
            "threat",
            "sexually_explicit",
        ]
    )

    def to_dict(self) -> dict:
        """Serialize configuration to a dictionary."""
        return {
            "average": self.average,
            "output_directory": self.output_directory,
            "save_figures": self.save_figures,
            "save_reports": self.save_reports,
            "figure_format": self.figure_format,
            "dpi": self.dpi,
            "class_labels": self.class_labels,
        }
