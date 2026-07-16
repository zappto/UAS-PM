"""Explainability configuration.

Provides a configuration dataclass to control explainability
parameters and visualization settings.
"""

from dataclasses import dataclass
from src.config.settings import (
    EXPLAINABILITY_REPORTS_DIR,
    EXPLAIN_FIGURES_DIR,
    EXPLAIN_RAW_DIR
)


@dataclass
class ExplainabilityConfig:
    """Configuration for explainability and reporting.

    Attributes:
        output_directory: Base directory for explainability reports.
        figures_directory: Directory for explainability visualizations.
        raw_directory: Directory for raw model explanation files.
        save_figures: Whether to save visualization figures to disk.
        save_csv: Whether to save CSV tables to disk.
        top_k_words: Number of top words to report for local explanations.
        max_features: Max features to show on global/summary plots.
        figure_format: File format for saved figures (e.g., 'png').
        dpi: Dots per inch for saved figures.
    """
    output_directory: str = EXPLAINABILITY_REPORTS_DIR
    figures_directory: str = EXPLAIN_FIGURES_DIR
    raw_directory: str = EXPLAIN_RAW_DIR
    save_figures: bool = True
    save_csv: bool = True
    top_k_words: int = 10
    max_features: int = 20
    figure_format: str = "png"
    dpi: int = 300

    def to_dict(self) -> dict:
        """Serialize configuration to a dictionary."""
        return {
            "output_directory": self.output_directory,
            "figures_directory": self.figures_directory,
            "raw_directory": self.raw_directory,
            "save_figures": self.save_figures,
            "save_csv": self.save_csv,
            "top_k_words": self.top_k_words,
            "max_features": self.max_features,
            "figure_format": self.figure_format,
            "dpi": self.dpi,
        }
