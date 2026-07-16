"""Error Analysis configuration.

Provides a configuration dataclass to control all error analysis
parameters and visualization settings.
"""

from dataclasses import dataclass

from src.config.settings import ERROR_ANALYSIS_REPORTS_DIR


@dataclass
class ErrorAnalysisConfig:
    """Configuration for error analysis and reporting.

    Attributes:
        output_directory: Base directory for error analysis reports.
        save_figures: Whether to save visualization figures to disk.
        save_tables: Whether to save CSV tables to disk.
        top_k_errors: Number of top errors/confusions to report.
        figure_format: File format for saved figures (e.g., 'png').
        dpi: Dots per inch for saved figures.
    """
    output_directory: str = ERROR_ANALYSIS_REPORTS_DIR
    save_figures: bool = True
    save_tables: bool = True
    top_k_errors: int = 10
    figure_format: str = "png"
    dpi: int = 300

    def to_dict(self) -> dict:
        """Serialize configuration to a dictionary."""
        return {
            "output_directory": self.output_directory,
            "save_figures": self.save_figures,
            "save_tables": self.save_tables,
            "top_k_errors": self.top_k_errors,
            "figure_format": self.figure_format,
            "dpi": self.dpi,
        }
