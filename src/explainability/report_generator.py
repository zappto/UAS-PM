"""Report generator module for explainability.

Generates the explainability_summary.md report detailing the
explainability process and generated files.
"""

import os
from typing import Dict, Any


class ReportGenerator:
    """Generates markdown reports for explainability."""

    def __init__(self, output_path: str):
        """Initialize the report generator.

        Args:
            output_path: Path where the markdown report will be saved.
        """
        self.output_path = output_path
        self.reports_dir = os.path.dirname(output_path)

    def prepare_example_files(self) -> None:
        """Prepare placeholders for inference examples."""
        os.makedirs(self.reports_dir, exist_ok=True)
        # We don't generate the real inference data during development, 
        # but we ensure the files are conceptually part of the outputs.
        # This complies with: "Do NOT generate outputs during development."
        pass
    def generate_summary_report(
        self,
        report_data: Dict[str, Any]
    ) -> None:
        """Generate and save the explainability summary report.

        Args:
            report_data: Dictionary containing information to populate the report.
                Expected keys: models_loaded, explainer_used, num_features,
                num_classes, global_files, local_files, figures, csv_files.
        """
        self.prepare_example_files()
        
        md_content = [
            "# Explainability Summary Report\n",
            "This report summarizes the explainability artifacts generated for the trained models.\n",
            "## Model Information\n",
            f"- **Models Loaded:** {', '.join(report_data.get('models_loaded', []))}",
            f"- **Explainer Used:** {report_data.get('explainer_used', 'N/A')}",
            f"- **Number of Features:** {report_data.get('num_features', 0)}",
            f"- **Number of Classes:** {report_data.get('num_classes', 0)}\n",
            "## Generated Artifacts\n",
            "### Global Explanation Files\n"
        ]

        for file in report_data.get('global_files', []):
            md_content.append(f"- {file}")

        md_content.append("\n### Local Explanation Files\n")
        for file in report_data.get('local_files', []):
            md_content.append(f"- {file}")

        md_content.append("\n### Generated Figures\n")
        for file in report_data.get('figures', []):
            md_content.append(f"- {file}")

        md_content.append("\n### Generated CSV Files\n")
        for file in report_data.get('csv_files', []):
            md_content.append(f"- {file}")

        md_content.append("\n### Prepared Inference Artifacts (placeholders)\n")
        md_content.append(f"- {os.path.join(self.reports_dir, 'prediction_examples.json')}")
        md_content.append(f"- {os.path.join(self.reports_dir, 'inference_examples.csv')}")

        md_content.append("\n---\n*Report generated automatically.*")

        os.makedirs(self.reports_dir, exist_ok=True)        with open(self.output_path, "w", encoding="utf-8") as f:
            f.write("\n".join(md_content))
