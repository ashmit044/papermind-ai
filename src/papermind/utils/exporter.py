"""Export digest results to Markdown and JSON formats."""

import json
from pathlib import Path
from typing import Dict, Any
from papermind.utils.logger import logger


class DigestExporter:
    """Handles exporting research paper digests and RAG insights to disk."""

    @staticmethod
    def export_to_json(data: Dict[str, Any], output_path: str | Path) -> Path:
        """Export dictionary data to formatted JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Successfully exported digest to JSON: {path}")
        return path

    @staticmethod
    def export_to_markdown(summary: Dict[str, Any], output_path: str | Path) -> Path:
        """Export paper digest summary into a clean Markdown report."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        title = summary.get("title", "Research Paper Digest")
        overview = summary.get("overview", "")
        key_insights = summary.get("key_insights", [])
        methodology = summary.get("methodology", "")
        limitations = summary.get("limitations", "")

        md_content = f"# 📄 Paper Digest: {title}\n\n"
        md_content += f"## 📌 Executive Summary\n{overview}\n\n"
        
        md_content += "## 💡 Key Takeaways & Contributions\n"
        for idx, insight in enumerate(key_insights, 1):
            md_content += f"{idx}. {insight}\n"
        md_content += "\n"
        
        if methodology:
            md_content += f"## ⚙️ Methodology & Architecture\n{methodology}\n\n"
            
        if limitations:
            md_content += f"## ⚠️ Limitations & Future Work\n{limitations}\n\n"
            
        md_content += "---\n*Generated automatically by [PaperMind AI](https://github.com/ashmit044/papermind-ai)*\n"

        with open(path, "w", encoding="utf-8") as f:
            f.write(md_content)

        logger.info(f"Successfully exported Markdown digest: {path}")
        return path
