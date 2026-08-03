"""Configuration module for PaperMind AI.

Manages environment variables, default settings, vector store configs, and model parameters.
"""

import os
from pathlib import Path
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Load environment variables from .env if present
load_dotenv()


class AppConfig(BaseModel):
    """Application settings and parameter models."""
    
    app_name: str = "PaperMind AI"
    version: str = "0.1.0"
    
    # LLM Settings
    llm_provider: str = Field(
        default_factory=lambda: os.getenv("PAPERMIND_LLM_PROVIDER", "offline")
    )
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY")
    )
    hf_token: str | None = Field(
        default_factory=lambda: os.getenv("HF_TOKEN")
    )
    
    # Embedding Settings
    embedding_model_name: str = Field(
        default_factory=lambda: os.getenv("PAPERMIND_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    chunk_size: int = Field(
        default_factory=lambda: int(os.getenv("PAPERMIND_CHUNK_SIZE", "500"))
    )
    chunk_overlap: int = Field(
        default_factory=lambda: int(os.getenv("PAPERMIND_CHUNK_OVERLAP", "50"))
    )
    
    # Directory Settings
    base_dir: Path = Path(__file__).resolve().parent.parent.parent
    cache_dir: Path = Field(
        default_factory=lambda: Path(os.getenv("PAPERMIND_CACHE_DIR", "./cache"))
    )
    
    def ensure_directories(self) -> None:
        """Create cache and output directories if they do not exist."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)


config = AppConfig()
config.ensure_directories()
