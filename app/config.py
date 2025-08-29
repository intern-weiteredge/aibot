import os
from dataclasses import dataclass


@dataclass
class AppConfig:
    chroma_dir: str = os.getenv("CHROMA_DIR", "data/chroma")
    embeddings_provider: str = os.getenv("EMBEDDINGS_PROVIDER", "sentence-transformers")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    default_st_model: str = os.getenv("ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    top_k: int = int(os.getenv("TOP_K", "5"))
    distance_threshold: float = float(os.getenv("DISTANCE_THRESHOLD", "0.35"))


def ensure_dirs(cfg: AppConfig) -> None:
    os.makedirs(cfg.chroma_dir, exist_ok=True)

