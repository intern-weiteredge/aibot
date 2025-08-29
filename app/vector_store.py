from __future__ import annotations

from typing import List, Dict, Any, Tuple

import json
import os
import numpy as np

from app.config import AppConfig, ensure_dirs


def cosine_distance_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    sims = a_norm @ b_norm.T
    return 1.0 - sims


class VectorStore:
    def __init__(self, cfg: AppConfig, collection_name: str = "faqs"):
        self.cfg = cfg
        ensure_dirs(cfg)
        self.path = os.path.join(cfg.chroma_dir, f"{collection_name}.json")
        self.ids: List[str] = []
        self.embeddings: np.ndarray | None = None
        self.metadatas: List[Dict[str, Any]] = []
        self.documents: List[str] = []
        self._load()

    def _load(self):
        if os.path.exists(self.path):
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.ids = data.get("ids", [])
            self.metadatas = data.get("metadatas", [])
            self.documents = data.get("documents", [])
            embs = np.array(data.get("embeddings", []), dtype=np.float32)
            self.embeddings = embs if embs.size else None

    def _persist(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        data = {
            "ids": self.ids,
            "embeddings": (self.embeddings.tolist() if self.embeddings is not None else []),
            "metadatas": self.metadatas,
            "documents": self.documents,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def clear(self):
        self.ids = []
        self.embeddings = None
        self.metadatas = []
        self.documents = []
        if os.path.exists(self.path):
            os.remove(self.path)

    def add(self, ids: List[str], embeddings, metadatas: List[Dict[str, Any]], documents: List[str]):
        vectors = np.array(embeddings, dtype=np.float32)
        self.ids = list(ids)
        self.embeddings = vectors
        self.metadatas = list(metadatas)
        self.documents = list(documents)
        self._persist()

    def count(self) -> int:
        return len(self.ids)

    def query(self, query_embeddings, n_results: int = 5):
        if self.embeddings is None or len(self.ids) == 0:
            return {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        q = np.array(query_embeddings, dtype=np.float32)
        dists = cosine_distance_matrix(q, self.embeddings)
        topk_idx = np.argsort(dists, axis=1)[:, :n_results]
        out_ids: List[List[str]] = []
        out_dists: List[List[float]] = []
        out_docs: List[List[str]] = []
        out_metas: List[List[Dict[str, Any]]] = []
        for row, idxs in enumerate(topk_idx):
            out_ids.append([self.ids[i] for i in idxs])
            out_dists.append([float(dists[row, i]) for i in idxs])
            out_docs.append([self.documents[i] for i in idxs])
            out_metas.append([self.metadatas[i] for i in idxs])
        return {"ids": out_ids, "distances": out_dists, "documents": out_docs, "metadatas": out_metas}


