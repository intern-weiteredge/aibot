from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Any

import pandas as pd


@dataclass
class FAQItem:
    question: str
    answer: str


def load_csv(source: Any) -> List[FAQItem]:
    """Load CSV from a filepath or a file-like object (e.g., Streamlit UploadedFile)."""
    df = pd.read_csv(source)
    cols = {c.lower(): c for c in df.columns}
    if "question" not in cols or "answer" not in cols:
        raise ValueError("CSV must contain 'question' and 'answer' columns")
    q_col = cols["question"]
    a_col = cols["answer"]
    items: List[FAQItem] = []
    for _, row in df.iterrows():
        q = str(row[q_col]).strip()
        a = str(row[a_col]).strip()
        if q and a:
            items.append(FAQItem(q, a))
    return items


def load_txt(source: Any) -> List[FAQItem]:
    """Load TXT from a filepath or a file-like object (e.g., Streamlit UploadedFile)."""
    if hasattr(source, "read"):
        raw = source.read()
        # Streamlit may return bytes
        content = raw.decode("utf-8", errors="ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
    else:
        with open(source, "r", encoding="utf-8") as f:
            content = f.read()
    chunks = [c.strip() for c in content.split("\n\n") if c.strip()]
    items: List[FAQItem] = []
    for chunk in chunks:
        lines = [l.strip() for l in chunk.splitlines() if l.strip()]
        q, a = None, None
        for line in lines:
            if line.lower().startswith("q:"):
                q = line[2:].strip()
            elif line.lower().startswith("a:"):
                a = line[2:].strip()
        if q and a:
            items.append(FAQItem(q, a))
    return items


def to_documents(items: List[FAQItem]) -> Tuple[list[str], list[str], list[dict]]:
    ids = [f"faq-{i}" for i in range(len(items))]
    documents = [f"Q: {it.question}\nA: {it.answer}" for it in items]
    metadatas = [{"question": it.question, "answer": it.answer} for it in items]
    return ids, documents, metadatas


