"""
Local RAG layer for Interview Forge.

Pipeline:
PDF/TXT -> text extraction -> chunking -> Hugging Face tokenization
-> SentenceTransformer embeddings -> FAISS index -> similarity retrieval.

The FAISS index is intentionally session-local for the free MVP. It is rebuilt
when a user uploads a new CV/job description and is not a hosted database.
"""

from io import BytesIO
from typing import Dict, List

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer


class RAGPipeline:
    EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    TOKENIZER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunk_tokens: int = 350, overlap_tokens: int = 60):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(self.TOKENIZER_MODEL)
        self.embedder = SentenceTransformer(self.EMBEDDING_MODEL)
        self.index = None
        self.records: List[Dict] = []

    @staticmethod
    def extract_pdf_bytes(data: bytes) -> str:
        reader = PdfReader(BytesIO(data))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages.append(text)
        return "\n\n".join(pages).strip()

    def tokenize(self, text: str):
        return self.tokenizer.encode(text, add_special_tokens=False)

    def detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def chunk_text(self, text: str, source: str) -> List[Dict]:
        token_ids = self.tokenize(text)
        if not token_ids:
            return []

        chunks = []
        start = 0
        chunk_no = 1
        step = max(1, self.chunk_tokens - self.overlap_tokens)

        while start < len(token_ids):
            end = min(start + self.chunk_tokens, len(token_ids))
            chunk = self.detokenize(token_ids[start:end]).strip()
            if chunk:
                chunks.append(
                    {
                        "source": source,
                        "chunk_id": chunk_no,
                        "text": chunk,
                    }
                )
                chunk_no += 1
            if end >= len(token_ids):
                break
            start += step

        return chunks

    def build(self, documents: List[Dict]) -> Dict:
        self.records = []
        for doc in documents:
            text = doc.get("text", "").strip()
            source = doc.get("source", "DOCUMENT")
            if text:
                self.records.extend(self.chunk_text(text, source))

        if not self.records:
            raise ValueError("No usable text was extracted from the uploaded documents.")

        vectors = self.embedder.encode(
            [r["text"] for r in self.records],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        vectors = np.asarray(vectors, dtype="float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

        return {
            "documents": len(documents),
            "chunks": len(self.records),
            "embedding_dimension": int(vectors.shape[1]),
        }

    def retrieve(self, query: str, k: int = 5) -> List[Dict]:
        if self.index is None:
            raise RuntimeError("Build the RAG index before retrieval.")

        query_vector = self.embedder.encode(
            [query],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_vector = np.asarray(query_vector, dtype="float32")
        k = min(k, self.index.ntotal)
        scores, indices = self.index.search(query_vector, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(self.records[int(idx)])
            item["score"] = float(score)
            results.append(item)
        return results

    def context(self, query: str, k: int = 5, max_chars: int = 9000) -> str:
        hits = self.retrieve(query, k=k)
        blocks = []
        total = 0
        for hit in hits:
            block = f"[{hit['source']} | score={hit['score']:.3f}]\n{hit['text']}"
            if total + len(block) > max_chars:
                break
            blocks.append(block)
            total += len(block)
        return "\n\n".join(blocks)
