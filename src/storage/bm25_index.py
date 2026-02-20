import json
import math
import sqlite3
import zlib
from pathlib import Path

from ..config import get_chroma_persist_dir, COLLECTION_NAME, get_logger

logger = get_logger(__name__)

class BM25SqliteIndex:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (get_chroma_persist_dir() / "bm25.sqlite3")
        self.idf_scores = {}
        self.avg_doc_len = 0.0
        self.num_docs = 0
        self.k1 = 1.5
        self.b = 0.75
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self._initialize_schema()
        self._load_statistics()

    def _initialize_schema(self):
        with self.conn:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS bm25_documents (
                chunk_id TEXT PRIMARY KEY, tokens BLOB NOT NULL, doc_length INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS bm25_statistics (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON bm25_documents(chunk_id)")

    def _tokenize(self, text: str) -> list[str]:
        return text.lower().split()

    def build(self, documents: list[str], doc_ids: list[str]) -> None:
        if not documents:
            return
        with self.conn:
            self.conn.execute("DELETE FROM bm25_documents")
            self.conn.execute("DELETE FROM bm25_statistics")
        for i in range(0, len(documents), 1000):
            self._add_batch(documents[i:i+1000], doc_ids[i:i+1000])
        self._update_statistics()

    def _add_batch(self, documents: list[str], doc_ids: list[str]):
        with self.conn:
            for doc_id, text in zip(doc_ids, documents):
                tokens = self._tokenize(text)
                self.conn.execute("INSERT OR REPLACE INTO bm25_documents (chunk_id, tokens, doc_length) VALUES (?, ?, ?)",
                    (doc_id, zlib.compress(json.dumps(tokens).encode('utf-8')), len(tokens)))

    def add_documents_atomic(self, documents: list[str], doc_ids: list[str]) -> None:
        if not documents:
            return
        with self.conn:
            self._add_batch(documents, doc_ids)
            self._update_statistics()

    def remove_documents(self, doc_ids_to_remove: list[str]) -> None:
        if not doc_ids_to_remove:
            return
        with self.conn:
            placeholders = ','.join('?' * len(doc_ids_to_remove))
            self.conn.execute(f"DELETE FROM bm25_documents WHERE chunk_id IN ({placeholders})", doc_ids_to_remove)
        self._update_statistics()

    def _update_statistics(self):
        self.num_docs = self.conn.execute("SELECT COUNT(*) FROM bm25_documents").fetchone()[0]
        if self.num_docs == 0:
            self.idf_scores = {}
            self.avg_doc_len = 0.0
            return
        self.avg_doc_len = self.conn.execute("SELECT AVG(doc_length) FROM bm25_documents").fetchone()[0] or 0.0
        term_doc_counts = {}
        for (tokens_blob,) in self.conn.execute("SELECT tokens FROM bm25_documents"):
            for token in set(json.loads(zlib.decompress(tokens_blob))):
                term_doc_counts[token] = term_doc_counts.get(token, 0) + 1
        self.idf_scores = {term: math.log((self.num_docs - dc + 0.5) / (dc + 0.5) + 1) for term, dc in term_doc_counts.items()}
        with self.conn:
            for key, value in [("idf_scores", json.dumps(self.idf_scores)), ("avg_doc_len", str(self.avg_doc_len)), ("num_docs", str(self.num_docs))]:
                self.conn.execute("INSERT OR REPLACE INTO bm25_statistics (key, value) VALUES (?, ?)", (key, value))

    def _load_statistics(self):
        stats = dict(self.conn.execute("SELECT key, value FROM bm25_statistics"))
        if not stats:
            if self.conn.execute("SELECT COUNT(*) FROM bm25_documents").fetchone()[0] > 0:
                self._update_statistics()
            return
        self.idf_scores = json.loads(stats.get("idf_scores", "{}"))
        self.avg_doc_len = float(stats.get("avg_doc_len", 0.0))
        self.num_docs = int(stats.get("num_docs", 0))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if self.num_docs == 0:
            self._load_statistics()
        if self.num_docs == 0:
            return []
        query_tokens = self._tokenize(query)
        scores = {}
        for chunk_id, tokens_blob, doc_len in self.conn.execute("SELECT chunk_id, tokens, doc_length FROM bm25_documents"):
            tokens = json.loads(zlib.decompress(tokens_blob))
            score = 0.0
            for qt in query_tokens:
                if qt not in self.idf_scores:
                    continue
                tf = tokens.count(qt)
                if tf == 0:
                    continue
                score += self.idf_scores[qt] * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len)))
            if score > 0:
                scores[chunk_id] = score
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

    def clear(self) -> None:
        with self.conn:
            self.conn.execute("DELETE FROM bm25_documents")
            self.conn.execute("DELETE FROM bm25_statistics")
        self.idf_scores = {}
        self.avg_doc_len = 0.0
        self.num_docs = 0

    def close(self):
        if self.conn:
            self.conn.close()
            self.conn = None

_bm25_index = None

def get_bm25_index() -> BM25SqliteIndex:
    global _bm25_index
    if _bm25_index is None:
        _bm25_index = BM25SqliteIndex()
    return _bm25_index

def rebuild_bm25_index() -> None:
    from .chroma_client import get_chroma_client, COLLECTION_NAME
    results = get_chroma_client().get_collection(COLLECTION_NAME).get()
    if results['ids']:
        get_bm25_index().build(documents=results['documents'], doc_ids=results['ids'])
