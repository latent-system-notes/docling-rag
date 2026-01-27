import json
import math
import sqlite3
import zlib
from pathlib import Path
from typing import List, Tuple, Optional

from ..config import get_chroma_persist_dir, COLLECTION_NAME, get_logger

logger = get_logger(__name__)

class BM25SqliteIndex:
    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (get_chroma_persist_dir() / "bm25.sqlite3")
        self.conn = None
        self.idf_scores = {}
        self.avg_doc_len = 0.0
        self.num_docs = 0
        self._is_loaded = False
        self.k1 = 1.5
        self.b = 0.75
        self._connect()
        self._initialize_schema()
        self._load_statistics()

    def _connect(self):
        if self.conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self.conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
            self.conn.execute("PRAGMA journal_mode=WAL")
            self.conn.execute("PRAGMA synchronous=NORMAL")

    def _initialize_schema(self):
        with self.conn:
            self.conn.execute("""CREATE TABLE IF NOT EXISTS bm25_documents (
                chunk_id TEXT PRIMARY KEY, tokens BLOB NOT NULL, doc_length INTEGER NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            self.conn.execute("""CREATE TABLE IF NOT EXISTS bm25_statistics (
                key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)""")
            self.conn.execute("CREATE INDEX IF NOT EXISTS idx_chunk_id ON bm25_documents(chunk_id)")

    def _tokenize(self, text: str) -> List[str]:
        return text.lower().split()

    def build(self, documents: List[str], doc_ids: List[str]) -> None:
        if not documents:
            return
        with self.conn:
            self.conn.execute("DELETE FROM bm25_documents")
            self.conn.execute("DELETE FROM bm25_statistics")
        batch_size = 1000
        for i in range(0, len(documents), batch_size):
            self._add_batch(documents[i:i+batch_size], doc_ids[i:i+batch_size])
        self._update_statistics()
        self._is_loaded = True

    def _add_batch(self, documents: List[str], doc_ids: List[str]):
        with self.conn:
            for doc_id, text in zip(doc_ids, documents):
                tokens = self._tokenize(text)
                tokens_blob = zlib.compress(json.dumps(tokens).encode('utf-8'))
                self.conn.execute("INSERT OR REPLACE INTO bm25_documents (chunk_id, tokens, doc_length) VALUES (?, ?, ?)",
                    (doc_id, tokens_blob, len(tokens)))

    def add_documents(self, documents: List[str], doc_ids: List[str]) -> None:
        if documents:
            self._add_batch(documents, doc_ids)

    def add_documents_atomic(self, documents: List[str], doc_ids: List[str]) -> None:
        if not documents:
            return
        try:
            with self.conn:
                self._add_batch(documents, doc_ids)
                self._update_statistics()
        except Exception as e:
            raise

    def rollback_in_memory_changes(self) -> None:
        self._load_statistics()

    def remove_documents(self, doc_ids_to_remove: List[str]) -> None:
        if not doc_ids_to_remove:
            return
        with self.conn:
            placeholders = ','.join('?' * len(doc_ids_to_remove))
            self.conn.execute(f"DELETE FROM bm25_documents WHERE chunk_id IN ({placeholders})", doc_ids_to_remove)
        self._update_statistics()

    def _update_statistics(self):
        cursor = self.conn.execute("SELECT COUNT(*) FROM bm25_documents")
        self.num_docs = cursor.fetchone()[0]
        if self.num_docs == 0:
            self.idf_scores = {}
            self.avg_doc_len = 0.0
            return
        cursor = self.conn.execute("SELECT AVG(doc_length) FROM bm25_documents")
        self.avg_doc_len = cursor.fetchone()[0] or 0.0
        term_doc_counts = {}
        cursor = self.conn.execute("SELECT tokens FROM bm25_documents")
        for (tokens_blob,) in cursor:
            tokens = json.loads(zlib.decompress(tokens_blob).decode('utf-8'))
            for token in set(tokens):
                term_doc_counts[token] = term_doc_counts.get(token, 0) + 1
        self.idf_scores = {term: math.log((self.num_docs - dc + 0.5) / (dc + 0.5) + 1) for term, dc in term_doc_counts.items()}
        self._save_statistics()

    def _save_statistics(self):
        with self.conn:
            self.conn.execute("INSERT OR REPLACE INTO bm25_statistics (key, value) VALUES (?, ?)", ("idf_scores", json.dumps(self.idf_scores)))
            self.conn.execute("INSERT OR REPLACE INTO bm25_statistics (key, value) VALUES (?, ?)", ("avg_doc_len", str(self.avg_doc_len)))
            self.conn.execute("INSERT OR REPLACE INTO bm25_statistics (key, value) VALUES (?, ?)", ("num_docs", str(self.num_docs)))

    def _load_statistics(self):
        cursor = self.conn.execute("SELECT key, value FROM bm25_statistics")
        stats_found = False
        for key, value in cursor:
            stats_found = True
            if key == "idf_scores":
                self.idf_scores = json.loads(value)
            elif key == "avg_doc_len":
                self.avg_doc_len = float(value)
            elif key == "num_docs":
                self.num_docs = int(value)
        if stats_found:
            self._is_loaded = True
        else:
            cursor = self.conn.execute("SELECT COUNT(*) FROM bm25_documents")
            if cursor.fetchone()[0] > 0:
                self._update_statistics()

    def save(self, unload_after_save: bool = False) -> None:
        pass

    def load(self) -> bool:
        try:
            self._load_statistics()
            return True
        except Exception:
            return False

    def unload(self) -> None:
        self.idf_scores = {}
        self.avg_doc_len = 0.0
        self.num_docs = 0
        self._is_loaded = False

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        if not self._is_loaded:
            self._load_statistics()
        if self.num_docs == 0:
            return []
        query_tokens = self._tokenize(query)
        cursor = self.conn.execute("SELECT chunk_id, tokens, doc_length FROM bm25_documents")
        scores = {}
        for chunk_id, tokens_blob, doc_len in cursor:
            tokens = json.loads(zlib.decompress(tokens_blob).decode('utf-8'))
            score = 0.0
            for query_token in query_tokens:
                if query_token not in self.idf_scores:
                    continue
                tf = tokens.count(query_token)
                if tf == 0:
                    continue
                idf = self.idf_scores[query_token]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)
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
        self._is_loaded = False

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
    from .chroma_client import get_chroma_client
    client = get_chroma_client()
    collection = client.get_collection(COLLECTION_NAME)
    results = collection.get()
    if not results['ids']:
        return
    bm25_index = get_bm25_index()
    bm25_index.build(documents=results['documents'], doc_ids=results['ids'])
    bm25_index.save()
