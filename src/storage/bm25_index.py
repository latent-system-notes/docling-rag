"""BM25 keyword search index for hybrid retrieval.

This module provides BM25 (Best Matching 25) keyword-based search
to complement vector search with exact term matching capabilities.

Supports incremental updates (add/remove) without full rebuilds.
"""
import pickle
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from ..config import settings, get_logger

logger = get_logger(__name__)


class BM25Index:
    """BM25 index for keyword-based document retrieval with incremental updates.

    Memory management:
    - Automatically saves and unloads from memory after operations
    - Only loads into memory when needed for search
    - Tracks max document count to warn about memory usage
    """

    def __init__(self, max_docs_in_memory: int = 10000):
        """Initialize BM25 index.

        Args:
            max_docs_in_memory: Maximum documents to keep in memory before warning.
                                Default 10000 (~100MB for average documents)
        """
        self.index = None
        self.doc_ids = []
        self.documents = []  # Store original documents for incremental updates
        self.index_path = Path(settings.chroma_persist_dir) / "bm25_index.pkl"
        self.is_dirty = False  # Track if index needs rebuild
        self.max_docs_in_memory = max_docs_in_memory
        self._is_loaded = False  # Track if index is loaded in memory

    def build(self, documents: List[str], doc_ids: List[str]) -> None:
        """Build BM25 index from documents.

        Args:
            documents: List of document texts
            doc_ids: List of corresponding document IDs
        """
        if not documents:
            logger.warning("No documents provided for BM25 index")
            return

        self.documents = documents
        self.doc_ids = doc_ids
        self._rebuild_index()

        logger.info(f"Built BM25 index with {len(documents)} documents")

    def _rebuild_index(self) -> None:
        """Rebuild BM25Okapi index from current documents."""
        if not self.documents:
            self.index = None
            self._is_loaded = False
            return

        # Tokenize documents (simple whitespace + lowercase)
        tokenized_docs = [doc.lower().split() for doc in self.documents]

        # Build BM25 index
        self.index = BM25Okapi(tokenized_docs)
        self.is_dirty = False
        self._is_loaded = True

    def add_documents(self, documents: List[str], doc_ids: List[str]) -> None:
        """Add documents to the index incrementally.

        Args:
            documents: List of document texts to add
            doc_ids: List of corresponding document IDs
        """
        if not documents:
            return

        # Load existing index from disk if not already loaded
        if not self._is_loaded and self.index_path.exists():
            self.load()

        self.documents.extend(documents)
        self.doc_ids.extend(doc_ids)
        self.is_dirty = True  # Mark for rebuild on next search

        # Check memory usage and warn if exceeding threshold
        total_docs = len(self.documents)
        if total_docs > self.max_docs_in_memory:
            logger.warning(
                f"BM25 index has {total_docs} documents (threshold: {self.max_docs_in_memory}). "
                f"Consider using HTTP mode for ChromaDB or increasing max_docs_in_memory."
            )

        logger.info(f"Added {len(documents)} documents to BM25 index (total: {total_docs}, rebuild pending)")

    def add_documents_atomic(self, documents: List[str], doc_ids: List[str]) -> None:
        """Add documents to the index and save atomically.

        This ensures the changes are persisted to disk immediately.
        If save fails, in-memory changes can be rolled back.

        Args:
            documents: List of document texts to add
            doc_ids: List of corresponding document IDs

        Raises:
            Exception: If save operation fails
        """
        if not documents:
            return

        # Load existing index from disk if not already loaded
        if not self._is_loaded and self.index_path.exists():
            self.load()

        # Save snapshot of current state for rollback
        snapshot_docs = self.documents.copy()
        snapshot_ids = self.doc_ids.copy()

        try:
            # Add documents to in-memory index
            self.documents.extend(documents)
            self.doc_ids.extend(doc_ids)

            # Rebuild index immediately
            self._rebuild_index()

            # Save to disk - this is the critical operation
            self.save()

            logger.info(f"Atomically added {len(documents)} documents to BM25 index")

        except Exception as e:
            # Rollback: restore snapshot
            logger.error(f"BM25 atomic add failed, rolling back: {e}")
            self.documents = snapshot_docs
            self.doc_ids = snapshot_ids
            self.is_dirty = True
            raise

    def rollback_in_memory_changes(self) -> None:
        """Rollback in-memory changes by reloading from disk.

        Discards any unsaved changes and restores the last saved state.
        """
        if self.index_path.exists():
            logger.warning("Rolling back BM25 in-memory changes by reloading from disk")
            self.load()
        else:
            # No saved index, clear everything
            logger.warning("No saved BM25 index to rollback to, clearing in-memory state")
            self.index = None
            self.doc_ids = []
            self.documents = []
            self.is_dirty = False
            self._is_loaded = False

    def remove_documents(self, doc_ids_to_remove: List[str]) -> None:
        """Remove documents from the index incrementally.

        Args:
            doc_ids_to_remove: List of document IDs to remove
        """
        if not doc_ids_to_remove:
            return

        # Load existing index from disk if not already loaded
        if not self._is_loaded and self.index_path.exists():
            self.load()

        # Convert to set for faster lookup
        remove_set = set(doc_ids_to_remove)

        # Filter out removed documents
        filtered = [
            (doc, doc_id)
            for doc, doc_id in zip(self.documents, self.doc_ids)
            if doc_id not in remove_set
        ]

        original_count = len(self.documents)
        if len(filtered) < original_count:
            self.documents = [doc for doc, _ in filtered]
            self.doc_ids = [doc_id for _, doc_id in filtered]
            self.is_dirty = True  # Mark for rebuild on next search

            removed_count = original_count - len(filtered)
            logger.info(f"Removed {removed_count} documents from BM25 index (rebuild pending)")

    def save(self, unload_after_save: bool = False) -> None:
        """Save BM25 index to disk.

        Args:
            unload_after_save: If True, unload index from memory after saving.
                              Useful for freeing memory in batch operations.
        """
        # Rebuild if dirty before saving
        if self.is_dirty and self.documents:
            self._rebuild_index()

        if self.index is None:
            logger.warning("No BM25 index to save")
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.index_path, 'wb') as f:
            pickle.dump({
                'index': self.index,
                'doc_ids': self.doc_ids,
                'documents': self.documents,  # Save documents for incremental updates
            }, f)

        logger.info(f"Saved BM25 index to {self.index_path}")

        # Optionally unload from memory to free resources
        if unload_after_save:
            self.unload()

    def load(self) -> bool:
        """Load BM25 index from disk.

        Returns:
            True if loaded successfully, False otherwise
        """
        if not self.index_path.exists():
            logger.warning(f"BM25 index not found at {self.index_path}")
            return False

        try:
            with open(self.index_path, 'rb') as f:
                data = pickle.load(f)
                self.index = data['index']
                self.doc_ids = data['doc_ids']
                self.documents = data.get('documents', [])  # Load documents (backward compat)
                self.is_dirty = False
                self._is_loaded = True

            logger.info(f"Loaded BM25 index with {len(self.doc_ids)} documents into memory")
            return True
        except Exception as e:
            logger.error(f"Failed to load BM25 index: {e}")
            return False

    def unload(self) -> None:
        """Unload BM25 index from memory to free resources.

        The index remains saved on disk and can be loaded again when needed.
        """
        if self._is_loaded or self.index is not None:
            self.index = None
            self.doc_ids = []
            self.documents = []
            self.is_dirty = False
            self._is_loaded = False
            logger.info("BM25 index unloaded from memory")

    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """Search BM25 index for query.

        Automatically loads index from disk if needed.
        Rebuilds index if dirty (after add/remove operations).

        Args:
            query: Search query text
            top_k: Number of results to return

        Returns:
            List of (doc_id, score) tuples sorted by score (descending)
        """
        # Auto-load from disk if not in memory
        if not self._is_loaded and self.index_path.exists():
            self.load()

        # Rebuild if dirty (lazy rebuild on search)
        if self.is_dirty and self.documents:
            logger.info("Rebuilding BM25 index before search...")
            self._rebuild_index()
            self.save()  # Save after rebuild

        if self.index is None:
            logger.warning("BM25 index not initialized")
            return []

        # Tokenize query
        tokenized_query = query.lower().split()

        # Get BM25 scores
        scores = self.index.get_scores(tokenized_query)

        # Sort by score and get top_k
        doc_scores = list(zip(self.doc_ids, scores))
        doc_scores.sort(key=lambda x: x[1], reverse=True)

        return doc_scores[:top_k]

    def clear(self) -> None:
        """Clear the BM25 index from memory and disk."""
        self.index = None
        self.doc_ids = []
        self.documents = []
        self.is_dirty = False
        self._is_loaded = False

        if self.index_path.exists():
            self.index_path.unlink()
            logger.info(f"Cleared BM25 index at {self.index_path}")


# Global BM25 index instance
_bm25_index = BM25Index()


def get_bm25_index() -> BM25Index:
    """Get the global BM25 index instance."""
    return _bm25_index


def rebuild_bm25_index() -> None:
    """Rebuild BM25 index from ChromaDB collection."""
    from .chroma_client import get_chroma_client

    logger.info("Rebuilding BM25 index from ChromaDB...")

    client = get_chroma_client()
    collection = client.get_collection(settings.chroma_collection_name)

    # Get all documents
    results = collection.get()

    if not results['ids']:
        logger.warning("No documents in ChromaDB to index")
        return

    # Build BM25 index
    bm25_index = get_bm25_index()
    bm25_index.build(
        documents=results['documents'],
        doc_ids=results['ids']
    )
    bm25_index.save()

    logger.info(f"BM25 index rebuilt with {len(results['ids'])} documents")
