"""Tests for SQLite-based BM25 index."""
import sys
from pathlib import Path
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.storage.bm25_index import BM25SqliteIndex


def test_bm25_sqlite_basic():
    """Test basic add and search operations."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test_bm25.sqlite3"
        index = BM25SqliteIndex(db_path)

        # Add documents (texts first, IDs second)
        texts = [
            "the quick brown fox jumps over the lazy dog",
            "the lazy dog sleeps all day",
            "quick brown animals run fast"
        ]
        chunk_ids = ["doc1", "doc2", "doc3"]
        index.add_documents_atomic(texts, chunk_ids)

        # Test search
        results = index.search("quick fox", top_k=10)

        print(f"   Query: 'quick fox'")
        print(f"   Results: {len(results)}")
        for i, (doc_id, score) in enumerate(results[:5]):
            print(f"     {i+1}. {doc_id}: {score:.4f}")

        # Verify - at least doc1 should match (has both quick and fox)
        assert len(results) >= 1, f"Expected at least 1 result, got {len(results)}"
        assert results[0][0] == "doc1", f"Expected doc1 first, got {results[0][0]}"
        assert results[0][1] > 0, "Score should be positive"

        print(f"[PASS] Basic search test passed")

        index.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bm25_sqlite_persistence():
    """Test that index persists across restarts."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test_bm25.sqlite3"

        # Create and populate (texts first, IDs second)
        index1 = BM25SqliteIndex(db_path)
        index1.add_documents_atomic(["hello world", "goodbye world"], ["doc1", "doc2"])
        initial_docs = index1.num_docs
        index1.close()

        # Reload and verify
        index2 = BM25SqliteIndex(db_path)
        assert index2.num_docs == initial_docs, "Document count should match after reload"

        results = index2.search("hello", top_k=1)
        assert len(results) == 1, "Should find hello document"
        assert results[0][0] == "doc1", "Should find doc1"

        print(f"[PASS] Persistence test passed")
        print(f"   Documents persisted: {index2.num_docs}")

        index2.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bm25_sqlite_remove():
    """Test document removal."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test_bm25.sqlite3"
        index = BM25SqliteIndex(db_path)

        # Add documents (texts first, IDs second)
        index.add_documents_atomic(
            ["first document", "second document", "third document"],
            ["doc1", "doc2", "doc3"]
        )
        assert index.num_docs == 3

        # Remove one document
        index.remove_documents(["doc2"])
        assert index.num_docs == 2

        # Verify it's gone
        results = index.search("second", top_k=10)
        assert len(results) == 0, "Removed document should not be found"

        print(f"[PASS] Removal test passed")
        print(f"   Documents remaining: {index.num_docs}")

        index.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bm25_sqlite_empty_query():
    """Test empty and missing terms."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test_bm25.sqlite3"
        index = BM25SqliteIndex(db_path)

        # texts first, IDs second
        index.add_documents_atomic(["hello world"], ["doc1"])

        # Query for term that doesn't exist
        results = index.search("nonexistent term", top_k=10)
        assert len(results) == 0, "Should return no results for nonexistent terms"

        print(f"[PASS] Empty query test passed")

        index.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bm25_sqlite_memory_usage():
    """Test memory-efficient design (documents not loaded)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_bm25.sqlite3"
        index = BM25SqliteIndex(db_path)

        # Add many documents
        num_docs = 1000
        chunk_ids = [f"doc{i}" for i in range(num_docs)]
        texts = [f"document number {i} with some random text" for i in range(num_docs)]

        index.build(texts, chunk_ids)

        # Verify documents are in database, not in memory
        assert index.num_docs == num_docs
        # SQLite index should NOT have a 'documents' attribute storing all text
        assert not hasattr(index, 'documents') or not isinstance(getattr(index, 'documents', None), list)

        # Search should still work
        results = index.search("document number 42", top_k=5)
        assert len(results) > 0, "Search should work even though docs not in memory"

        print(f"[PASS] Memory usage test passed")
        print(f"   Indexed {num_docs} documents")
        print(f"   Statistics in memory only (not full text)")

        index.close()


def test_bm25_sqlite_scoring():
    """Test that BM25 scoring makes sense."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test_bm25.sqlite3"
        index = BM25SqliteIndex(db_path)

        # Add documents with varying relevance
        texts = [
            "python programming language",
            "python snake reptile",
            "java programming language"
        ]
        ids = ["doc1", "doc2", "doc3"]
        index.add_documents_atomic(texts, ids)

        # Query for "python programming"
        results = index.search("python programming", top_k=3)

        # doc1 should score highest (has both terms)
        assert results[0][0] == "doc1", f"Expected doc1 first, got {results[0][0]}"
        assert results[0][1] > results[1][1], "First result should score higher than second"

        print(f"[PASS] Scoring test passed")
        print(f"   Query: 'python programming'")
        for i, (doc_id, score) in enumerate(results):
            print(f"     {i+1}. {doc_id}: {score:.4f}")

        index.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bm25_sqlite_atomic_transaction():
    """Test atomic transaction behavior."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test_bm25.sqlite3"
        index = BM25SqliteIndex(db_path)

        # Add documents atomically
        texts = ["doc one", "doc two", "doc three"]
        ids = ["id1", "id2", "id3"]
        index.add_documents_atomic(texts, ids)

        # Verify all documents were added
        assert index.num_docs == 3

        # Close and reopen to verify persistence
        index.close()
        index2 = BM25SqliteIndex(db_path)
        assert index2.num_docs == 3

        print(f"[PASS] Atomic transaction test passed")
        print(f"   All 3 documents committed atomically")

        index2.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bm25_sqlite_statistics_accuracy():
    """Test that statistics (doc count, avg doc length) are accurate."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test_bm25.sqlite3"
        index = BM25SqliteIndex(db_path)

        # Add documents with known lengths
        texts = [
            "one two three",        # 3 tokens
            "four five",            # 2 tokens
            "six seven eight nine"  # 4 tokens
        ]
        ids = ["doc1", "doc2", "doc3"]
        index.add_documents_atomic(texts, ids)

        # Check document count
        assert index.num_docs == 3

        # Average doc length should be (3 + 2 + 4) / 3 = 3.0
        assert abs(index.avg_doc_len - 3.0) < 0.01, f"Expected avg_doc_len ~3.0, got {index.avg_doc_len}"

        print(f"[PASS] Statistics accuracy test passed")
        print(f"   Document count: {index.num_docs}")
        print(f"   Average doc length: {index.avg_doc_len:.2f}")

        index.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bm25_sqlite_multiple_additions():
    """Test adding documents in multiple batches."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test_bm25.sqlite3"
        index = BM25SqliteIndex(db_path)

        # First batch
        index.add_documents_atomic(["first batch doc"], ["batch1_doc1"])
        assert index.num_docs == 1

        # Second batch
        index.add_documents_atomic(["second batch doc"], ["batch2_doc1"])
        assert index.num_docs == 2

        # Third batch
        index.add_documents_atomic(["third batch doc"], ["batch3_doc1"])
        assert index.num_docs == 3

        # Verify all are searchable
        results = index.search("batch", top_k=10)
        assert len(results) == 3, f"Expected 3 results, got {len(results)}"

        print(f"[PASS] Multiple additions test passed")
        print(f"   Added documents in 3 separate batches")
        print(f"   Total documents: {index.num_docs}")

        index.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_bm25_sqlite_duplicate_ids():
    """Test handling of duplicate document IDs."""
    tmpdir = tempfile.mkdtemp()
    try:
        db_path = Path(tmpdir) / "test_bm25.sqlite3"
        index = BM25SqliteIndex(db_path)

        # Add initial document
        index.add_documents_atomic(["original content"], ["doc1"])
        assert index.num_docs == 1

        # Add document with same ID (should replace)
        index.add_documents_atomic(["updated content"], ["doc1"])

        # Should still have 1 document
        assert index.num_docs == 1, f"Expected 1 doc after duplicate, got {index.num_docs}"

        # Search should find the updated content
        results = index.search("updated", top_k=1)
        assert len(results) == 1, "Should find updated document"
        assert results[0][0] == "doc1"

        # Original content should not be found
        results = index.search("original", top_k=1)
        assert len(results) == 0, "Should not find original content"

        print(f"[PASS] Duplicate IDs test passed")
        print(f"   Document replaced correctly")

        index.close()
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    print("Running BM25 SQLite tests...\n")

    # Core functionality tests
    test_bm25_sqlite_basic()
    test_bm25_sqlite_persistence()
    test_bm25_sqlite_remove()
    test_bm25_sqlite_empty_query()
    test_bm25_sqlite_memory_usage()

    # Additional comprehensive tests
    test_bm25_sqlite_scoring()
    test_bm25_sqlite_atomic_transaction()
    test_bm25_sqlite_statistics_accuracy()
    test_bm25_sqlite_multiple_additions()
    test_bm25_sqlite_duplicate_ids()

    print("\n[SUCCESS] All tests passed!")
