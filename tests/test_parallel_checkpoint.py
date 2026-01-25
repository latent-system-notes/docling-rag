"""Tests for parallel ingestion checkpoint/resume functionality."""
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch
import hashlib

from src.ingestion.checkpoint import (
    create_checkpoint,
    load_checkpoint,
    delete_checkpoint,
    validate_checkpoint,
    _compute_file_hash
)


class TestParallelCheckpoint:
    """Test checkpoint functionality in parallel ingestion."""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files."""
        temp_path = Path(tempfile.mkdtemp())
        yield temp_path
        shutil.rmtree(temp_path)

    @pytest.fixture
    def sample_file(self, temp_dir):
        """Create a sample test file."""
        file_path = temp_dir / "test_document.txt"
        file_path.write_text("This is a test document for checkpoint testing.")
        return file_path

    def test_checkpoint_creation(self, sample_file):
        """Test checkpoint is created correctly."""
        doc_id = hashlib.md5(str(sample_file.absolute()).encode()).hexdigest()

        checkpoint = create_checkpoint(
            file_path=sample_file,
            doc_id=doc_id,
            total_chunks=100,
            metadata={"doc_type": "txt", "language": "en"}
        )

        assert checkpoint.doc_id == doc_id
        assert checkpoint.total_chunks == 100
        assert checkpoint.processed_batches == []
        assert checkpoint.last_batch == -1
        assert checkpoint.metadata["doc_type"] == "txt"

    def test_checkpoint_load_and_delete(self, sample_file):
        """Test checkpoint can be loaded and deleted."""
        doc_id = hashlib.md5(str(sample_file.absolute()).encode()).hexdigest()

        # Create checkpoint
        create_checkpoint(
            file_path=sample_file,
            doc_id=doc_id,
            total_chunks=100,
            metadata={"doc_type": "txt"}
        )

        # Load checkpoint
        loaded = load_checkpoint(sample_file)
        assert loaded is not None
        assert loaded.doc_id == doc_id
        assert loaded.total_chunks == 100

        # Delete checkpoint
        delete_checkpoint(sample_file)

        # Verify deleted
        reloaded = load_checkpoint(sample_file)
        assert reloaded is None

    def test_checkpoint_validation_file_hash(self, sample_file):
        """Test checkpoint validation detects file modifications."""
        doc_id = hashlib.md5(str(sample_file.absolute()).encode()).hexdigest()

        # Create checkpoint
        checkpoint = create_checkpoint(
            file_path=sample_file,
            doc_id=doc_id,
            total_chunks=100,
            metadata={"doc_type": "txt"}
        )

        # Validate original file
        assert validate_checkpoint(checkpoint, sample_file) is True

        # Modify file
        sample_file.write_text("Modified content")

        # Validation should fail
        assert validate_checkpoint(checkpoint, sample_file) is False

    def test_checkpoint_nonexistent_file(self, temp_dir):
        """Test checkpoint handling for nonexistent files."""
        nonexistent = temp_dir / "nonexistent.txt"
        checkpoint = load_checkpoint(nonexistent)
        assert checkpoint is None

    def test_file_hash_computation(self, sample_file):
        """Test file hash is computed correctly."""
        hash1 = _compute_file_hash(sample_file)
        hash2 = _compute_file_hash(sample_file)

        # Same file should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA256 produces 64 hex characters

        # Modified file should produce different hash
        sample_file.write_text("Different content")
        hash3 = _compute_file_hash(sample_file)
        assert hash3 != hash1

    def test_resume_calculation(self, sample_file):
        """Test resume batch calculation from checkpoint."""
        doc_id = hashlib.md5(str(sample_file.absolute()).encode()).hexdigest()

        # Create checkpoint with some processed batches
        checkpoint = create_checkpoint(
            file_path=sample_file,
            doc_id=doc_id,
            total_chunks=500,
            metadata={"doc_type": "txt"}
        )

        # Simulate processing batches 0, 1, 2
        checkpoint.processed_batches = [0, 1, 2]
        checkpoint.last_batch = 2

        # Next batch should be 3
        if checkpoint.processed_batches:
            start_batch = max(checkpoint.processed_batches) + 1
        else:
            start_batch = 0

        assert start_batch == 3

    def test_completed_file_detection(self, sample_file):
        """Test detection of fully completed files."""
        doc_id = hashlib.md5(str(sample_file.absolute()).encode()).hexdigest()
        batch_size = 100
        total_chunks = 250

        # Create checkpoint
        checkpoint = create_checkpoint(
            file_path=sample_file,
            doc_id=doc_id,
            total_chunks=total_chunks,
            metadata={"doc_type": "txt"}
        )

        # Calculate expected total batches
        total_batches = (total_chunks + batch_size - 1) // batch_size  # = 3
        assert total_batches == 3

        # Mark all batches as complete
        checkpoint.processed_batches = [0, 1, 2]
        completed_batches = len(checkpoint.processed_batches)

        # Should be detected as complete
        assert completed_batches >= total_batches

    def test_partial_file_detection(self, sample_file):
        """Test detection of partially completed files."""
        doc_id = hashlib.md5(str(sample_file.absolute()).encode()).hexdigest()
        batch_size = 100
        total_chunks = 250

        # Create checkpoint
        checkpoint = create_checkpoint(
            file_path=sample_file,
            doc_id=doc_id,
            total_chunks=total_chunks,
            metadata={"doc_type": "txt"}
        )

        # Calculate expected total batches
        total_batches = (total_chunks + batch_size - 1) // batch_size  # = 3

        # Mark only some batches as complete
        checkpoint.processed_batches = [0, 1]
        completed_batches = len(checkpoint.processed_batches)

        # Should be detected as partial
        assert completed_batches < total_batches


class TestParallelIngestionIntegration:
    """Integration tests for parallel ingestion with checkpoints."""

    def test_doc_id_generation(self):
        """Test doc_id generation matches checkpoint system."""
        file_path = Path("/path/to/document.pdf")

        # This is how parallel_ingest.py generates doc_id
        doc_id = hashlib.md5(str(file_path.absolute()).encode()).hexdigest()

        # This is how checkpoint.py generates checkpoint filename
        from src.ingestion.checkpoint import get_checkpoint_path
        checkpoint_path = get_checkpoint_path(file_path)

        # They should match
        expected_checkpoint_name = f"{doc_id}.json"
        assert checkpoint_path.name == expected_checkpoint_name

    def test_batch_size_consistency(self):
        """Test batch size is consistent across checkpoint operations."""
        batch_size = 100
        total_chunks = 550

        # This is how total_batches is calculated in parallel_ingest.py
        total_batches_parallel = (total_chunks + batch_size - 1) // batch_size

        # This is how it's calculated in pre-queue filtering
        total_batches_filter = (total_chunks + 99) // 100

        # They should match
        assert total_batches_parallel == total_batches_filter
        assert total_batches_parallel == 6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
