"""Tests for dataset versioning module."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from mcpbr.dataset_versioning import (
    DatasetVersion,
    get_dataset_info,
    load_dataset_pinned,
    load_version_manifest,
    pin_dataset_version,
    save_version_manifest,
)


@pytest.fixture
def sample_version() -> DatasetVersion:
    """Create a sample DatasetVersion for testing."""
    return DatasetVersion(
        dataset_id="SWE-bench/SWE-bench_Lite",
        revision="abc123def456",
        download_date="2024-06-15T12:00:00+00:00",
        checksum="sha256_checksum_value",
    )


@pytest.fixture
def sample_versions() -> dict[str, DatasetVersion]:
    """Create a sample version manifest for testing."""
    return {
        "SWE-bench/SWE-bench_Lite": DatasetVersion(
            dataset_id="SWE-bench/SWE-bench_Lite",
            revision="abc123def456",
            download_date="2024-06-15T12:00:00+00:00",
            checksum="checksum_swebench",
        ),
        "openai/humaneval": DatasetVersion(
            dataset_id="openai/humaneval",
            revision="789ghi012jkl",
            download_date="2024-06-16T14:30:00+00:00",
            checksum="checksum_humaneval",
        ),
    }


class TestDatasetVersion:
    """Tests for the DatasetVersion dataclass."""

    def test_create_version(self, sample_version: DatasetVersion):
        """Test DatasetVersion creation with all fields."""
        assert sample_version.dataset_id == "SWE-bench/SWE-bench_Lite"
        assert sample_version.revision == "abc123def456"
        assert sample_version.download_date == "2024-06-15T12:00:00+00:00"
        assert sample_version.checksum == "sha256_checksum_value"

    def test_create_version_with_none_revision(self):
        """Test DatasetVersion with None revision."""
        version = DatasetVersion(
            dataset_id="test/dataset",
            revision=None,
            download_date="2024-01-01T00:00:00+00:00",
            checksum=None,
        )
        assert version.revision is None
        assert version.checksum is None

    def test_create_version_with_none_checksum(self):
        """Test DatasetVersion with None checksum."""
        version = DatasetVersion(
            dataset_id="test/dataset",
            revision="abc123",
            download_date="2024-01-01T00:00:00+00:00",
            checksum=None,
        )
        assert version.revision == "abc123"
        assert version.checksum is None


class TestPinDatasetVersion:
    """Tests for pin_dataset_version function."""

    @patch("mcpbr.dataset_versioning.dataset_info")
    def test_pin_version_latest(self, mock_dataset_info: MagicMock):
        """Test pinning the latest version of a dataset."""
        # Set up mock
        mock_info = MagicMock()
        mock_info.sha = "abc123def456789"
        mock_sibling = MagicMock()
        mock_sibling.rfilename = "data.parquet"
        mock_info.siblings = [mock_sibling]
        mock_dataset_info.return_value = mock_info

        version = pin_dataset_version("SWE-bench/SWE-bench_Lite")

        assert version.dataset_id == "SWE-bench/SWE-bench_Lite"
        assert version.revision == "abc123def456789"
        assert version.download_date is not None
        assert version.checksum is not None
        mock_dataset_info.assert_called_once_with("SWE-bench/SWE-bench_Lite", revision=None)

    @patch("mcpbr.dataset_versioning.dataset_info")
    def test_pin_version_specific_revision(self, mock_dataset_info: MagicMock):
        """Test pinning a specific revision of a dataset."""
        mock_info = MagicMock()
        mock_info.sha = "specific_revision_hash"
        mock_info.siblings = []
        mock_dataset_info.return_value = mock_info

        version = pin_dataset_version("SWE-bench/SWE-bench_Lite", revision="specific_revision_hash")

        assert version.revision == "specific_revision_hash"
        mock_dataset_info.assert_called_once_with(
            "SWE-bench/SWE-bench_Lite", revision="specific_revision_hash"
        )

    @patch("mcpbr.dataset_versioning.dataset_info")
    def test_pin_version_checksum_includes_files(self, mock_dataset_info: MagicMock):
        """Test that checksum incorporates file listing."""
        mock_info = MagicMock()
        mock_info.sha = "rev123"
        sibling_a = MagicMock()
        sibling_a.rfilename = "file_a.parquet"
        sibling_b = MagicMock()
        sibling_b.rfilename = "file_b.parquet"
        mock_info.siblings = [sibling_b, sibling_a]  # Unsorted to test sorting
        mock_dataset_info.return_value = mock_info

        version = pin_dataset_version("test/dataset")

        assert version.checksum is not None
        # Checksum should be deterministic
        version2 = pin_dataset_version("test/dataset")
        assert version.checksum == version2.checksum

    @patch("mcpbr.dataset_versioning.dataset_info")
    def test_pin_version_no_siblings(self, mock_dataset_info: MagicMock):
        """Test pinning when dataset has no siblings metadata."""
        mock_info = MagicMock()
        mock_info.sha = "rev123"
        mock_info.siblings = None
        mock_dataset_info.return_value = mock_info

        version = pin_dataset_version("test/dataset")

        assert version.checksum is not None
        assert version.revision == "rev123"


class TestLoadDatasetPinned:
    """Tests for load_dataset_pinned function."""

    @patch("mcpbr.dataset_versioning.load_dataset")
    def test_load_with_pinned_version(
        self, mock_load_dataset: MagicMock, sample_version: DatasetVersion
    ):
        """Test that pinned revision is passed to load_dataset."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        result = load_dataset_pinned(
            "SWE-bench/SWE-bench_Lite",
            version=sample_version,
            split="test",
        )

        mock_load_dataset.assert_called_once_with(
            "SWE-bench/SWE-bench_Lite",
            revision="abc123def456",
            split="test",
        )
        assert result is mock_dataset

    @patch("mcpbr.dataset_versioning.load_dataset")
    def test_load_without_pin_uses_latest(self, mock_load_dataset: MagicMock):
        """Test that loading without a pin uses the latest version."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        result = load_dataset_pinned("SWE-bench/SWE-bench_Lite", split="test")

        mock_load_dataset.assert_called_once_with(
            "SWE-bench/SWE-bench_Lite",
            revision=None,
            split="test",
        )
        assert result is mock_dataset

    @patch("mcpbr.dataset_versioning.load_dataset")
    def test_load_passes_kwargs(self, mock_load_dataset: MagicMock):
        """Test that additional kwargs are forwarded to load_dataset."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        version = DatasetVersion(
            dataset_id="test/dataset",
            revision="rev123",
            download_date="2024-01-01T00:00:00+00:00",
            checksum=None,
        )

        load_dataset_pinned(
            "test/dataset",
            version=version,
            split="train",
            name="subset_name",
            streaming=True,
        )

        mock_load_dataset.assert_called_once_with(
            "test/dataset",
            revision="rev123",
            split="train",
            name="subset_name",
            streaming=True,
        )

    @patch("mcpbr.dataset_versioning.load_dataset")
    def test_load_with_none_version(self, mock_load_dataset: MagicMock):
        """Test that explicitly passing version=None loads latest."""
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset

        load_dataset_pinned("test/dataset", version=None, split="test")

        mock_load_dataset.assert_called_once_with(
            "test/dataset",
            revision=None,
            split="test",
        )


class TestVersionManifest:
    """Tests for manifest save and load operations."""

    def test_save_and_load_roundtrip(
        self, tmp_path: Path, sample_versions: dict[str, DatasetVersion]
    ):
        """Test that saving and loading a manifest preserves all data."""
        manifest_path = tmp_path / "dataset_versions.json"

        save_version_manifest(sample_versions, manifest_path)

        assert manifest_path.exists()

        loaded = load_version_manifest(manifest_path)

        assert len(loaded) == len(sample_versions)
        for dataset_id, original in sample_versions.items():
            assert dataset_id in loaded
            loaded_version = loaded[dataset_id]
            assert loaded_version.dataset_id == original.dataset_id
            assert loaded_version.revision == original.revision
            assert loaded_version.download_date == original.download_date
            assert loaded_version.checksum == original.checksum

    def test_save_creates_parent_directories(self, tmp_path: Path):
        """Test that save creates parent directories if needed."""
        manifest_path = tmp_path / "nested" / "dir" / "manifest.json"

        versions = {
            "test/dataset": DatasetVersion(
                dataset_id="test/dataset",
                revision="rev123",
                download_date="2024-01-01T00:00:00+00:00",
                checksum=None,
            ),
        }

        save_version_manifest(versions, manifest_path)

        assert manifest_path.exists()

    def test_save_manifest_format(self, tmp_path: Path, sample_versions: dict[str, DatasetVersion]):
        """Test that the manifest file has the expected JSON structure."""
        manifest_path = tmp_path / "manifest.json"

        save_version_manifest(sample_versions, manifest_path)

        with open(manifest_path) as f:
            data = json.load(f)

        assert "format_version" in data
        assert data["format_version"] == "1.0"
        assert "created_at" in data
        assert "datasets" in data
        assert len(data["datasets"]) == 2

    def test_save_empty_manifest(self, tmp_path: Path):
        """Test saving an empty manifest."""
        manifest_path = tmp_path / "empty_manifest.json"

        save_version_manifest({}, manifest_path)

        loaded = load_version_manifest(manifest_path)
        assert len(loaded) == 0

    def test_load_nonexistent_manifest(self, tmp_path: Path):
        """Test that loading a nonexistent manifest raises FileNotFoundError."""
        manifest_path = tmp_path / "nonexistent.json"

        with pytest.raises(FileNotFoundError):
            load_version_manifest(manifest_path)

    def test_load_invalid_json_manifest(self, tmp_path: Path):
        """Test that loading invalid JSON raises JSONDecodeError."""
        manifest_path = tmp_path / "invalid.json"
        manifest_path.write_text("not valid json {{{")

        with pytest.raises(json.JSONDecodeError):
            load_version_manifest(manifest_path)

    def test_roundtrip_with_none_values(self, tmp_path: Path):
        """Test roundtrip with None revision and checksum."""
        manifest_path = tmp_path / "manifest.json"

        versions = {
            "test/dataset": DatasetVersion(
                dataset_id="test/dataset",
                revision=None,
                download_date="2024-01-01T00:00:00+00:00",
                checksum=None,
            ),
        }

        save_version_manifest(versions, manifest_path)
        loaded = load_version_manifest(manifest_path)

        loaded_version = loaded["test/dataset"]
        assert loaded_version.revision is None
        assert loaded_version.checksum is None

    def test_overwrite_existing_manifest(
        self, tmp_path: Path, sample_versions: dict[str, DatasetVersion]
    ):
        """Test that saving to an existing path overwrites the file."""
        manifest_path = tmp_path / "manifest.json"

        # Save original
        save_version_manifest(sample_versions, manifest_path)

        # Overwrite with different data
        new_versions = {
            "new/dataset": DatasetVersion(
                dataset_id="new/dataset",
                revision="new_rev",
                download_date="2025-01-01T00:00:00+00:00",
                checksum="new_checksum",
            ),
        }
        save_version_manifest(new_versions, manifest_path)

        loaded = load_version_manifest(manifest_path)
        assert len(loaded) == 1
        assert "new/dataset" in loaded
        assert "SWE-bench/SWE-bench_Lite" not in loaded


class TestGetDatasetInfo:
    """Tests for get_dataset_info function."""

    @patch("mcpbr.dataset_versioning.dataset_info")
    def test_get_dataset_info_basic(self, mock_dataset_info: MagicMock):
        """Test retrieving basic dataset metadata."""
        mock_info = MagicMock()
        mock_info.sha = "latest_rev_hash"
        mock_info.description = "A benchmark dataset."
        mock_info.tags = ["benchmark", "code"]
        mock_info.downloads = 50000
        mock_info.last_modified = MagicMock()
        mock_info.last_modified.isoformat.return_value = "2024-06-15T12:00:00+00:00"
        sibling = MagicMock()
        sibling.rfilename = "data.parquet"
        mock_info.siblings = [sibling]
        mock_dataset_info.return_value = mock_info

        result = get_dataset_info("SWE-bench/SWE-bench_Lite")

        assert result["dataset_id"] == "SWE-bench/SWE-bench_Lite"
        assert result["latest_revision"] == "latest_rev_hash"
        assert result["description"] == "A benchmark dataset."
        assert result["tags"] == ["benchmark", "code"]
        assert result["downloads"] == 50000
        assert result["last_modified"] == "2024-06-15T12:00:00+00:00"
        assert result["files"] == ["data.parquet"]

    @patch("mcpbr.dataset_versioning.dataset_info")
    def test_get_dataset_info_minimal(self, mock_dataset_info: MagicMock):
        """Test retrieving dataset info with minimal metadata."""
        mock_info = MagicMock()
        mock_info.sha = "rev123"
        mock_info.description = None
        mock_info.tags = None
        mock_info.downloads = None
        mock_info.last_modified = None
        mock_info.siblings = None
        mock_dataset_info.return_value = mock_info

        result = get_dataset_info("test/dataset")

        assert result["dataset_id"] == "test/dataset"
        assert result["latest_revision"] == "rev123"
        assert result["description"] == ""
        assert result["tags"] == []
        assert result["downloads"] == 0
        assert result["last_modified"] is None
        assert result["files"] == []

    @patch("mcpbr.dataset_versioning.dataset_info")
    def test_get_dataset_info_multiple_files(self, mock_dataset_info: MagicMock):
        """Test that all files are included in the result."""
        mock_info = MagicMock()
        mock_info.sha = "rev123"
        mock_info.description = "Test"
        mock_info.tags = []
        mock_info.downloads = 100
        mock_info.last_modified = None
        siblings = []
        for name in ["README.md", "data/train.parquet", "data/test.parquet"]:
            s = MagicMock()
            s.rfilename = name
            siblings.append(s)
        mock_info.siblings = siblings
        mock_dataset_info.return_value = mock_info

        result = get_dataset_info("test/dataset")

        assert len(result["files"]) == 3
        assert "README.md" in result["files"]
        assert "data/train.parquet" in result["files"]
        assert "data/test.parquet" in result["files"]
