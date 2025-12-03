"""
File I/O Integration Tests

Tests file operations including:
- SigMF data file reading/writing
- Configuration file loading
- Upload/download workflows
- Large file handling
- Binary IQ data integrity
"""

import pytest
import json
import tempfile
import os
import struct
from pathlib import Path
import numpy as np


class TestSigMFDataFiles:
    """Test SigMF data file operations."""

    def test_write_complex_float32(self, tmp_path):
        """Test writing cf32_le format (complex float32, little-endian)."""
        data_file = tmp_path / "test.sigmf-data"

        # Generate test samples
        samples = np.array([
            1.0 + 2.0j,
            -1.5 + 0.5j,
            0.0 + 0.0j,
            3.14159 - 2.71828j
        ], dtype=np.complex64)

        # Write to file
        samples.tofile(str(data_file))

        # Verify file size (8 bytes per sample)
        assert data_file.stat().st_size == len(samples) * 8

        # Read back and verify
        loaded = np.fromfile(str(data_file), dtype=np.complex64)
        np.testing.assert_array_almost_equal(samples, loaded, decimal=5)

    def test_read_partial_file(self, tmp_path):
        """Test reading a subset of samples from a large file."""
        data_file = tmp_path / "large.sigmf-data"

        # Write 10000 samples
        samples = np.random.randn(10000) + 1j * np.random.randn(10000)
        samples = samples.astype(np.complex64)
        samples.tofile(str(data_file))

        # Read only samples 100-200
        offset = 100 * 8  # 8 bytes per sample
        count = 100

        with open(data_file, 'rb') as f:
            f.seek(offset)
            data = f.read(count * 8)

        partial = np.frombuffer(data, dtype=np.complex64)

        np.testing.assert_array_almost_equal(samples[100:200], partial, decimal=5)

    def test_append_to_data_file(self, tmp_path):
        """Test appending samples to an existing data file."""
        data_file = tmp_path / "append.sigmf-data"

        # Write initial samples
        initial = np.array([1+1j, 2+2j, 3+3j], dtype=np.complex64)
        initial.tofile(str(data_file))

        # Append more samples
        additional = np.array([4+4j, 5+5j], dtype=np.complex64)
        with open(data_file, 'ab') as f:
            additional.tofile(f)

        # Read all and verify
        all_samples = np.fromfile(str(data_file), dtype=np.complex64)

        expected = np.concatenate([initial, additional])
        np.testing.assert_array_almost_equal(expected, all_samples, decimal=5)

    def test_data_integrity_large_file(self, tmp_path):
        """Test data integrity for larger files."""
        data_file = tmp_path / "integrity.sigmf-data"

        # Create 1M samples (8MB file)
        num_samples = 1_000_000
        samples = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)).astype(np.complex64)

        # Calculate checksum before write
        checksum_before = np.sum(samples)

        # Write
        samples.tofile(str(data_file))

        # Read back
        loaded = np.fromfile(str(data_file), dtype=np.complex64)

        # Verify
        checksum_after = np.sum(loaded)

        assert len(loaded) == num_samples
        np.testing.assert_almost_equal(checksum_before, checksum_after, decimal=3)


class TestSigMFMetadataFiles:
    """Test SigMF metadata file operations."""

    def test_write_valid_metadata(self, tmp_path):
        """Test writing valid SigMF metadata."""
        meta_file = tmp_path / "test.sigmf-meta"

        metadata = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 10000000,
                "core:version": "1.0.0",
                "core:description": "Test recording"
            },
            "captures": [{
                "core:sample_start": 0,
                "core:frequency": 915000000
            }],
            "annotations": []
        }

        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        # Read back and verify
        with open(meta_file, 'r') as f:
            loaded = json.load(f)

        assert loaded["global"]["core:datatype"] == "cf32_le"
        assert loaded["global"]["core:sample_rate"] == 10000000

    def test_read_metadata_with_annotations(self, tmp_path):
        """Test reading metadata with signal annotations."""
        meta_file = tmp_path / "annotated.sigmf-meta"

        metadata = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 10000000
            },
            "captures": [{
                "core:sample_start": 0,
                "core:frequency": 433000000
            }],
            "annotations": [
                {
                    "core:sample_start": 1000,
                    "core:sample_count": 5000,
                    "core:freq_lower_edge": 432500000,
                    "core:freq_upper_edge": 433500000,
                    "core:description": "Detected signal"
                },
                {
                    "core:sample_start": 10000,
                    "core:sample_count": 3000,
                    "rf_forensics:modulation_type": "FSK",
                    "rf_forensics:snr_db": 15.5
                }
            ]
        }

        with open(meta_file, 'w') as f:
            json.dump(metadata, f)

        # Read and parse
        with open(meta_file, 'r') as f:
            loaded = json.load(f)

        assert len(loaded["annotations"]) == 2
        assert loaded["annotations"][0]["core:sample_count"] == 5000
        assert loaded["annotations"][1]["rf_forensics:modulation_type"] == "FSK"

    def test_metadata_unicode_handling(self, tmp_path):
        """Test metadata with Unicode characters."""
        meta_file = tmp_path / "unicode.sigmf-meta"

        metadata = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 10000000,
                "core:description": "Тест запис 测试记录 🔬"
            },
            "captures": [{"core:sample_start": 0}],
            "annotations": []
        }

        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False)

        with open(meta_file, 'r', encoding='utf-8') as f:
            loaded = json.load(f)

        assert "Тест" in loaded["global"]["core:description"]
        assert "测试" in loaded["global"]["core:description"]


class TestConfigurationFiles:
    """Test configuration file operations."""

    def test_load_yaml_config(self, tmp_path):
        """Test loading YAML configuration."""
        config_file = tmp_path / "config.yaml"

        config_content = """
sdr:
  center_freq_hz: 915000000
  sample_rate_hz: 10000000
  gain_db: 40

pipeline:
  fft_size: 2048
  overlap_percent: 50

cfar:
  pfa: 0.000001
  num_reference_cells: 32
  num_guard_cells: 4
"""
        config_file.write_text(config_content)

        import yaml
        with open(config_file) as f:
            config = yaml.safe_load(f)

        assert config["sdr"]["center_freq_hz"] == 915000000
        assert config["pipeline"]["fft_size"] == 2048
        assert config["cfar"]["pfa"] == 0.000001

    def test_config_validation(self, tmp_path):
        """Test configuration validation via Pydantic."""
        try:
            from rf_forensics.config.schema import SDRConfig

            # Valid config
            config = SDRConfig(
                center_freq_hz=915e6,
                sample_rate_hz=10e6,
                gain_db=40
            )
            assert config.center_freq_hz == 915e6

            # Invalid config should raise (frequency too low)
            try:
                invalid = SDRConfig(
                    center_freq_hz=100,  # Below minimum (1e6)
                    sample_rate_hz=10e6,
                    gain_db=40
                )
                # If validation passes, test failed
                pytest.fail("Should have raised validation error")
            except Exception:
                pass  # Expected

        except ImportError:
            pytest.skip("SDRConfig not available")


class TestUploadDownload:
    """Test file upload/download workflows."""

    def test_upload_iq_file(self, tmp_path):
        """Test uploading an IQ file for analysis."""
        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir()

        # Simulate file upload
        filename = "uploaded_signal.sigmf-data"
        samples = np.random.randn(1000).astype(np.float32).view(np.complex64)

        upload_path = upload_dir / filename
        samples.tofile(str(upload_path))

        # Verify upload
        assert upload_path.exists()
        assert upload_path.stat().st_size == len(samples) * 8

    def test_download_creates_archive(self, tmp_path):
        """Test creating downloadable archive from recording."""
        import zipfile

        # Create source files
        recording_id = "download-test-001"
        data_file = tmp_path / f"{recording_id}.sigmf-data"
        meta_file = tmp_path / f"{recording_id}.sigmf-meta"

        # Write data
        samples = np.zeros(100, dtype=np.complex64)
        samples.tofile(str(data_file))

        # Write metadata
        meta = {"global": {"core:datatype": "cf32_le"}, "captures": [], "annotations": []}
        with open(meta_file, 'w') as f:
            json.dump(meta, f)

        # Create ZIP
        zip_path = tmp_path / f"{recording_id}.sigmf.zip"
        with zipfile.ZipFile(zip_path, 'w') as zf:
            zf.write(data_file, f"{recording_id}.sigmf-data")
            zf.write(meta_file, f"{recording_id}.sigmf-meta")

        # Verify ZIP
        assert zip_path.exists()
        with zipfile.ZipFile(zip_path, 'r') as zf:
            names = zf.namelist()
            assert f"{recording_id}.sigmf-data" in names
            assert f"{recording_id}.sigmf-meta" in names

    def test_extract_uploaded_archive(self, tmp_path):
        """Test extracting an uploaded SigMF archive."""
        import zipfile

        upload_dir = tmp_path / "uploads"
        extract_dir = tmp_path / "extracted"
        upload_dir.mkdir()
        extract_dir.mkdir()

        # Create archive to upload
        archive_path = upload_dir / "signal.sigmf.zip"
        with zipfile.ZipFile(archive_path, 'w') as zf:
            # Add mock files
            zf.writestr("signal.sigmf-meta", '{"global": {}}')
            zf.writestr("signal.sigmf-data", b'\x00' * 800)

        # Extract
        with zipfile.ZipFile(archive_path, 'r') as zf:
            zf.extractall(extract_dir)

        # Verify extraction
        assert (extract_dir / "signal.sigmf-meta").exists()
        assert (extract_dir / "signal.sigmf-data").exists()


class TestLargeFileHandling:
    """Test handling of large files."""

    @pytest.mark.slow
    def test_streaming_write(self, tmp_path):
        """Test streaming write for large recordings."""
        data_file = tmp_path / "stream.sigmf-data"

        total_samples = 0
        chunk_size = 100_000  # 100K samples per chunk

        with open(data_file, 'wb') as f:
            for _ in range(10):  # 1M total samples
                chunk = np.random.randn(chunk_size).astype(np.float32).view(np.complex64)
                chunk.tofile(f)
                total_samples += len(chunk)

        # Verify file size
        expected_size = total_samples * 8
        actual_size = data_file.stat().st_size
        assert actual_size == expected_size

    @pytest.mark.slow
    def test_streaming_read(self, tmp_path):
        """Test streaming read for large files."""
        data_file = tmp_path / "large_read.sigmf-data"

        # Create large file (complex64 = 8 bytes per sample)
        total_samples = 500_000
        samples = (np.random.randn(total_samples) + 1j * np.random.randn(total_samples)).astype(np.complex64)
        samples.tofile(str(data_file))

        # Stream read in chunks
        chunk_size = 50_000
        chunks_read = 0
        total_read = 0

        with open(data_file, 'rb') as f:
            while True:
                data = f.read(chunk_size * 8)
                if not data:
                    break
                chunk = np.frombuffer(data, dtype=np.complex64)
                chunks_read += 1
                total_read += len(chunk)

        assert total_read == total_samples
        assert chunks_read >= total_samples // chunk_size  # Allow for rounding

    def test_memory_mapped_access(self, tmp_path):
        """Test memory-mapped file access for random access."""
        data_file = tmp_path / "mmap.sigmf-data"

        # Create file
        num_samples = 100_000
        samples = np.arange(num_samples, dtype=np.float32) + 1j * np.arange(num_samples, dtype=np.float32)
        samples = samples.astype(np.complex64)
        samples.tofile(str(data_file))

        # Memory-map for random access
        mmap_samples = np.memmap(str(data_file), dtype=np.complex64, mode='r')

        # Random access
        assert mmap_samples[0] == samples[0]
        assert mmap_samples[50000] == samples[50000]
        assert mmap_samples[-1] == samples[-1]

        # Cleanup
        del mmap_samples


class TestBinaryDataIntegrity:
    """Test binary data integrity across operations."""

    def test_endianness_correct(self, tmp_path):
        """Test that little-endian format is used correctly."""
        data_file = tmp_path / "endian.sigmf-data"

        # Known value: 1.0 + 2.0j
        # IEEE 754 float32: 1.0 = 0x3F800000, 2.0 = 0x40000000
        # Little-endian: bytes reversed

        sample = np.array([1.0 + 2.0j], dtype=np.complex64)
        sample.tofile(str(data_file))

        with open(data_file, 'rb') as f:
            raw = f.read()

        # First 4 bytes should be 1.0 in little-endian
        i_value = struct.unpack('<f', raw[0:4])[0]
        q_value = struct.unpack('<f', raw[4:8])[0]

        assert abs(i_value - 1.0) < 1e-6
        assert abs(q_value - 2.0) < 1e-6

    def test_special_values(self, tmp_path):
        """Test handling of special float values."""
        data_file = tmp_path / "special.sigmf-data"

        # Create complex array with special values manually
        # np.complex64 stores I and Q as separate float32 values
        special = np.zeros(4, dtype=np.complex64)
        special[0] = 0.0 + 0.0j           # Zero
        special[1] = 1e-30 + 1e-30j       # Small (within float32 range)
        special[2] = 1e30 + 1e30j         # Large (within float32 range)
        # For infinity, set manually via view to avoid NaN from inf*1j
        special_view = special.view(np.float32)
        special_view[6] = np.inf   # Real part of index 3
        special_view[7] = np.inf   # Imag part of index 3

        special.tofile(str(data_file))

        loaded = np.fromfile(str(data_file), dtype=np.complex64)

        assert loaded[0] == 0.0 + 0.0j
        assert np.isinf(loaded[3].real)
        assert np.isinf(loaded[3].imag)

    def test_nan_handling(self, tmp_path):
        """Test handling of NaN values."""
        data_file = tmp_path / "nan.sigmf-data"

        samples = np.array([
            1.0 + 1.0j,
            np.nan + np.nan*1j,
            2.0 + 2.0j
        ], dtype=np.complex64)

        samples.tofile(str(data_file))

        loaded = np.fromfile(str(data_file), dtype=np.complex64)

        assert not np.isnan(loaded[0])
        assert np.isnan(loaded[1].real)
        assert np.isnan(loaded[1].imag)
        assert not np.isnan(loaded[2])


class TestFilePermissions:
    """Test file permission handling."""

    def test_create_in_nonexistent_directory(self, tmp_path):
        """Test creating files in a new directory."""
        new_dir = tmp_path / "new" / "nested" / "directory"
        new_dir.mkdir(parents=True)

        data_file = new_dir / "test.sigmf-data"
        samples = np.zeros(10, dtype=np.complex64)
        samples.tofile(str(data_file))

        assert data_file.exists()

    def test_read_only_file_detection(self, tmp_path):
        """Test handling of read-only files."""
        data_file = tmp_path / "readonly.sigmf-data"

        # Create file
        samples = np.zeros(10, dtype=np.complex64)
        samples.tofile(str(data_file))

        # Make read-only
        os.chmod(str(data_file), 0o444)

        # Should be able to read
        loaded = np.fromfile(str(data_file), dtype=np.complex64)
        assert len(loaded) == 10

        # Writing should fail
        try:
            with open(data_file, 'wb') as f:
                samples.tofile(f)
            pytest.fail("Should have raised permission error")
        except PermissionError:
            pass  # Expected

        # Cleanup: restore permissions
        os.chmod(str(data_file), 0o644)


class TestFileCleanup:
    """Test file cleanup operations."""

    def test_delete_recording_files(self, tmp_path):
        """Test deleting all files for a recording."""
        recording_id = "cleanup-test"

        # Create files
        data_file = tmp_path / f"{recording_id}.sigmf-data"
        meta_file = tmp_path / f"{recording_id}.sigmf-meta"

        data_file.write_bytes(b'\x00' * 100)
        meta_file.write_text('{}')

        # Delete
        data_file.unlink()
        meta_file.unlink()

        assert not data_file.exists()
        assert not meta_file.exists()

    def test_cleanup_on_error(self, tmp_path):
        """Test that partial files are cleaned up on error."""
        recording_id = "partial-cleanup"
        data_file = tmp_path / f"{recording_id}.sigmf-data"

        # Simulate partial write
        with open(data_file, 'wb') as f:
            f.write(b'\x00' * 100)
            # Simulate error before completion

        # Cleanup
        if data_file.exists():
            data_file.unlink()

        assert not data_file.exists()


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
