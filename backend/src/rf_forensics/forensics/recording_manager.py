"""
GPU RF Forensics Engine - SigMF Recording Manager

Captures IQ samples to disk and generates SigMF-compliant metadata.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RecordingMetadata:
    """Metadata for a recording."""

    id: str
    name: str
    description: str
    center_freq_hz: float
    sample_rate_hz: float
    gain_db: float
    num_samples: int = 0
    file_size_bytes: int = 0
    duration_seconds: float = 0.0
    created_at: str = ""
    stopped_at: str = ""
    status: str = "recording"
    annotations: list[dict] = field(default_factory=list)


class RecordingManager:
    """
    SigMF-compliant recording manager.

    Captures raw IQ samples during live monitoring and generates
    SigMF format files (.sigmf-meta + .sigmf-data).

    SigMF Format:
    - Datatype: cf32_le (complex float32, little-endian)
    - 8 bytes per sample (4 I + 4 Q interleaved)
    """

    DATATYPE = "cf32_le"  # Complex float32, little-endian
    BYTES_PER_SAMPLE = 8  # 4 bytes I + 4 bytes Q

    def __init__(self, output_dir: str = None):
        """
        Initialize recording manager.

        Args:
            output_dir: Directory to store recordings. If None, uses RECORDINGS_DIR env var.
        """
        if output_dir is None:
            output_dir = os.getenv("RECORDINGS_DIR", "/data/recordings")
        self._output_dir = Path(output_dir)
        self._output_dir.mkdir(parents=True, exist_ok=True)

        # Thread lock for concurrent access to recording state
        self._lock = threading.Lock()

        self._active_recordings: dict[str, dict[str, Any]] = {}
        self._completed_recordings: dict[str, RecordingMetadata] = {}

        # Load existing recordings from disk
        self._load_existing_recordings()

    def _load_existing_recordings(self):
        """Load metadata for existing recordings on disk."""
        for meta_file in self._output_dir.glob("*.sigmf-meta"):
            recording_id = meta_file.stem
            data_file = self._output_dir / f"{recording_id}.sigmf-data"

            if data_file.exists():
                try:
                    with open(meta_file) as f:
                        meta = json.load(f)

                    # Extract info from SigMF metadata
                    global_meta = meta.get("global", {})
                    captures = meta.get("captures", [{}])

                    self._completed_recordings[recording_id] = RecordingMetadata(
                        id=recording_id,
                        name=global_meta.get("core:description", recording_id),
                        description=global_meta.get("core:description", ""),
                        center_freq_hz=captures[0].get("core:frequency", 0) if captures else 0,
                        sample_rate_hz=global_meta.get("core:sample_rate", 0),
                        gain_db=global_meta.get("rf_forensics:gain_db", 0),
                        num_samples=data_file.stat().st_size // self.BYTES_PER_SAMPLE,
                        file_size_bytes=data_file.stat().st_size,
                        duration_seconds=data_file.stat().st_size
                        / self.BYTES_PER_SAMPLE
                        / global_meta.get("core:sample_rate", 1),
                        created_at=captures[0].get("core:datetime", "") if captures else "",
                        status="stopped",
                        annotations=meta.get("annotations", []),
                    )
                except Exception as e:
                    logger.debug(f"Skipping corrupted recording file {meta_file}: {e}")

    def start_recording(
        self,
        name: str,
        description: str,
        sdr_config: dict[str, Any],
        duration_seconds: float | None = None,
    ) -> str:
        """
        Start a new recording.

        Args:
            name: Recording name.
            description: Recording description.
            sdr_config: SDR configuration dict with center_freq_hz, sample_rate_hz, gain_db.
            duration_seconds: Optional auto-stop duration.

        Returns:
            Recording ID.
        """
        recording_id = f"rec_{uuid.uuid4().hex[:8]}"
        filepath = self._output_dir / f"{recording_id}.sigmf-data"

        file_handle = None
        try:
            file_handle = open(filepath, "wb")

            recording_data = {
                "file": file_handle,
                "filepath": filepath,
                "name": name,
                "description": description,
                "start_time": time.time(),
                "start_datetime": datetime.utcnow().isoformat() + "Z",
                "num_samples": 0,
                "sdr_config": sdr_config,
                "duration_limit": duration_seconds,
                "annotations": [],
            }

            with self._lock:
                self._active_recordings[recording_id] = recording_data

        except Exception as e:
            # Ensure file is closed on any error
            if file_handle is not None:
                try:
                    file_handle.close()
                except Exception:
                    pass
            # Clean up the file if it was created
            if filepath.exists():
                try:
                    filepath.unlink()
                except Exception:
                    pass
            raise RuntimeError(f"Failed to start recording: {e}")

        return recording_id

    def write_samples(self, recording_id: str, iq_samples: np.ndarray) -> bool:
        """
        Write IQ samples to recording file.

        Args:
            recording_id: Recording ID.
            iq_samples: Complex64 numpy array of IQ samples.

        Returns:
            True if samples written, False if recording not found or stopped.
        """
        # Ensure complex64 format (cf32_le) - do this outside lock
        samples = np.asarray(iq_samples, dtype=np.complex64)
        samples_bytes = samples.tobytes()
        should_stop = False

        with self._lock:
            rec = self._active_recordings.get(recording_id)
            if rec is None:
                return False

            # Write interleaved I/Q samples
            rec["file"].write(samples_bytes)
            rec["num_samples"] += len(samples)

            # Check duration limit
            if rec["duration_limit"]:
                elapsed = time.time() - rec["start_time"]
                if elapsed >= rec["duration_limit"]:
                    should_stop = True

        # Stop outside lock to avoid deadlock
        if should_stop:
            self.stop_recording(recording_id)
            return False

        return True

    def write_samples_to_all(self, iq_samples: np.ndarray):
        """Write samples to all active recordings."""
        for recording_id in list(self._active_recordings.keys()):
            self.write_samples(recording_id, iq_samples)

    def add_annotation(
        self,
        recording_id: str,
        sample_start: int,
        sample_count: int,
        freq_lower_hz: float,
        freq_upper_hz: float,
        label: str,
        comment: str = "",
    ):
        """Add a detection annotation to a recording."""
        with self._lock:
            rec = self._active_recordings.get(recording_id)
            if rec is not None:
                rec["annotations"].append(
                    {
                        "core:sample_start": sample_start,
                        "core:sample_count": sample_count,
                        "core:freq_lower_edge": freq_lower_hz,
                        "core:freq_upper_edge": freq_upper_hz,
                        "core:label": label,
                        "core:comment": comment,
                    }
                )

    def stop_recording(self, recording_id: str) -> RecordingMetadata | None:
        """
        Stop a recording and generate SigMF metadata.

        Args:
            recording_id: Recording ID.

        Returns:
            Recording metadata if found, None otherwise.
        """
        with self._lock:
            rec = self._active_recordings.get(recording_id)
            if rec is None:
                return None

            # Remove from active recordings first to prevent double-stop
            del self._active_recordings[recording_id]

        # Close file outside lock (blocking I/O)
        try:
            rec["file"].close()
        except Exception as e:
            logger.warning(f"Error closing recording file: {e}")

        # Generate SigMF metadata
        self._generate_sigmf_meta(recording_id, rec)

        # Get file size
        try:
            file_size = rec["filepath"].stat().st_size
        except OSError:
            file_size = rec["num_samples"] * self.BYTES_PER_SAMPLE

        sample_rate = rec["sdr_config"].get("sample_rate_hz", 1)

        # Create completed metadata
        metadata = RecordingMetadata(
            id=recording_id,
            name=rec["name"],
            description=rec["description"],
            center_freq_hz=rec["sdr_config"].get("center_freq_hz", 0),
            sample_rate_hz=sample_rate,
            gain_db=rec["sdr_config"].get("gain_db", 0),
            num_samples=rec["num_samples"],
            file_size_bytes=file_size,
            duration_seconds=rec["num_samples"] / sample_rate if sample_rate > 0 else 0,
            created_at=rec["start_datetime"],
            stopped_at=datetime.utcnow().isoformat() + "Z",
            status="stopped",
            annotations=rec["annotations"],
        )

        with self._lock:
            self._completed_recordings[recording_id] = metadata

        return metadata

    def _generate_sigmf_meta(self, recording_id: str, rec: dict[str, Any]):
        """Generate SigMF .sigmf-meta JSON file."""
        sdr = rec["sdr_config"]

        metadata = {
            "global": {
                "core:datatype": self.DATATYPE,
                "core:sample_rate": sdr.get("sample_rate_hz", 10e6),
                "core:version": "1.0.0",
                "core:description": rec["description"],
                "core:author": "RF Forensics GPU Pipeline",
                "core:recorder": "RAPIDS cuSignal",
                "core:hw": "RTX 4090",
                "rf_forensics:name": rec["name"],
                "rf_forensics:gain_db": sdr.get("gain_db", 0),
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": sdr.get("center_freq_hz", 915e6),
                    "core:datetime": rec["start_datetime"],
                }
            ],
            "annotations": rec["annotations"],
        }

        meta_path = self._output_dir / f"{recording_id}.sigmf-meta"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def list_recordings(self) -> list[dict[str, Any]]:
        """List all recordings (active and completed)."""
        recordings = []

        # Active recordings
        for rec_id, rec in self._active_recordings.items():
            sample_rate = rec["sdr_config"].get("sample_rate_hz", 1)
            recordings.append(
                {
                    "id": rec_id,
                    "name": rec["name"],
                    "description": rec["description"],
                    "center_freq_hz": rec["sdr_config"].get("center_freq_hz", 0),
                    "sample_rate_hz": sample_rate,
                    "num_samples": rec["num_samples"],
                    "duration_seconds": rec["num_samples"] / sample_rate if sample_rate > 0 else 0,
                    "file_size_bytes": rec["num_samples"] * self.BYTES_PER_SAMPLE,
                    "created_at": rec["start_datetime"],
                    "status": "recording",
                    "sigmf_meta_path": str(self._output_dir / f"{rec_id}.sigmf-meta"),
                    "sigmf_data_path": str(self._output_dir / f"{rec_id}.sigmf-data"),
                }
            )

        # Completed recordings
        for rec_id, meta in self._completed_recordings.items():
            recordings.append(
                {
                    "id": meta.id,
                    "name": meta.name,
                    "description": meta.description,
                    "center_freq_hz": meta.center_freq_hz,
                    "sample_rate_hz": meta.sample_rate_hz,
                    "num_samples": meta.num_samples,
                    "duration_seconds": meta.duration_seconds,
                    "file_size_bytes": meta.file_size_bytes,
                    "created_at": meta.created_at,
                    "status": meta.status,
                    "sigmf_meta_path": str(self._output_dir / f"{rec_id}.sigmf-meta"),
                    "sigmf_data_path": str(self._output_dir / f"{rec_id}.sigmf-data"),
                }
            )

        return recordings

    def get_recording(self, recording_id: str) -> dict[str, Any] | None:
        """Get metadata for a specific recording."""
        for rec in self.list_recordings():
            if rec["id"] == recording_id:
                return rec
        return None

    def delete_recording(self, recording_id: str) -> bool:
        """
        Delete a recording and its files.

        Args:
            recording_id: Recording ID.

        Returns:
            True if deleted, False if not found.
        """
        # Can't delete active recording
        if recording_id in self._active_recordings:
            return False

        if recording_id not in self._completed_recordings:
            return False

        # Delete files
        meta_path = self._output_dir / f"{recording_id}.sigmf-meta"
        data_path = self._output_dir / f"{recording_id}.sigmf-data"

        if meta_path.exists():
            meta_path.unlink()
        if data_path.exists():
            data_path.unlink()

        del self._completed_recordings[recording_id]
        return True

    def create_download_zip(self, recording_id: str) -> Path | None:
        """
        Create a ZIP file for download containing .sigmf-meta and .sigmf-data.

        Args:
            recording_id: Recording ID.

        Returns:
            Path to ZIP file, or None if recording not found.
        """
        meta_path = self._output_dir / f"{recording_id}.sigmf-meta"
        data_path = self._output_dir / f"{recording_id}.sigmf-data"

        if not meta_path.exists() or not data_path.exists():
            return None

        zip_path = Path(f"/tmp/{recording_id}.sigmf.zip")

        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(meta_path, f"{recording_id}.sigmf-meta")
            zf.write(data_path, f"{recording_id}.sigmf-data")

        return zip_path

    def has_active_recordings(self) -> bool:
        """Check if any recordings are active."""
        return len(self._active_recordings) > 0

    @property
    def active_recording_ids(self) -> list[str]:
        """Get list of active recording IDs."""
        return list(self._active_recordings.keys())

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def __del__(self):
        """Cleanup any active recordings on destruction."""
        for rec_id in list(self._active_recordings.keys()):
            try:
                rec = self._active_recordings[rec_id]
                if "file" in rec and rec["file"]:
                    rec["file"].close()
            except Exception:
                pass
