"""
GPU RF Forensics Engine - Spectral Peak Detection

Spectral peak detection, merging, and tracking for signal characterization.
"""

from __future__ import annotations

import time
from bisect import bisect_left
from dataclasses import dataclass, field

from rf_forensics.core.gpu_compat import CUPY_AVAILABLE, cp, np

# Track timeout as fraction of max_age_frames (10%)
TRACK_TIMEOUT_FRACTION = 0.1


@dataclass
class Detection:
    """
    A detected signal in the spectrum.

    Fields align with BACKEND_CONTRACT.md Section 2 - Detection Stream.
    """

    detection_id: int
    center_freq_hz: float
    bandwidth_hz: float
    bandwidth_3db_hz: float
    bandwidth_6db_hz: float
    peak_power_db: float
    snr_db: float
    start_bin: int
    end_bin: int
    peak_bin: int
    timestamp: float = field(default_factory=time.time)

    # AMC classification fields
    modulation_type: str = "Unknown"
    modulation_confidence: float = 0.0
    top_k_predictions: list[tuple[str, float]] = field(default_factory=list)

    # Tracking fields (per BACKEND_CONTRACT.md)
    track_id: str | None = None  # String format: "trk-abc123"
    duty_cycle: float | None = None  # 0.0-1.0
    frames_seen: int | None = None

    # Clustering & Anomaly fields (HIGH priority per BACKEND_CONTRACT.md)
    cluster_id: int | None = None  # Links detection to DBSCAN cluster
    anomaly_score: float | None = None  # Autoencoder output: 0.0-1.0, >0.5 = anomalous
    symbol_rate: float | None = None  # Estimated symbol rate in baud


@dataclass
class TrackedDetection(Detection):
    """
    A detection tracked across multiple frames.

    Inherits tracking fields from Detection but adds temporal tracking state.
    Note: track_id, duty_cycle, frames_seen are inherited from Detection
    but initialized with tracking-specific defaults here.
    """

    # Tracking state (override defaults from Detection)
    track_id: str | None = field(default=None)  # Will be set as "trk-{id}"
    frames_seen: int | None = 1
    duty_cycle: float | None = 1.0

    # Additional tracking fields
    track_id_numeric: int = 0  # Internal numeric ID for association
    first_seen: float = 0.0
    last_seen: float = 0.0
    frequency_drift_hz_per_s: float = 0.0


class PeakDetector:
    """
    Spectral peak detector with signal characterization.

    Features:
    - Find local maxima above CFAR threshold
    - Merge adjacent detections into single signals
    - Estimate signal bandwidth (3dB and 6dB)
    - Track detections across time frames
    """

    def __init__(
        self,
        merge_bandwidth_hz: float = 10000,
        min_snr_db: float = 3.0,
        min_bandwidth_bins: int = 1,
    ):
        """
        Initialize peak detector.

        Args:
            merge_bandwidth_hz: Maximum frequency gap to merge detections.
            min_snr_db: Minimum SNR for valid detection.
            min_bandwidth_bins: Minimum detection width in bins.
        """
        self._merge_bandwidth_hz = merge_bandwidth_hz
        self._min_snr_db = min_snr_db
        self._min_bandwidth_bins = min_bandwidth_bins
        self._detection_counter = 0

        # Tracking state
        self._active_tracks: list[TrackedDetection] = []
        self._track_counter = 0
        self._association_threshold_hz = merge_bandwidth_hz * 2

    def find_peaks(
        self,
        psd_db: cp.ndarray,
        cfar_mask: cp.ndarray,
        freq_axis: cp.ndarray,
        noise_floor_db: float | None = None,
    ) -> list[Detection]:
        """
        Find spectral peaks above CFAR threshold.

        Args:
            psd_db: PSD in dB (CuPy or NumPy array).
            cfar_mask: Boolean detection mask from CFAR.
            freq_axis: Frequency axis in Hz.
            noise_floor_db: Optional noise floor estimate.

        Returns:
            List of Detection objects.
        """
        # Use appropriate array library (CuPy if available and input is on GPU)
        is_cupy_array = CUPY_AVAILABLE and hasattr(psd_db, "__cuda_array_interface__")
        xp = cp if is_cupy_array else np

        # Estimate noise floor on GPU if not provided
        if noise_floor_db is None:
            non_detect = psd_db[~cfar_mask]
            if len(non_detect) > 0:
                noise_floor_db = float(xp.median(non_detect))
            elif len(psd_db) > 0:
                noise_floor_db = float(xp.min(psd_db))
            else:
                noise_floor_db = -120.0  # Default for empty input

        # Find contiguous detection regions on GPU
        regions = self._find_contiguous_regions_gpu(cfar_mask, xp)

        # Convert to NumPy for CPU-bound characterization loop (single transfer)
        if xp is cp:
            psd_db_np = cp.asnumpy(psd_db)
            freq_axis_np = cp.asnumpy(freq_axis)
        else:
            psd_db_np = psd_db
            freq_axis_np = freq_axis

        # Merge nearby regions
        merged_regions = self._merge_regions(regions, freq_axis_np)

        if not merged_regions:
            return []

        # Vectorized characterization of all detections at once
        detections = self._characterize_detections_batch(
            psd_db_np, freq_axis_np, merged_regions, noise_floor_db
        )

        return detections

    def _find_contiguous_regions_gpu(self, mask, xp) -> list[tuple[int, int]]:
        """Find contiguous True regions using GPU-compatible operations."""
        if not mask.any():
            return []

        # Pad with False to detect transitions at boundaries
        padded = xp.concatenate([[False], mask, [False]])

        # Find transitions: 1 = start (False->True), -1 = end (True->False)
        diff = xp.diff(padded.astype(xp.int8))

        starts = xp.where(diff == 1)[0]
        ends = xp.where(diff == -1)[0]

        # Transfer small index arrays to CPU (minimal overhead)
        if xp is cp:
            starts = cp.asnumpy(starts)
            ends = cp.asnumpy(ends)

        return list(zip(starts.tolist(), ends.tolist()))

    def _find_contiguous_regions(self, mask: np.ndarray) -> list[tuple[int, int]]:
        """Find contiguous True regions in mask using vectorized operations."""
        if not mask.any():
            return []

        # Pad with False to detect transitions at boundaries
        padded = np.concatenate([[False], mask, [False]])

        # Find transitions: 1 = start (False->True), -1 = end (True->False)
        diff = np.diff(padded.astype(np.int8))

        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0]

        return list(zip(starts.tolist(), ends.tolist()))

    def _merge_regions(
        self, regions: list[tuple[int, int]], freq_axis: np.ndarray
    ) -> list[tuple[int, int]]:
        """Merge nearby regions based on frequency gap."""
        if not regions:
            return []

        freq_resolution = abs(freq_axis[1] - freq_axis[0]) if len(freq_axis) > 1 else 1.0
        merge_bins = int(self._merge_bandwidth_hz / freq_resolution)

        merged = [regions[0]]
        for start, end in regions[1:]:
            prev_start, prev_end = merged[-1]

            # Check if gap is small enough to merge
            if start - prev_end <= merge_bins:
                merged[-1] = (prev_start, end)
            else:
                merged.append((start, end))

        return merged

    def _characterize_detections_batch(
        self,
        psd_db: np.ndarray,
        freq_axis: np.ndarray,
        regions: list[tuple[int, int]],
        noise_floor_db: float,
    ) -> list[Detection]:
        """
        Vectorized characterization of all detections at once.

        Instead of looping through each detection, this method:
        1. Filters valid regions by minimum bandwidth
        2. Extracts all peak bins in one vectorized pass
        3. Computes all SNRs, centroids, and bandwidths using array operations
        4. Creates Detection objects only for valid detections

        This eliminates O(n) per-detection overhead for n detections.
        """
        if not regions:
            return []

        # Convert regions to arrays for vectorized operations
        starts = np.array([r[0] for r in regions], dtype=np.int32)
        ends = np.array([r[1] for r in regions], dtype=np.int32)
        widths = ends - starts

        # Filter by minimum bandwidth
        valid_mask = widths >= self._min_bandwidth_bins
        if not valid_mask.any():
            return []

        starts = starts[valid_mask]
        ends = ends[valid_mask]
        widths = widths[valid_mask]
        n_regions = len(starts)

        # Pre-compute frequency resolution
        freq_resolution = abs(freq_axis[1] - freq_axis[0]) if len(freq_axis) > 1 else 1.0

        # Vectorized peak finding within each region using masked maximum
        # Create a mask array where only valid regions are considered
        max_width = int(np.max(widths))
        n_bins = len(psd_db)

        # Build padded region matrix for vectorized processing
        # Each row is a region, padded with -inf for unused positions
        if max_width <= 256 and n_regions <= 1000:
            # For small regions/counts, use fully vectorized approach
            region_matrix = np.full((n_regions, max_width), -np.inf, dtype=np.float64)
            for i in range(n_regions):
                w = widths[i]
                region_matrix[i, :w] = psd_db[starts[i] : ends[i]]

            # Vectorized argmax across all regions
            local_peaks = np.argmax(region_matrix, axis=1)
            peak_bins = starts + local_peaks
            peak_powers = region_matrix[np.arange(n_regions), local_peaks]
        else:
            # For large regions, use optimized loop with pre-allocated arrays
            peak_bins = np.empty(n_regions, dtype=np.int32)
            peak_powers = np.empty(n_regions, dtype=np.float64)
            for i in range(n_regions):
                region_psd = psd_db[starts[i] : ends[i]]
                local_peak = np.argmax(region_psd)
                peak_bins[i] = starts[i] + local_peak
                peak_powers[i] = region_psd[local_peak]

        # Vectorized SNR calculation
        snr_dbs = peak_powers - noise_floor_db

        # Filter by minimum SNR early (before expensive bandwidth calculation)
        snr_valid = snr_dbs >= self._min_snr_db
        if not snr_valid.any():
            return []

        # Apply SNR filter
        valid_indices = np.where(snr_valid)[0]
        starts = starts[snr_valid]
        ends = ends[snr_valid]
        peak_bins = peak_bins[snr_valid]
        peak_powers = peak_powers[snr_valid]
        snr_dbs = snr_dbs[snr_valid]
        n_valid = len(starts)

        # Vectorized center frequency (power-weighted centroid)
        # For small numbers of detections, use simple fallback to peak frequency
        # For larger counts, use padded matrix approach
        widths_valid = ends - starts
        max_width_valid = int(np.max(widths_valid)) if n_valid > 0 else 0

        if n_valid <= 100 and max_width_valid <= 256:
            # Build padded matrices for all regions
            power_matrix = np.zeros((n_valid, max_width_valid), dtype=np.float64)
            freq_matrix = np.zeros((n_valid, max_width_valid), dtype=np.float64)

            for i in range(n_valid):
                w = widths_valid[i]
                region_psd = psd_db[starts[i] : ends[i]]
                power_matrix[i, :w] = np.power(10.0, region_psd / 10.0)
                freq_matrix[i, :w] = freq_axis[starts[i] : ends[i]]

            # Vectorized centroid calculation
            total_powers = np.sum(power_matrix, axis=1)
            weighted_freqs = np.sum(freq_matrix * power_matrix, axis=1)

            # Avoid division by zero
            valid_power_mask = total_powers > 0
            center_freqs = np.where(
                valid_power_mask,
                weighted_freqs / np.maximum(total_powers, 1e-12),
                freq_axis[peak_bins],
            )
        else:
            # Fallback for large/many regions
            center_freqs = np.empty(n_valid, dtype=np.float64)
            for i in range(n_valid):
                region_psd = psd_db[starts[i] : ends[i]]
                region_freq = freq_axis[starts[i] : ends[i]]
                linear_power = np.power(10.0, region_psd / 10.0)
                total_power = np.sum(linear_power)
                if total_power > 0:
                    center_freqs[i] = np.sum(region_freq * linear_power) / total_power
                else:
                    center_freqs[i] = freq_axis[peak_bins[i]]

        # Vectorized bandwidth calculation using threshold crossing
        bandwidths = np.abs(freq_axis[ends - 1] - freq_axis[starts])

        # Vectorized 3dB and 6dB bandwidth using binary search
        thresholds_3db = peak_powers - 3.0
        thresholds_6db = peak_powers - 6.0

        bw_3db = self._calculate_bandwidths_batch(psd_db, freq_axis, peak_bins, thresholds_3db)
        bw_6db = self._calculate_bandwidths_batch(psd_db, freq_axis, peak_bins, thresholds_6db)

        # Create Detection objects
        detections = []
        current_time = time.time()

        for i in range(n_valid):
            self._detection_counter += 1
            detections.append(
                Detection(
                    detection_id=self._detection_counter,
                    center_freq_hz=float(center_freqs[i]),
                    bandwidth_hz=float(bandwidths[i]),
                    bandwidth_3db_hz=float(bw_3db[i]),
                    bandwidth_6db_hz=float(bw_6db[i]),
                    peak_power_db=float(peak_powers[i]),
                    snr_db=float(snr_dbs[i]),
                    start_bin=int(starts[i]),
                    end_bin=int(ends[i]),
                    peak_bin=int(peak_bins[i]),
                    timestamp=current_time,
                )
            )

        return detections

    def _calculate_bandwidths_batch(
        self,
        psd_db: np.ndarray,
        freq_axis: np.ndarray,
        peak_bins: np.ndarray,
        thresholds: np.ndarray,
    ) -> np.ndarray:
        """
        Vectorized bandwidth calculation for multiple peaks.

        Uses vectorized threshold crossing instead of while-loops.
        For each peak, finds the left and right edges where PSD drops below threshold.
        """
        n_peaks = len(peak_bins)
        n_bins = len(psd_db)
        bandwidths = np.empty(n_peaks, dtype=np.float64)

        for i in range(n_peaks):
            peak = peak_bins[i]
            threshold = thresholds[i]

            # Vectorized left edge search: find first bin below threshold going left
            # Create boolean array and find first True (below threshold)
            left_region = psd_db[: peak + 1][::-1]  # Reverse to search from peak leftward
            below_threshold_left = left_region <= threshold

            if below_threshold_left.any():
                # First True in reversed array = distance from peak to edge
                left_offset = np.argmax(below_threshold_left)
                left_bin = peak - left_offset
            else:
                left_bin = 0

            # Vectorized right edge search
            right_region = psd_db[peak:]
            below_threshold_right = right_region <= threshold

            if below_threshold_right.any():
                right_offset = np.argmax(below_threshold_right)
                right_bin = peak + right_offset
            else:
                right_bin = n_bins - 1

            bandwidths[i] = abs(freq_axis[right_bin] - freq_axis[left_bin])

        return bandwidths

    def _characterize_detection(
        self,
        psd_db: np.ndarray,
        freq_axis: np.ndarray,
        start_bin: int,
        end_bin: int,
        noise_floor_db: float,
    ) -> Detection | None:
        """Characterize a single detection."""
        self._detection_counter += 1

        # Extract region
        region_psd = psd_db[start_bin:end_bin]
        region_freq = freq_axis[start_bin:end_bin]

        if len(region_psd) == 0:
            return None

        # Find peak
        peak_idx_local = int(np.argmax(region_psd))
        peak_bin = start_bin + peak_idx_local
        peak_power_db = float(region_psd[peak_idx_local])

        # Calculate SNR
        snr_db = peak_power_db - noise_floor_db

        # Calculate center frequency (centroid)
        linear_power = 10 ** (region_psd / 10)
        total_power = np.sum(linear_power)
        if total_power > 0:
            center_freq = float(np.sum(region_freq * linear_power) / total_power)
        else:
            center_freq = float(region_freq[peak_idx_local])

        # Calculate bandwidth
        bandwidth_hz = float(abs(freq_axis[end_bin - 1] - freq_axis[start_bin]))

        # 3dB and 6dB bandwidth
        threshold_3db = peak_power_db - 3
        threshold_6db = peak_power_db - 6

        bw_3db = self._calculate_bandwidth(psd_db, freq_axis, peak_bin, threshold_3db)
        bw_6db = self._calculate_bandwidth(psd_db, freq_axis, peak_bin, threshold_6db)

        return Detection(
            detection_id=self._detection_counter,
            center_freq_hz=center_freq,
            bandwidth_hz=bandwidth_hz,
            bandwidth_3db_hz=bw_3db,
            bandwidth_6db_hz=bw_6db,
            peak_power_db=peak_power_db,
            snr_db=snr_db,
            start_bin=start_bin,
            end_bin=end_bin,
            peak_bin=peak_bin,
            timestamp=time.time(),
        )

    def _calculate_bandwidth(
        self, psd_db: np.ndarray, freq_axis: np.ndarray, peak_bin: int, threshold_db: float
    ) -> float:
        """Calculate bandwidth at given threshold."""
        n_bins = len(psd_db)

        # Find left edge
        left_bin = peak_bin
        while left_bin > 0 and psd_db[left_bin] > threshold_db:
            left_bin -= 1

        # Find right edge
        right_bin = peak_bin
        while right_bin < n_bins - 1 and psd_db[right_bin] > threshold_db:
            right_bin += 1

        return float(abs(freq_axis[right_bin] - freq_axis[left_bin]))

    def track_detections(
        self, current: list[Detection], max_age_frames: int = 10
    ) -> list[TrackedDetection]:
        """
        Track detections across frames using O(n log m) binary search.

        Args:
            current: Current frame detections.
            max_age_frames: Maximum frames without detection before track is dropped.

        Returns:
            List of tracked detections.
        """
        current_time = time.time()
        associated = set()

        # Sort tracks by frequency for binary search - O(m log m)
        if self._active_tracks:
            sorted_tracks = sorted(self._active_tracks, key=lambda t: t.center_freq_hz)
            track_freqs = [t.center_freq_hz for t in sorted_tracks]
        else:
            sorted_tracks = []
            track_freqs = []

        # Associate current detections with existing tracks - O(n log m)
        for detection in current:
            best_track = None
            best_distance = float("inf")

            if track_freqs:
                # Binary search for closest track
                idx = bisect_left(track_freqs, detection.center_freq_hz)

                # Check nearby tracks (up to 2 on each side)
                for i in range(max(0, idx - 1), min(len(sorted_tracks), idx + 2)):
                    dist = abs(detection.center_freq_hz - track_freqs[i])
                    if dist < self._association_threshold_hz and dist < best_distance:
                        best_track = sorted_tracks[i]
                        best_distance = dist

            if best_track is not None:
                # Update existing track
                associated.add(best_track.track_id_numeric)
                best_track.center_freq_hz = detection.center_freq_hz
                best_track.bandwidth_hz = detection.bandwidth_hz
                best_track.peak_power_db = detection.peak_power_db
                best_track.snr_db = detection.snr_db
                best_track.frames_seen += 1
                best_track.last_seen = current_time

                # Copy AMC and anomaly fields from detection
                best_track.modulation_type = detection.modulation_type
                best_track.modulation_confidence = detection.modulation_confidence
                best_track.top_k_predictions = detection.top_k_predictions
                best_track.cluster_id = detection.cluster_id
                best_track.anomaly_score = detection.anomaly_score
                best_track.symbol_rate = detection.symbol_rate

                # Calculate duty cycle as ratio of observed frames to elapsed time
                total_time = best_track.last_seen - best_track.first_seen
                if total_time > 0 and best_track.frames_seen > 1:
                    # Estimate expected frame count based on average interval
                    avg_interval = total_time / (best_track.frames_seen - 1)
                    expected_frames = (
                        total_time / avg_interval if avg_interval > 0 else best_track.frames_seen
                    )
                    best_track.duty_cycle = min(1.0, best_track.frames_seen / expected_frames)
                else:
                    best_track.duty_cycle = 1.0
            else:
                # Create new track
                self._track_counter += 1
                track_id_str = f"trk-{self._track_counter:06d}"  # String format per contract
                new_track = TrackedDetection(
                    detection_id=detection.detection_id,
                    center_freq_hz=detection.center_freq_hz,
                    bandwidth_hz=detection.bandwidth_hz,
                    bandwidth_3db_hz=detection.bandwidth_3db_hz,
                    bandwidth_6db_hz=detection.bandwidth_6db_hz,
                    peak_power_db=detection.peak_power_db,
                    snr_db=detection.snr_db,
                    start_bin=detection.start_bin,
                    end_bin=detection.end_bin,
                    peak_bin=detection.peak_bin,
                    timestamp=detection.timestamp,
                    modulation_type=detection.modulation_type,
                    modulation_confidence=detection.modulation_confidence,
                    top_k_predictions=detection.top_k_predictions,
                    cluster_id=detection.cluster_id,
                    anomaly_score=detection.anomaly_score,
                    symbol_rate=detection.symbol_rate,
                    track_id=track_id_str,
                    track_id_numeric=self._track_counter,
                    frames_seen=1,
                    first_seen=current_time,
                    last_seen=current_time,
                )
                self._active_tracks.append(new_track)

        # Remove stale tracks
        self._active_tracks = [
            t
            for t in self._active_tracks
            if t.track_id_numeric in associated
            or (current_time - t.last_seen) < max_age_frames * TRACK_TIMEOUT_FRACTION
        ]

        return list(self._active_tracks)

    def reset_tracking(self) -> None:
        """Reset all tracking state."""
        self._active_tracks.clear()
        self._track_counter = 0


class SignalClassifier:
    """
    Signal classifier based on spectral characteristics using template matching.
    """

    def __init__(self):
        # Known signal templates with bandwidth ratio ranges and minimum SNR
        self._templates = {
            "narrowband_cw": {"bw_ratio": (0, 0.01), "snr_min": 10},
            "narrowband": {"bw_ratio": (0, 0.01), "snr_min": 0},
            "medium_band": {"bw_ratio": (0.01, 0.1), "snr_min": 0},
            "wideband": {"bw_ratio": (0.1, 1.0), "snr_min": 3},
        }

    def classify(self, detection: Detection, sample_rate: float) -> str:
        """
        Classify a detection based on characteristics using template matching.

        Args:
            detection: Detection to classify.
            sample_rate: Sample rate for bandwidth ratio calculation.

        Returns:
            Classification string.
        """
        bw_ratio = detection.bandwidth_hz / sample_rate if sample_rate > 0 else 0

        # Match against templates
        for name, template in self._templates.items():
            if "bw_ratio" in template:
                min_bw, max_bw = template["bw_ratio"]
                if min_bw <= bw_ratio <= max_bw:
                    # Check SNR requirement if specified
                    if "snr_min" not in template or detection.snr_db >= template["snr_min"]:
                        return name

        return "unknown"


def plot_detections(
    psd_db: np.ndarray,
    freq_axis: np.ndarray,
    detections: list[Detection],
    threshold: np.ndarray | None = None,
):
    """
    Generate visualization data for detections.

    Args:
        psd_db: PSD in dB.
        freq_axis: Frequency axis in Hz.
        detections: List of detections.
        threshold: Optional CFAR threshold.

    Returns:
        Dictionary with plot data.
    """
    plot_data = {
        "frequencies_hz": freq_axis.tolist() if hasattr(freq_axis, "tolist") else list(freq_axis),
        "psd_db": psd_db.tolist() if hasattr(psd_db, "tolist") else list(psd_db),
        "detections": [],
    }

    if threshold is not None:
        plot_data["threshold_db"] = (
            threshold.tolist() if hasattr(threshold, "tolist") else list(threshold)
        )

    for det in detections:
        plot_data["detections"].append(
            {
                "id": det.detection_id,
                "center_freq_hz": det.center_freq_hz,
                "bandwidth_hz": det.bandwidth_hz,
                "peak_power_db": det.peak_power_db,
                "snr_db": det.snr_db,
                "start_bin": det.start_bin,
                "end_bin": det.end_bin,
            }
        )

    return plot_data
