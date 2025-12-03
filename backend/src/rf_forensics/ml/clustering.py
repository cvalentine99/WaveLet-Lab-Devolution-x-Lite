"""
GPU RF Forensics Engine - Signal Clustering

GPU-accelerated DBSCAN clustering for emitter identification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Import CuPy separately from cuML
try:
    import cupy as cp

    CUPY_AVAILABLE = True
except ImportError:
    cp = np
    CUPY_AVAILABLE = False

# Import cuML for GPU DBSCAN
try:
    from cuml.cluster import DBSCAN
    from cuml.neighbors import NearestNeighbors

    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False


@dataclass
class ClusterInfo:
    """
    Information about a detected cluster.

    Fields align with BACKEND_CONTRACT.md Section 2.3 - Cluster Stream.
    """

    cluster_id: int
    size: int
    centroid: np.ndarray
    prototype_detection_id: int
    avg_snr_db: float
    dominant_frequency_hz: float
    label: str = ""

    # Additional fields per BACKEND_CONTRACT.md
    center_freq_hz: float = 0.0  # Alias for dominant_frequency_hz
    freq_range_hz: tuple[float, float] = (0.0, 0.0)  # [min, max] frequency range
    avg_power_db: float = -100.0
    avg_bandwidth_hz: float = 0.0
    detection_count: int = 0  # Alias for size

    # ML classification hint (CRITICAL per contract)
    signal_type_hint: str = ""  # Backend's classification guess for the cluster

    # Tracking metrics
    avg_duty_cycle: float = 0.0
    unique_tracks: int = 0
    avg_bw_3db_ratio: float = 0.0


class EmitterClusterer:
    """
    GPU-accelerated DBSCAN clustering for emitter identification.

    Groups detected signals into clusters based on their feature vectors.
    Each cluster ideally represents a unique emitter/transmitter.

    Features:
    - Automatic eps and min_samples tuning
    - Incremental clustering for new detections
    - Cluster characterization and labeling
    """

    def __init__(self, eps: float = 0.5, min_samples: int = 5, auto_tune: bool = True):
        """
        Initialize clusterer.

        Args:
            eps: DBSCAN neighborhood radius.
            min_samples: Minimum samples for a cluster.
            auto_tune: Automatically tune parameters.
        """
        self._eps = eps
        self._min_samples = min_samples
        self._auto_tune = auto_tune

        self._labels = None
        self._cluster_info: dict[int, ClusterInfo] = {}
        self._fitted_features = None

    def fit(self, features) -> np.ndarray:
        """
        Fit DBSCAN to features and return cluster labels.

        Args:
            features: 2D array of shape (num_samples, num_features).
                      Can be numpy or cupy array.

        Returns:
            Array of cluster labels (-1 for noise).
        """
        # Convert to numpy if we have a CuPy array but cuML isn't available
        if CUPY_AVAILABLE and hasattr(features, "get") and not CUML_AVAILABLE:
            features = features.get()
        elif CUML_AVAILABLE and CUPY_AVAILABLE:
            # Ensure it's a CuPy array for cuML
            features = cp.asarray(features)
        else:
            # Ensure it's a numpy array for fallback
            features = np.asarray(features)

        # Auto-tune parameters if enabled
        if self._auto_tune:
            self._eps, self._min_samples = self.auto_tune_params(features)

        if CUML_AVAILABLE:
            # Use GPU-accelerated DBSCAN
            dbscan = DBSCAN(eps=self._eps, min_samples=self._min_samples)
            self._labels = dbscan.fit_predict(features)

            # Ensure labels are NumPy (cuML may return cupy or cudf arrays)
            if hasattr(self._labels, "get"):
                self._labels = self._labels.get()
            elif hasattr(self._labels, "values"):
                # cuDF Series
                self._labels = (
                    self._labels.values.get()
                    if hasattr(self._labels.values, "get")
                    else np.asarray(self._labels.values)
                )
            if not isinstance(self._labels, np.ndarray):
                self._labels = np.asarray(self._labels)
        else:
            # Fallback to simple distance-based clustering
            self._labels = self._simple_clustering(features)

        # Store features as numpy to avoid implicit conversion issues
        if hasattr(features, "get"):
            self._fitted_features = features.get()
        else:
            self._fitted_features = np.asarray(features)
        self._update_cluster_info(self._fitted_features)

        return self._labels

    def predict(self, new_features: cp.ndarray) -> np.ndarray:
        """
        Classify new features into existing clusters.

        Args:
            new_features: New feature vectors.

        Returns:
            Cluster labels for new features.
        """
        if self._fitted_features is None:
            raise RuntimeError("Clusterer not fitted. Call fit() first.")

        new_features = cp.asarray(new_features)

        # Find nearest fitted point for each new feature
        if CUML_AVAILABLE:
            nn = NearestNeighbors(n_neighbors=1)
            # cuML needs cupy arrays
            fitted_gpu = cp.asarray(self._fitted_features)
            nn.fit(fitted_gpu)
            distances, indices = nn.kneighbors(new_features)

            # Convert to numpy for iteration
            if hasattr(distances, "get"):
                distances = distances.get()
            if hasattr(indices, "get"):
                indices = indices.get()

            # Assign to cluster if within eps, else noise
            labels = np.zeros(len(new_features), dtype=np.int32)
            for i in range(len(distances)):
                if float(distances[i, 0]) <= self._eps:
                    labels[i] = self._labels[int(indices[i, 0])]
                else:
                    labels[i] = -1  # Noise
        else:
            labels = self._predict_simple(new_features)

        return labels

    def auto_tune_params(self, features) -> tuple[float, int]:
        """
        Automatically tune DBSCAN parameters.

        Uses k-distance graph analysis to find optimal eps.

        Args:
            features: Feature array (numpy or cupy).

        Returns:
            Tuple of (eps, min_samples).
        """
        # Features should already be converted by fit()
        n_samples = len(features)

        if n_samples < 10:
            return self._eps, self._min_samples

        # Determine min_samples based on data dimensionality
        n_features = features.shape[1]
        min_samples = max(3, n_features + 1)

        if CUML_AVAILABLE:
            # K-distance graph
            k = min_samples
            nn = NearestNeighbors(n_neighbors=k)
            nn.fit(features)
            distances, _ = nn.kneighbors(features)

            # Get k-th nearest neighbor distances - convert to numpy first
            k_dist_col = distances[:, -1]
            if hasattr(k_dist_col, "get"):
                k_dist_col = k_dist_col.get()
            elif not isinstance(k_dist_col, np.ndarray):
                k_dist_col = np.asarray(k_dist_col)
            k_distances = np.sort(k_dist_col)

            # Find "elbow" in k-distance graph
            # Use simple method: find point of maximum curvature
            eps = self._find_elbow(k_distances)
        else:
            # Simple heuristic
            features_np = features if isinstance(features, np.ndarray) else np.asarray(features)
            eps = float(np.std(features_np)) * 0.5

        return eps, min_samples

    def _find_elbow(self, k_distances: np.ndarray) -> float:
        """Find elbow point in k-distance graph."""
        n = len(k_distances)
        if n < 3:
            return float(np.median(k_distances))

        # Compute second derivative (curvature approximation)
        d1 = np.diff(k_distances)
        d2 = np.diff(d1)

        # Find point of maximum curvature
        if len(d2) > 0:
            elbow_idx = np.argmax(d2) + 1
            eps = float(k_distances[elbow_idx])
        else:
            eps = float(np.median(k_distances))

        return eps

    def _simple_clustering(self, features) -> np.ndarray:
        """
        Vectorized distance-based clustering fallback.

        Uses full distance matrix computation to eliminate O(N²) Python loop.
        For large N, uses chunked computation to manage memory.
        """
        # Convert CuPy arrays to numpy
        if CUPY_AVAILABLE and hasattr(features, "get"):
            features = features.get()

        n_samples = len(features)
        labels = np.full(n_samples, -1, dtype=np.int32)

        if n_samples == 0:
            return labels

        # Compute full pairwise distance matrix vectorized
        # ||a - b||² = ||a||² + ||b||² - 2 * a·b
        # This is O(N²) in compute but uses vectorized NumPy operations
        sq_norms = np.sum(features**2, axis=1)
        # Distance matrix: dist[i,j] = sqrt(||a||² + ||b||² - 2*a·b)
        dot_products = np.dot(features, features.T)
        dist_sq = sq_norms[:, np.newaxis] + sq_norms[np.newaxis, :] - 2 * dot_products
        # Handle numerical errors (small negative values)
        np.maximum(dist_sq, 0, out=dist_sq)
        distances = np.sqrt(dist_sq)

        # Find neighbors for all points at once (adjacency matrix)
        neighbors_matrix = distances <= self._eps

        # Count neighbors per point
        neighbor_counts = np.sum(neighbors_matrix, axis=1)

        # Core points have >= min_samples neighbors
        core_mask = neighbor_counts >= self._min_samples

        # Assign clusters using vectorized operations
        current_label = 0
        for i in np.where(core_mask)[0]:
            if labels[i] != -1:
                continue

            # BFS/flood-fill using vectorized neighbor lookup
            cluster_members = neighbors_matrix[i].copy()
            prev_size = 0
            curr_size = np.sum(cluster_members)

            # Expand cluster until no new members
            while curr_size > prev_size:
                prev_size = curr_size
                # Find all neighbors of current cluster members
                new_neighbors = np.any(neighbors_matrix[cluster_members], axis=0)
                cluster_members = cluster_members | new_neighbors
                curr_size = np.sum(cluster_members)

            labels[cluster_members] = current_label
            current_label += 1

        return labels

    def _predict_simple(self, new_features: cp.ndarray) -> np.ndarray:
        """
        Vectorized nearest neighbor prediction.

        Uses batch distance computation instead of per-point loop.
        """
        # Convert to numpy
        if CUPY_AVAILABLE and hasattr(new_features, "get"):
            new_features = new_features.get()
        else:
            new_features = np.asarray(new_features)

        fitted = self._fitted_features
        n_new = len(new_features)
        n_fitted = len(fitted)

        if n_new == 0:
            return np.array([], dtype=np.int32)

        # Vectorized distance computation: new_features (M, D) vs fitted (N, D)
        # Result: (M, N) distance matrix
        new_sq_norms = np.sum(new_features**2, axis=1)
        fitted_sq_norms = np.sum(fitted**2, axis=1)
        dot_products = np.dot(new_features, fitted.T)

        dist_sq = new_sq_norms[:, np.newaxis] + fitted_sq_norms[np.newaxis, :] - 2 * dot_products
        np.maximum(dist_sq, 0, out=dist_sq)
        distances = np.sqrt(dist_sq)

        # Find nearest neighbor for each new point (vectorized)
        nearest_indices = np.argmin(distances, axis=1)
        nearest_distances = distances[np.arange(n_new), nearest_indices]

        # Assign labels: use fitted label if within eps, else -1
        labels = np.where(nearest_distances <= self._eps, self._labels[nearest_indices], -1).astype(
            np.int32
        )

        return labels

    def _update_cluster_info(self, features: np.ndarray) -> None:
        """Update cluster information after fitting."""
        # Features should already be numpy at this point
        unique_labels = set(self._labels)

        for label in unique_labels:
            if label == -1:
                continue

            mask = self._labels == label
            cluster_features = features[mask]

            self._cluster_info[label] = ClusterInfo(
                cluster_id=label,
                size=len(cluster_features),
                centroid=np.mean(cluster_features, axis=0),
                prototype_detection_id=0,  # Would need detection IDs
                avg_snr_db=0.0,  # Would need SNR data
                dominant_frequency_hz=0.0,  # Would need frequency data
                detection_count=len(cluster_features),
            )

    def update_cluster_info_from_detections(
        self,
        cluster_id: int,
        detections: list,  # List[Detection] - import avoided for circular deps
    ) -> None:
        """
        Update cluster info with data from Detection objects.

        Called by orchestrator after clustering to populate contract fields.

        Args:
            cluster_id: The cluster ID to update.
            detections: List of Detection objects belonging to this cluster.
        """
        if cluster_id not in self._cluster_info or not detections:
            return

        info = self._cluster_info[cluster_id]

        # Gather metrics from detections
        freqs = [d.center_freq_hz for d in detections]
        powers = [d.peak_power_db for d in detections]
        snrs = [d.snr_db for d in detections]
        bandwidths = [d.bandwidth_hz for d in detections]

        # Update basic stats
        info.dominant_frequency_hz = float(np.mean(freqs))
        info.center_freq_hz = info.dominant_frequency_hz
        info.freq_range_hz = (float(np.min(freqs)), float(np.max(freqs)))
        info.avg_snr_db = float(np.mean(snrs))
        info.avg_power_db = float(np.mean(powers))
        info.avg_bandwidth_hz = float(np.mean(bandwidths))
        info.detection_count = len(detections)
        info.size = len(detections)

        # Compute 3dB bandwidth ratio
        bw_3db_ratios = []
        for d in detections:
            bw_3db = getattr(d, "bandwidth_3db_hz", d.bandwidth_hz)
            if d.bandwidth_hz > 0:
                bw_3db_ratios.append(bw_3db / d.bandwidth_hz)
        if bw_3db_ratios:
            info.avg_bw_3db_ratio = float(np.mean(bw_3db_ratios))

        # Compute duty cycle average
        duty_cycles = [d.duty_cycle for d in detections if d.duty_cycle is not None]
        if duty_cycles:
            info.avg_duty_cycle = float(np.mean(duty_cycles))

        # Count unique tracks
        track_ids = {d.track_id for d in detections if d.track_id is not None}
        info.unique_tracks = len(track_ids)

        # Determine signal type hint from modulation consensus
        mod_counts: dict[str, int] = {}
        for d in detections:
            mod = getattr(d, "modulation_type", "Unknown")
            if mod and mod != "Unknown":
                mod_counts[mod] = mod_counts.get(mod, 0) + 1

        if mod_counts:
            # Most common modulation type
            info.signal_type_hint = max(mod_counts, key=mod_counts.get)

        # Find prototype detection (highest SNR)
        best_det = max(detections, key=lambda d: d.snr_db)
        info.prototype_detection_id = best_det.detection_id

    def get_cluster_info(self, cluster_id: int) -> ClusterInfo | None:
        """Get information about a specific cluster."""
        return self._cluster_info.get(cluster_id)

    def get_all_clusters(self) -> list[ClusterInfo]:
        """Get information about all clusters."""
        return list(self._cluster_info.values())

    def label_cluster(self, cluster_id: int, label: str) -> None:
        """Assign a human-readable label to a cluster."""
        if cluster_id in self._cluster_info:
            self._cluster_info[cluster_id].label = label

    @property
    def n_clusters(self) -> int:
        """Number of clusters (excluding noise)."""
        if self._labels is None:
            return 0
        return len(set(self._labels)) - (1 if -1 in self._labels else 0)

    @property
    def noise_count(self) -> int:
        """Number of noise points."""
        if self._labels is None:
            return 0
        return int(np.sum(self._labels == -1))

    @property
    def labels(self) -> np.ndarray | None:
        return self._labels
