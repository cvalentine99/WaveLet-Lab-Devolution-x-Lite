/**
 * Cluster Types - Emitter clustering from cuML DBSCAN
 */

/**
 * Emitter cluster from cuML DBSCAN
 *
 * Fields match backend orchestrator.py _run_clustering() output
 */
export interface Cluster {
  /** Cluster ID */
  id: number;

  /** Number of detections in cluster */
  size: number;

  /** Centroid in feature space [freq, snr] */
  centroid: number[];

  /** Average SNR of cluster members */
  avgSnrDb: number;

  /** Dominant/center frequency in Hz */
  dominantFreqHz: number;

  /** Frequency range [min, max] in Hz */
  freqRangeHz?: [number, number];

  /** Average power in dB */
  avgPowerDb?: number;

  /** Average bandwidth in Hz */
  avgBandwidthHz?: number;

  /** Total detection count in cluster */
  detectionCount?: number;

  /** User-assigned or auto-generated label */
  label?: string;

  /** Display color (hex) */
  color: string;

  /** Cluster confidence score */
  confidence?: number;

  /** Signal type hint from backend heuristics */
  signalTypeHint?: string;

  /** Average duty cycle (0-1) from tracking */
  avgDutyCycle?: number;

  /** Number of unique track IDs in cluster */
  uniqueTracks?: number;

  /** Average 3dB/total bandwidth ratio (shape metric) */
  avgBw3dbRatio?: number;
}
