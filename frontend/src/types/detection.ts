/**
 * Detection Types - RF signal detection from CFAR detector
 */

/**
 * Single RF signal detection from CFAR detector
 */
export interface Detection {
  /** Unique detection identifier */
  id: string;

  /** Center frequency in Hz */
  centerFreqHz: number;

  /** Signal bandwidth in Hz (generic) */
  bandwidthHz: number;

  /** 3dB bandwidth in Hz */
  bandwidth3dbHz?: number;

  /** 6dB bandwidth in Hz */
  bandwidth6dbHz?: number;

  /** Peak power in dBm */
  peakPowerDb: number;

  /** Signal-to-Noise Ratio in dB */
  snrDb: number;

  /** Start frequency bin index */
  startBin: number;

  /** End frequency bin index */
  endBin: number;

  /** Detection timestamp (Unix milliseconds) */
  timestamp: number;

  /** Duration in microseconds */
  durationUs?: number;

  /** Classified modulation type */
  modulationType?: string;

  /** Classification confidence (0-1) */
  confidence?: number;

  /** Top-K modulation predictions with confidence */
  topKPredictions?: Array<{ modulation: string; confidence: number }>;

  /** Assigned cluster ID */
  clusterId?: number;

  /** Symbol rate in symbols/second */
  symbolRate?: number;

  /** Anomaly score from autoencoder */
  anomalyScore?: number;

  /** Raw feature vector for ML */
  rawFeatures?: Record<string, number>;
}

/**
 * Tracked signal across multiple frames
 */
export interface TrackedSignal {
  /** Tracking ID */
  id: string;

  /** Associated detections */
  detections: Detection[];

  /** First seen timestamp */
  firstSeen: number;

  /** Last seen timestamp */
  lastSeen: number;

  /** Duty cycle (0-1) */
  dutyCycle: number;

  /** Average frequency */
  avgFreqHz: number;

  /** Frequency stability (std dev) */
  freqStdHz: number;
}
