/**
 * SDR Types - Software Defined Radio configuration and status
 */

/**
 * Software Defined Radio configuration
 */
export interface SDRConfig {
  /** Center frequency in Hz */
  centerFreqHz: number;

  /** Sample rate in Hz */
  sampleRateHz: number;

  /** RF bandwidth in Hz */
  bandwidthHz: number;

  /** RF gain (device-specific units) */
  gain: number;

  /** SDR device type */
  deviceType: 'USRP' | 'HackRF' | 'LimeSDR' | 'RTL-SDR' | 'Simulated';

  /** Device-specific parameters */
  deviceParams?: Record<string, unknown>;
}

/**
 * Processing pipeline status
 */
export interface PipelineStatus {
  /** Current pipeline state */
  state: 'IDLE' | 'CONFIGURING' | 'RUNNING' | 'PAUSED' | 'ERROR';

  /** Uptime in seconds */
  uptimeSeconds: number;

  /** Total samples processed */
  samplesProcessed: number;

  /** Total detections count */
  detectionsCount: number;

  /** Current throughput in Msps */
  currentThroughputMsps: number;

  /** GPU memory used in GB */
  gpuMemoryUsedGb: number;

  /** GPU utilization percentage */
  gpuUtilizationPercent?: number;

  /** Error message if state is ERROR */
  errorMessage?: string;
}
