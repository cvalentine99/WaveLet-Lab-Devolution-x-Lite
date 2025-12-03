/**
 * WebSocket Message Types - Real-time communication with backend
 */

import type { Detection } from './detection';
import type { Cluster } from './cluster';
import type { SDRConfig, PipelineStatus } from './sdr';

/**
 * Binary spectrum message from backend
 * Header: 16 bytes (timestamp: 8, centerFreq: 4, span: 4)
 * Payload: uint8 array of quantized dB values
 */
export interface SpectrumMessage {
  /** Timestamp (Unix milliseconds) */
  timestamp: number;

  /** Center frequency in Hz */
  centerFreq: number;

  /** Frequency span in Hz */
  span: number;

  /** PSD data (uint8 quantized to 0-255) */
  psdData: Uint8Array;

  /** Number of frequency bins */
  numBins: number;
}

/**
 * Detection event from WebSocket
 */
export interface DetectionEvent {
  /** Event type */
  type: 'detection';

  /** Detection data */
  detection: Detection;
}

/**
 * Cluster update from WebSocket
 */
export interface ClusterUpdate {
  /** Event type */
  type: 'cluster';

  /** Cluster data */
  cluster: Cluster;
}

/**
 * Configuration change event
 */
export interface ConfigChange {
  /** Event type */
  type: 'config';

  /** Updated configuration */
  config: Partial<SDRConfig>;
}

/**
 * Pipeline status update
 */
export interface StatusUpdate {
  /** Event type */
  type: 'status';

  /** Pipeline status */
  status: PipelineStatus;
}

/**
 * Union type for all WebSocket messages
 */
export type WebSocketMessage =
  | DetectionEvent
  | ClusterUpdate
  | ConfigChange
  | StatusUpdate;
