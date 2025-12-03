/**
 * Demodulation Types - Protocol-specific decoded frames
 */

/**
 * LoRa demodulation result
 */
export interface LoRaFrame {
  /** Timestamp in nanoseconds */
  timestamp_ns: number;

  /** Center frequency in Hz */
  freqHz: number;

  /** Spreading factor (7-12) */
  spreadingFactor: number;

  /** Bandwidth in Hz */
  bandwidthHz: number;

  /** Coding rate (4/5, 4/6, 4/7, 4/8) */
  codingRate: string;

  /** Payload string */
  payload?: string;

  /** CRC valid flag */
  crcValid: boolean;

  /** RSSI in dBm */
  rssi: number;

  /** SNR in dB */
  snrDb: number;
}

/**
 * Bluetooth Low Energy packet
 */
export interface BLEPacket {
  /** Timestamp in nanoseconds */
  timestamp_ns: number;

  /** Channel number (0-39) */
  channel: number;

  /** Access address */
  accessAddress: number;

  /** Packet type (ADV_IND, SCAN_REQ, etc.) */
  packetType: string;

  /** Payload string */
  payload?: string;

  /** CRC valid flag */
  crcValid?: boolean;

  /** RSSI in dBm */
  rssi: number;

  /** Advertising address (if applicable) */
  advAddress?: string;
}

/**
 * Generic demodulated frame
 */
export interface GenericFrame {
  /** Protocol name */
  protocol: string;

  /** Frame type */
  frameType: string;

  /** Payload bytes */
  payload: Uint8Array;

  /** Metadata */
  metadata: Record<string, unknown>;
}

/**
 * Demodulation result union type
 */
export type DemodulationResult = LoRaFrame | BLEPacket | GenericFrame;
