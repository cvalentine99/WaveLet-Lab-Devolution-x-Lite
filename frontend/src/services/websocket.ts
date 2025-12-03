/**
 * WebSocket Service for GPU RF Forensics Backend
 * Handles binary spectrum data and JSON events
 */

import { useSpectrumStore } from '@/stores/spectrumStore';
import { useDetectionStore } from '@/stores/detectionStore';
import { useClusterStore } from '@/stores/clusterStore';
import { API_BASE } from '@/lib/api';

const WS_BASE = import.meta.env.VITE_BACKEND_WS_URL || 'ws://localhost:8765';

// Log WebSocket configuration on load
console.log('[WebSocket] Base URL:', WS_BASE);

export interface SpectrumMessage {
  timestamp: number;
  centerFreq: number;
  span: number;
  numBins: number;
  psdDb: Float32Array;
  frequencies: Float32Array;
}

export interface DetectionEvent {
  type: "detection";
  timestamp: number;
  data: {
    detection_id: number;
    center_freq_hz: number;
    bandwidth_hz: number;
    bandwidth_3db_hz?: number;
    bandwidth_6db_hz?: number;
    peak_power_db: number;
    snr_db: number;
    start_bin: number;
    end_bin: number;
    // AMC classification fields from backend
    modulation_type?: string;
    modulation_confidence?: number;
    top_k_predictions?: Array<{ modulation_type: string; confidence: number }>;
    // Tracking fields
    track_id?: string;
    duty_cycle?: number;
    frames_seen?: number;
    // Clustering & Anomaly fields
    cluster_id?: number;
    anomaly_score?: number;
    symbol_rate?: number;
  };
}

export interface ClusterUpdate {
  type: "clusters";
  timestamp: number;
  data: Array<{
    cluster_id: number;
    size: number;
    center_freq_hz: number;  // Backend uses center_freq_hz
    dominant_frequency_hz?: number;  // Legacy field alias
    freq_range_hz?: [number, number];  // [min, max] frequency range
    avg_snr_db: number;
    avg_power_db?: number;
    avg_bandwidth_hz?: number;
    detection_count?: number;
    label?: string;
    signal_type_hint?: string;  // Classification hint from backend
    avg_duty_cycle?: number;
    unique_tracks?: number;
    avg_bw_3db_ratio?: number;  // Bandwidth shape metric
  }>;
}

export interface DemodulationEvent {
  type: "lora" | "ble";
  timestamp: number;
  data: any;
}

export interface LoRaFrameEvent {
  type: "lora";
  timestamp: number;
  frame_id: number;
  center_freq_hz: number;
  spreading_factor: number;
  bandwidth_hz: number;
  coding_rate: string;
  payload_hex: string;
  crc_valid: boolean;
  snr_db: number;
  rssi_dbm: number;
}

export interface BLEPacketEvent {
  type: "ble";
  timestamp: number;
  packet_id: number;
  channel: number;
  access_address: string;
  pdu_type: string;
  payload_hex: string;
  crc_valid: boolean;
  rssi_dbm: number;
}

export interface IQMessage {
  timestamp: number;
  centerFreqHz: number;
  sampleRateHz: number;
  numSamples: number;
  iSamples: Float32Array;
  qSamples: Float32Array;
}

/**
 * Parse binary spectrum message from backend (v2 format with full precision)
 * Returns null if parsing fails (malformed data)
 *
 * Binary format (28-byte header):
 * - timestamp: float64 at offset 0 (8 bytes)
 * - center_freq: float64 at offset 8 (8 bytes) - full precision for GHz frequencies
 * - span: float64 at offset 16 (8 bytes) - full precision for GHz frequencies
 * - num_bins: uint32 at offset 24 (4 bytes) - supports >65535 bins
 * - payload: uint8 array starting at offset 28
 */
export function parseSpectrumMessage(arrayBuffer: ArrayBuffer): SpectrumMessage | null {
  try {
    // Validate minimum buffer size (28 byte header for v2 format)
    if (arrayBuffer.byteLength < 28) {
      console.error('[WebSocket] Spectrum message too short:', arrayBuffer.byteLength);
      return null;
    }

    const view = new DataView(arrayBuffer);

    // Parse header (little-endian, v2 format with float64 frequencies)
    const timestamp = view.getFloat64(0, true);
    const centerFreq = view.getFloat64(8, true);   // float64 for GHz precision
    const span = view.getFloat64(16, true);        // float64 for GHz precision
    const numBins = view.getUint32(24, true);      // uint32 for large FFT sizes

    // Validate numBins matches buffer size
    const expectedSize = 28 + numBins;
    if (arrayBuffer.byteLength < expectedSize) {
      console.error('[WebSocket] Spectrum buffer size mismatch. Expected:', expectedSize, 'Got:', arrayBuffer.byteLength);
      return null;
    }

    // Sanity check values
    if (numBins === 0 || !isFinite(centerFreq) || !isFinite(span) || span <= 0) {
      console.error('[WebSocket] Invalid spectrum header values:', { numBins, centerFreq, span });
      return null;
    }

    // Parse payload (header is 28 bytes in v2 format)
    const payload = new Uint8Array(arrayBuffer, 28, numBins);

    // Convert uint8 to dB values (0-255 maps to -120 to -20 dB)
    const psdDb = new Float32Array(numBins);
    for (let i = 0; i < numBins; i++) {
      psdDb[i] = (payload[i] / 255) * 100 - 120;
    }

    // Generate frequency axis
    const frequencies = new Float32Array(numBins);
    for (let i = 0; i < numBins; i++) {
      frequencies[i] = centerFreq - span / 2 + (i / numBins) * span;
    }

    return {
      timestamp,
      centerFreq,
      span,
      numBins,
      psdDb,
      frequencies
    };
  } catch (error) {
    console.error('[WebSocket] Failed to parse spectrum message:', error);
    return null;
  }
}

/**
 * Parse binary IQ message from backend
 * Binary format (20-byte header, matching spectrum format):
 * - Header (20 bytes): timestamp(f64), center_freq(f32), sample_rate(f32), num_samples(u32)
 * - Payload: interleaved float32 IQ pairs [I0, Q0, I1, Q1, ...]
 */
export function parseIQMessage(arrayBuffer: ArrayBuffer): IQMessage | null {
  try {
    // Validate minimum buffer size (20 byte header)
    if (arrayBuffer.byteLength < 20) {
      console.error('[WebSocket] IQ message too short:', arrayBuffer.byteLength);
      return null;
    }

    const view = new DataView(arrayBuffer);

    // Parse header (little-endian)
    const timestamp = view.getFloat64(0, true);
    const centerFreqHz = view.getFloat32(8, true);
    const sampleRateHz = view.getFloat32(12, true);
    const numSamples = view.getUint32(16, true);

    // Validate buffer size
    const expectedSize = 20 + numSamples * 2 * 4; // 2 floats per sample, 4 bytes per float
    if (arrayBuffer.byteLength < expectedSize) {
      console.error('[WebSocket] IQ buffer size mismatch. Expected:', expectedSize, 'Got:', arrayBuffer.byteLength);
      return null;
    }

    // Parse interleaved IQ samples
    const interleavedView = new Float32Array(arrayBuffer, 20, numSamples * 2);
    const iSamples = new Float32Array(numSamples);
    const qSamples = new Float32Array(numSamples);

    for (let i = 0; i < numSamples; i++) {
      iSamples[i] = interleavedView[i * 2];
      qSamples[i] = interleavedView[i * 2 + 1];
    }

    return {
      timestamp,
      centerFreqHz,
      sampleRateHz,
      numSamples,
      iSamples,
      qSamples
    };
  } catch (error) {
    console.error('[WebSocket] Failed to parse IQ message:', error);
    return null;
  }
}

/**
 * Parse JSON IQ message from backend (fallback for JSON mode)
 */
export function parseIQJsonMessage(data: any): IQMessage | null {
  try {
    if (data.type !== 'iq') return null;

    return {
      timestamp: data.timestamp,
      centerFreqHz: data.centerFreqHz,
      sampleRateHz: data.sampleRateHz,
      numSamples: data.numSamples,
      iSamples: new Float32Array(data.iSamples),
      qSamples: new Float32Array(data.qSamples)
    };
  } catch (error) {
    console.error('[WebSocket] Failed to parse IQ JSON message:', error);
    return null;
  }
}

/**
 * WebSocket Manager Class
 */
export class WebSocketManager {
  private ws: WebSocket | null = null;
  private reconnectTimeout: number | null = null;
  private reconnectInterval = 3000;
  private url: string;
  private isIntentionallyClosed = false;

  constructor(endpoint: string) {
    this.url = `${WS_BASE}${endpoint}`;
  }

  connect(
    onBinaryMessage?: (data: ArrayBuffer) => void,
    onJsonMessage?: (data: any) => void
  ) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      return; // Already connected
    }

    this.isIntentionallyClosed = false;
    this.ws = new WebSocket(this.url);
    this.ws.binaryType = 'arraybuffer';

    this.ws.onopen = () => {
      console.log(`[WebSocket] Connected to ${this.url}`);
    };

    this.ws.onmessage = (event) => {
      if (event.data instanceof ArrayBuffer) {
        onBinaryMessage?.(event.data);
      } else {
        try {
          const json = JSON.parse(event.data);
          onJsonMessage?.(json);
        } catch (e) {
          console.error('[WebSocket] Failed to parse JSON:', e);
        }
      }
    };

    this.ws.onclose = () => {
      console.log(`[WebSocket] Disconnected from ${this.url}`);
      if (!this.isIntentionallyClosed) {
        this.reconnectTimeout = window.setTimeout(() => {
          this.connect(onBinaryMessage, onJsonMessage);
        }, this.reconnectInterval);
      }
    };

    this.ws.onerror = (error) => {
      console.error('[WebSocket] Error:', error);
    };
  }

  disconnect() {
    this.isIntentionallyClosed = true;
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  send(data: string | ArrayBuffer) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(data);
    } else {
      console.warn('[WebSocket] Cannot send, not connected');
    }
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

/**
 * Global WebSocket instances
 */
let spectrumWS: WebSocketManager | null = null;
let detectionsWS: WebSocketManager | null = null;
let clustersWS: WebSocketManager | null = null;
let iqWS: WebSocketManager | null = null;
let demodulationWS: WebSocketManager | null = null;

// LoRa frame callback - set by DecoderPanel or other component
let loraFrameCallback: ((frame: LoRaFrameEvent) => void) | null = null;

// BLE packet callback - set by DecoderPanel or other component
let blePacketCallback: ((packet: BLEPacketEvent) => void) | null = null;

export function setLoRaFrameCallback(callback: (frame: LoRaFrameEvent) => void) {
  loraFrameCallback = callback;
}

export function setBLEPacketCallback(callback: (packet: BLEPacketEvent) => void) {
  blePacketCallback = callback;
}

/**
 * Start spectrum WebSocket
 */
export function startSpectrumStream() {
  if (!spectrumWS) {
    spectrumWS = new WebSocketManager('/ws/spectrum');
  }

  spectrumWS.connect(
    // Binary message handler (high performance)
    (data) => {
      const message = parseSpectrumMessage(data);
      if (message) {
        useSpectrumStore.getState().updatePsd(message.psdDb, message.timestamp);
        // Binary protocol: span = visible bandwidth = sampleRateHz for full-span FFT
        // centerFreq, span are in Hz
        useSpectrumStore.getState().setFrequencyRange(
          message.centerFreq,  // centerHz
          message.span,        // spanHz (visible bandwidth)
          message.span         // sampleRateHz (= span for full-span display)
        );
      }
      // Silently skip malformed messages - error already logged in parser
    },
    // JSON message handler (fallback)
    (json: any) => {
      if (json.type === 'spectrum' && json.magnitudeDb) {
        const psdDb = new Float32Array(json.magnitudeDb);
        useSpectrumStore.getState().updatePsd(psdDb, json.timestamp);
        // JSON format: use spanHz if available, else fall back to sampleRateHz
        const spanHz = json.spanHz || json.bandwidth_hz || json.sampleRateHz;
        useSpectrumStore.getState().setFrequencyRange(
          json.centerFreqHz,    // centerHz
          spanHz,               // spanHz (visible bandwidth)
          json.sampleRateHz     // sampleRateHz (actual sample rate)
        );
      }
    }
  );
}

/**
 * Start detections WebSocket
 */
export function startDetectionsStream() {
  if (!detectionsWS) {
    detectionsWS = new WebSocketManager('/ws/detections');
  }

  detectionsWS.connect(
    undefined,
    (event: DetectionEvent) => {
      if (event.type === 'detection') {
        // Map top_k_predictions from backend format to frontend format
        const topKPredictions = event.data.top_k_predictions?.map(p => ({
          modulation: p.modulation_type,
          confidence: p.confidence
        }));

        const detection = {
          id: `det-${event.data.detection_id}`,
          centerFreqHz: event.data.center_freq_hz,
          bandwidthHz: event.data.bandwidth_hz,
          bandwidth3dbHz: event.data.bandwidth_3db_hz,
          bandwidth6dbHz: event.data.bandwidth_6db_hz,
          peakPowerDb: event.data.peak_power_db,
          snrDb: event.data.snr_db,
          startBin: event.data.start_bin,
          endBin: event.data.end_bin,
          timestamp: event.timestamp * 1000,
          // AMC classification fields
          modulationType: event.data.modulation_type,
          confidence: event.data.modulation_confidence,
          topKPredictions,
          // Clustering & Anomaly fields
          clusterId: event.data.cluster_id,
          anomalyScore: event.data.anomaly_score,
          symbolRate: event.data.symbol_rate,
        };
        useDetectionStore.getState().addDetection(detection);
      }
    }
  );
}

/**
 * Start clusters WebSocket
 */
export function startClustersStream() {
  if (!clustersWS) {
    clustersWS = new WebSocketManager('/ws/clusters');
  }

  clustersWS.connect(
    undefined,
    (event: ClusterUpdate) => {
      if (event.type === 'clusters') {
        event.data.forEach(cluster => {
          // Use center_freq_hz with fallback to legacy dominant_frequency_hz
          const freqHz = cluster.center_freq_hz || cluster.dominant_frequency_hz || 0;

          useClusterStore.getState().updateCluster({
            id: cluster.cluster_id,
            size: cluster.size,
            avgSnrDb: cluster.avg_snr_db,
            dominantFreqHz: freqHz,
            freqRangeHz: cluster.freq_range_hz,
            avgPowerDb: cluster.avg_power_db,
            avgBandwidthHz: cluster.avg_bandwidth_hz,
            detectionCount: cluster.detection_count,
            label: cluster.label,
            signalTypeHint: cluster.signal_type_hint,
            avgDutyCycle: cluster.avg_duty_cycle,
            uniqueTracks: cluster.unique_tracks,
            avgBw3dbRatio: cluster.avg_bw_3db_ratio,
            centroid: [freqHz, cluster.avg_snr_db],
            color: '',
          });
        });
      }
    }
  );
}

/**
 * Start IQ WebSocket stream (for constellation display)
 */
export function startIQStream() {
  if (!iqWS) {
    iqWS = new WebSocketManager('/ws/iq');
  }

  iqWS.connect(
    // Binary message handler
    (data) => {
      const message = parseIQMessage(data);
      if (message) {
        useSpectrumStore.getState().updateIQ(
          message.iSamples,
          message.qSamples,
          message.timestamp
        );
      }
    },
    // JSON message handler (fallback)
    (json) => {
      const message = parseIQJsonMessage(json);
      if (message) {
        useSpectrumStore.getState().updateIQ(
          message.iSamples,
          message.qSamples,
          message.timestamp
        );
      }
    }
  );
}

/**
 * Stop IQ WebSocket stream
 */
export function stopIQStream() {
  iqWS?.disconnect();
}

/**
 * Start unified demodulation WebSocket stream (for decoded LoRa/BLE packets)
 * Backend sends both LoRa and BLE frames on /ws/demodulation
 */
export function startDemodulationStream() {
  if (!demodulationWS) {
    demodulationWS = new WebSocketManager('/ws/demodulation');
  }

  demodulationWS.connect(
    undefined, // No binary handler for demodulation
    (json: LoRaFrameEvent | BLEPacketEvent) => {
      if (json.type === 'lora' && loraFrameCallback) {
        loraFrameCallback(json as LoRaFrameEvent);
      } else if (json.type === 'ble' && blePacketCallback) {
        blePacketCallback(json as BLEPacketEvent);
      }
    }
  );
}

/**
 * Stop demodulation WebSocket stream
 */
export function stopDemodulationStream() {
  demodulationWS?.disconnect();
}

// Legacy aliases for backwards compatibility
export const startLoRaStream = startDemodulationStream;
export const stopLoRaStream = stopDemodulationStream;
export const startBLEStream = startDemodulationStream;
export const stopBLEStream = stopDemodulationStream;

/**
 * Stop all WebSocket streams
 */
export function stopAllStreams() {
  spectrumWS?.disconnect();
  detectionsWS?.disconnect();
  clustersWS?.disconnect();
  iqWS?.disconnect();
  demodulationWS?.disconnect();
}

/**
 * Check if streams are connected
 */
export function areStreamsConnected() {
  return {
    spectrum: spectrumWS?.isConnected ?? false,
    detections: detectionsWS?.isConnected ?? false,
    clusters: clustersWS?.isConnected ?? false,
    iq: iqWS?.isConnected ?? false,
    demodulation: demodulationWS?.isConnected ?? false,
  };
}

/**
 * Fetch initial detections and clusters from REST API
 * Called when starting streams to populate stores with existing data
 */
export async function fetchInitialData() {
  try {
    // Fetch existing detections
    const detectionsResponse = await fetch(`${API_BASE}/api/detections`);
    if (detectionsResponse.ok) {
      const data = await detectionsResponse.json();
      const detections = data.detections || [];
      console.log(`[REST] Loaded ${detections.length} initial detections`);

      detections.forEach((det: any) => {
        const topKPredictions = det.top_k_predictions?.map((p: any) => ({
          modulation: p.modulation_type,
          confidence: p.confidence,
        }));

        useDetectionStore.getState().addDetection({
          id: `det-${det.detection_id}`,
          centerFreqHz: det.center_freq_hz,
          bandwidthHz: det.bandwidth_hz,
          bandwidth3dbHz: det.bandwidth_3db_hz,
          bandwidth6dbHz: det.bandwidth_6db_hz,
          peakPowerDb: det.peak_power_db,
          snrDb: det.snr_db,
          startBin: det.start_bin,
          endBin: det.end_bin,
          timestamp: det.timestamp * 1000,
          modulationType: det.modulation_type,
          confidence: det.modulation_confidence,
          topKPredictions,
          clusterId: det.cluster_id,
          anomalyScore: det.anomaly_score,
          symbolRate: det.symbol_rate,
        });
      });
    }

    // Fetch existing clusters
    const clustersResponse = await fetch(`${API_BASE}/api/clusters`);
    if (clustersResponse.ok) {
      const data = await clustersResponse.json();
      const clusters = data.clusters || [];
      console.log(`[REST] Loaded ${clusters.length} initial clusters`);

      clusters.forEach((cluster: any) => {
        const freqHz = cluster.center_freq_hz || cluster.dominant_frequency_hz || 0;
        useClusterStore.getState().updateCluster({
          id: cluster.cluster_id,
          size: cluster.size,
          avgSnrDb: cluster.avg_snr_db,
          dominantFreqHz: freqHz,
          freqRangeHz: cluster.freq_range_hz,
          avgPowerDb: cluster.avg_power_db,
          avgBandwidthHz: cluster.avg_bandwidth_hz,
          detectionCount: cluster.detection_count,
          label: cluster.label,
          signalTypeHint: cluster.signal_type_hint,
          avgDutyCycle: cluster.avg_duty_cycle,
          uniqueTracks: cluster.unique_tracks,
          avgBw3dbRatio: cluster.avg_bw_3db_ratio,
          centroid: [freqHz, cluster.avg_snr_db],
          color: '',
        });
      });
    }
  } catch (error) {
    console.error('[REST] Failed to fetch initial data:', error);
  }
}
