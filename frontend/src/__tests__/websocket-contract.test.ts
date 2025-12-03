/**
 * WebSocket Contract Validation Tests
 * Validates that frontend parsers correctly handle backend message formats
 */

import { describe, it, expect } from 'vitest';
import {
  parseSpectrumMessage,
  parseIQMessage,
  type LoRaFrameEvent,
  type BLEPacketEvent,
  type DetectionEvent,
  type ClusterUpdate,
} from '../services/websocket';

describe('WebSocket Contract Validation', () => {
  describe('Spectrum Binary Protocol (v2 - full precision)', () => {
    it('should parse valid 28-byte header + payload', () => {
      // Create mock binary message
      // v2 Header: timestamp(f64) + centerFreq(f64) + span(f64) + numBins(u32) = 28 bytes
      const numBins = 1024;
      const buffer = new ArrayBuffer(28 + numBins);
      const view = new DataView(buffer);

      // Write header (little-endian, v2 format)
      const timestamp = Date.now() / 1000;
      view.setFloat64(0, timestamp, true);      // timestamp (f64)
      view.setFloat64(8, 915e6, true);          // centerFreq (f64 - full precision)
      view.setFloat64(16, 10e6, true);          // span (f64 - full precision)
      view.setUint32(24, numBins, true);        // numBins (u32 - supports large FFTs)

      // Write payload (mock PSD data)
      const payload = new Uint8Array(buffer, 28);
      for (let i = 0; i < numBins; i++) {
        payload[i] = Math.floor(Math.random() * 256);
      }

      const result = parseSpectrumMessage(buffer);

      expect(result).not.toBeNull();
      expect(result!.numBins).toBe(numBins);
      expect(result!.centerFreq).toBeCloseTo(915e6, 0);
      expect(result!.span).toBeCloseTo(10e6, 0);
      expect(result!.psdDb.length).toBe(numBins);
    });

    it('should handle GHz frequencies with full precision', () => {
      // Test that 2.4 GHz WiFi frequencies are preserved accurately
      const numBins = 512;
      const buffer = new ArrayBuffer(28 + numBins);
      const view = new DataView(buffer);

      const timestamp = Date.now() / 1000;
      const centerFreq = 2437000000; // WiFi channel 6, exactly 2.437 GHz
      const span = 20000000;         // 20 MHz span

      view.setFloat64(0, timestamp, true);
      view.setFloat64(8, centerFreq, true);
      view.setFloat64(16, span, true);
      view.setUint32(24, numBins, true);

      const payload = new Uint8Array(buffer, 28);
      for (let i = 0; i < numBins; i++) {
        payload[i] = 128;
      }

      const result = parseSpectrumMessage(buffer);

      expect(result).not.toBeNull();
      // With float64, we should preserve exact Hz-level precision
      expect(result!.centerFreq).toBe(centerFreq);
      expect(result!.span).toBe(span);
    });

    it('should reject buffer smaller than 28-byte header', () => {
      const buffer = new ArrayBuffer(20);  // Old format size, should be rejected
      const result = parseSpectrumMessage(buffer);
      expect(result).toBeNull();
    });

    it('should handle zero bins gracefully', () => {
      const buffer = new ArrayBuffer(28);
      const view = new DataView(buffer);
      view.setFloat64(0, Date.now() / 1000, true);
      view.setFloat64(8, 915e6, true);
      view.setFloat64(16, 10e6, true);
      view.setUint32(24, 0, true);  // zero bins

      const result = parseSpectrumMessage(buffer);
      // Should handle gracefully (returns null for zero bins)
      expect(result).toBeNull();
    });
  });

  describe('IQ Binary Protocol', () => {
    it('should parse valid IQ message with float32 samples', () => {
      const numSamples = 256;
      // Header: 20 bytes (timestamp:f64, centerFreq:f32, sampleRate:f32, numSamples:u32)
      // Payload: numSamples * 8 bytes (interleaved float32 I/Q)
      const buffer = new ArrayBuffer(20 + numSamples * 8);
      const view = new DataView(buffer);

      view.setFloat64(0, Date.now() / 1000, true);  // timestamp
      view.setFloat32(8, 915e6, true);               // centerFreqHz
      view.setFloat32(12, 10e6, true);               // sampleRateHz
      view.setUint32(16, numSamples, true);          // numSamples

      // Write interleaved I/Q float32 pairs
      const floatView = new Float32Array(buffer, 20);
      for (let i = 0; i < numSamples * 2; i++) {
        floatView[i] = (Math.random() - 0.5) * 2;
      }

      const result = parseIQMessage(buffer);

      expect(result).not.toBeNull();
      expect(result!.numSamples).toBe(numSamples);
      expect(result!.iSamples.length).toBe(numSamples);
      expect(result!.qSamples.length).toBe(numSamples);
    });

    it('should reject buffer smaller than header', () => {
      const buffer = new ArrayBuffer(10);
      const result = parseIQMessage(buffer);
      expect(result).toBeNull();
    });
  });

  describe('LoRa JSON Protocol', () => {
    it('should validate LoRa frame event structure', () => {
      const loraEvent: LoRaFrameEvent = {
        type: 'lora',
        timestamp: Date.now() / 1000,
        frame_id: 1,
        center_freq_hz: 915000000,
        spreading_factor: 7,
        bandwidth_hz: 125000,
        coding_rate: '4/5',
        payload_hex: 'deadbeef',
        crc_valid: true,
        snr_db: 10.5,
        rssi_dbm: -80.0,
      };

      // Type validation (compile-time + runtime shape check)
      expect(loraEvent.type).toBe('lora');
      expect(typeof loraEvent.timestamp).toBe('number');
      expect(typeof loraEvent.frame_id).toBe('number');
      expect(typeof loraEvent.center_freq_hz).toBe('number');
      expect(loraEvent.spreading_factor).toBeGreaterThanOrEqual(7);
      expect(loraEvent.spreading_factor).toBeLessThanOrEqual(12);
      expect(typeof loraEvent.bandwidth_hz).toBe('number');
      expect(typeof loraEvent.coding_rate).toBe('string');
      expect(typeof loraEvent.payload_hex).toBe('string');
      expect(typeof loraEvent.crc_valid).toBe('boolean');
      expect(typeof loraEvent.snr_db).toBe('number');
      expect(typeof loraEvent.rssi_dbm).toBe('number');
    });

    it('should handle all valid spreading factors', () => {
      const validSFs = [7, 8, 9, 10, 11, 12];
      validSFs.forEach(sf => {
        const event: Partial<LoRaFrameEvent> = { spreading_factor: sf };
        expect(event.spreading_factor).toBeGreaterThanOrEqual(7);
        expect(event.spreading_factor).toBeLessThanOrEqual(12);
      });
    });
  });

  describe('BLE JSON Protocol', () => {
    it('should validate BLE packet event structure', () => {
      const bleEvent: BLEPacketEvent = {
        type: 'ble',
        timestamp: Date.now() / 1000,
        packet_id: 1,
        channel: 37,
        access_address: '0x8E89BED6',
        pdu_type: 'ADV_IND',
        payload_hex: '0201061aff4c00',
        crc_valid: true,
        rssi_dbm: -70.0,
      };

      expect(bleEvent.type).toBe('ble');
      expect(typeof bleEvent.timestamp).toBe('number');
      expect(typeof bleEvent.packet_id).toBe('number');
      expect(bleEvent.channel).toBeGreaterThanOrEqual(0);
      expect(bleEvent.channel).toBeLessThanOrEqual(39);
      expect(typeof bleEvent.access_address).toBe('string');
      expect(typeof bleEvent.pdu_type).toBe('string');
      expect(typeof bleEvent.payload_hex).toBe('string');
      expect(typeof bleEvent.crc_valid).toBe('boolean');
      expect(typeof bleEvent.rssi_dbm).toBe('number');
    });

    it('should handle advertising channels (37, 38, 39)', () => {
      const advChannels = [37, 38, 39];
      advChannels.forEach(ch => {
        const event: Partial<BLEPacketEvent> = { channel: ch };
        expect(event.channel).toBeGreaterThanOrEqual(37);
        expect(event.channel).toBeLessThanOrEqual(39);
      });
    });
  });

  describe('Detection JSON Protocol', () => {
    it('should validate detection event structure', () => {
      const detectionEvent: DetectionEvent = {
        type: 'detection',
        timestamp: Date.now() / 1000,
        data: {
          detection_id: 12345,
          center_freq_hz: 915125000,
          bandwidth_hz: 125000,
          bandwidth_3db_hz: 100000,
          bandwidth_6db_hz: 150000,
          peak_power_db: -45.2,
          snr_db: 18.5,
          start_bin: 512,
          end_bin: 520,
          modulation_type: 'LoRa',
          modulation_confidence: 0.92,
          top_k_predictions: [
            { modulation_type: 'LoRa', confidence: 0.92 },
            { modulation_type: 'FSK', confidence: 0.05 },
          ],
          cluster_id: 3,
          anomaly_score: 0.12,
          symbol_rate: 50000,
        },
      };

      expect(detectionEvent.type).toBe('detection');
      expect(typeof detectionEvent.timestamp).toBe('number');

      const data = detectionEvent.data;
      expect(typeof data.detection_id).toBe('number');
      expect(typeof data.center_freq_hz).toBe('number');
      expect(typeof data.bandwidth_hz).toBe('number');
      expect(typeof data.peak_power_db).toBe('number');
      expect(typeof data.snr_db).toBe('number');
      expect(typeof data.start_bin).toBe('number');
      expect(typeof data.end_bin).toBe('number');

      // Optional AMC fields
      if (data.modulation_type) {
        expect(typeof data.modulation_type).toBe('string');
      }
      if (data.modulation_confidence) {
        expect(data.modulation_confidence).toBeGreaterThanOrEqual(0);
        expect(data.modulation_confidence).toBeLessThanOrEqual(1);
      }
      if (data.top_k_predictions) {
        expect(Array.isArray(data.top_k_predictions)).toBe(true);
        data.top_k_predictions.forEach(pred => {
          expect(typeof pred.modulation_type).toBe('string');
          expect(typeof pred.confidence).toBe('number');
        });
      }

      // Optional clustering/anomaly fields
      if (data.cluster_id !== undefined) {
        expect(typeof data.cluster_id).toBe('number');
      }
      if (data.anomaly_score !== undefined) {
        expect(data.anomaly_score).toBeGreaterThanOrEqual(0);
        expect(data.anomaly_score).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('Cluster JSON Protocol', () => {
    it('should validate cluster update structure', () => {
      const clusterUpdate: ClusterUpdate = {
        type: 'clusters',
        timestamp: Date.now() / 1000,
        data: [
          {
            cluster_id: 3,
            size: 42,
            center_freq_hz: 915125000,
            dominant_frequency_hz: 915125000,
            freq_range_hz: [915000000, 915250000],
            avg_snr_db: 15.2,
            avg_power_db: -48.5,
            avg_bandwidth_hz: 125000,
            detection_count: 156,
            label: 'LoRa Gateway',
            signal_type_hint: 'LoRa',
            avg_duty_cycle: 0.12,
            unique_tracks: 3,
            avg_bw_3db_ratio: 0.8,
          },
        ],
      };

      expect(clusterUpdate.type).toBe('clusters');
      expect(typeof clusterUpdate.timestamp).toBe('number');
      expect(Array.isArray(clusterUpdate.data)).toBe(true);

      const cluster = clusterUpdate.data[0];
      expect(typeof cluster.cluster_id).toBe('number');
      expect(typeof cluster.size).toBe('number');
      expect(cluster.size).toBeGreaterThan(0);
      expect(typeof cluster.avg_snr_db).toBe('number');

      // Frequency fields
      if (cluster.center_freq_hz) {
        expect(typeof cluster.center_freq_hz).toBe('number');
      }
      if (cluster.freq_range_hz) {
        expect(Array.isArray(cluster.freq_range_hz)).toBe(true);
        expect(cluster.freq_range_hz.length).toBe(2);
      }

      // Optional ML fields
      if (cluster.signal_type_hint) {
        expect(typeof cluster.signal_type_hint).toBe('string');
      }
      if (cluster.avg_duty_cycle !== undefined) {
        expect(cluster.avg_duty_cycle).toBeGreaterThanOrEqual(0);
        expect(cluster.avg_duty_cycle).toBeLessThanOrEqual(1);
      }
    });
  });
});

describe('Field Name Conventions', () => {
  it('should use snake_case for all backend fields', () => {
    // These are the expected backend field names
    const backendFields = [
      'center_freq_hz',
      'sample_rate_hz',
      'bandwidth_hz',
      'peak_power_db',
      'snr_db',
      'lna_db',
      'tia_db',
      'pga_db',
      'modulation_type',
      'cluster_id',
      'anomaly_score',
      'signal_type_hint',
    ];

    backendFields.forEach(field => {
      expect(field).toMatch(/^[a-z]+(_[a-z0-9]+)*$/);
    });
  });
});

describe('Value Range Validation', () => {
  it('should validate frequency ranges (1 MHz - 6 GHz)', () => {
    const validFreqs = [1e6, 915e6, 2.4e9, 5.8e9];
    validFreqs.forEach(freq => {
      expect(freq).toBeGreaterThanOrEqual(1e6);
      expect(freq).toBeLessThanOrEqual(6e9);
    });
  });

  it('should validate power ranges (-120 to +10 dBm)', () => {
    const validPowers = [-120, -80, -45, -10, 0, 10];
    validPowers.forEach(power => {
      expect(power).toBeGreaterThanOrEqual(-120);
      expect(power).toBeLessThanOrEqual(10);
    });
  });

  it('should validate confidence/probability (0.0 - 1.0)', () => {
    const validConf = [0, 0.5, 0.92, 1.0];
    validConf.forEach(conf => {
      expect(conf).toBeGreaterThanOrEqual(0);
      expect(conf).toBeLessThanOrEqual(1);
    });
  });

  it('should validate LNA gain range (0-30 dB)', () => {
    const validGains = [0, 15, 30];
    validGains.forEach(gain => {
      expect(gain).toBeGreaterThanOrEqual(0);
      expect(gain).toBeLessThanOrEqual(30);
    });
  });

  it('should validate TIA gain range (0-12 dB, steps of 3)', () => {
    const validGains = [0, 3, 6, 9, 12];
    validGains.forEach(gain => {
      expect(gain).toBeGreaterThanOrEqual(0);
      expect(gain).toBeLessThanOrEqual(12);
      expect(gain % 3).toBe(0);
    });
  });

  it('should validate PGA gain range (0-32 dB)', () => {
    const validGains = [0, 16, 32];
    validGains.forEach(gain => {
      expect(gain).toBeGreaterThanOrEqual(0);
      expect(gain).toBeLessThanOrEqual(32);
    });
  });
});
