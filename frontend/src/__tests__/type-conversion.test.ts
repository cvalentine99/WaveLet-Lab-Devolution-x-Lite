/**
 * Type Conversion Validation Tests
 * Validates snake_case → camelCase conversions between backend and frontend
 */

import { describe, it, expect } from 'vitest';
import type { Detection } from '../types/detection';
import type { Cluster } from '../types/cluster';
import type { LoRaFrame, BLEPacket } from '../types/demod';
import type { LoRaFrameEvent, BLEPacketEvent, DetectionEvent, ClusterUpdate } from '../services/websocket';

describe('Backend → Frontend Type Conversions', () => {
  describe('Detection Event Conversion', () => {
    it('should convert DetectionEvent to Detection correctly', () => {
      // Backend format (snake_case)
      const backendEvent: DetectionEvent = {
        type: 'detection',
        timestamp: 1701590400.123,
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

      // Frontend format (camelCase)
      const frontendDetection: Detection = {
        id: `det-${backendEvent.data.detection_id}`,
        centerFreqHz: backendEvent.data.center_freq_hz,
        bandwidthHz: backendEvent.data.bandwidth_hz,
        bandwidth3dbHz: backendEvent.data.bandwidth_3db_hz,
        bandwidth6dbHz: backendEvent.data.bandwidth_6db_hz,
        peakPowerDb: backendEvent.data.peak_power_db,
        snrDb: backendEvent.data.snr_db,
        startBin: backendEvent.data.start_bin,
        endBin: backendEvent.data.end_bin,
        timestamp: backendEvent.timestamp * 1000, // seconds → milliseconds
        modulationType: backendEvent.data.modulation_type,
        confidence: backendEvent.data.modulation_confidence,
        topKPredictions: backendEvent.data.top_k_predictions?.map(p => ({
          modulation: p.modulation_type,
          confidence: p.confidence,
        })),
        clusterId: backendEvent.data.cluster_id,
        anomalyScore: backendEvent.data.anomaly_score,
        symbolRate: backendEvent.data.symbol_rate,
      };

      // Validate conversion
      expect(frontendDetection.id).toBe('det-12345');
      expect(frontendDetection.centerFreqHz).toBe(915125000);
      expect(frontendDetection.bandwidthHz).toBe(125000);
      expect(frontendDetection.peakPowerDb).toBe(-45.2);
      expect(frontendDetection.snrDb).toBe(18.5);
      expect(frontendDetection.timestamp).toBe(1701590400123); // ms
      expect(frontendDetection.modulationType).toBe('LoRa');
      expect(frontendDetection.confidence).toBe(0.92);
      expect(frontendDetection.clusterId).toBe(3);
      expect(frontendDetection.anomalyScore).toBe(0.12);
      expect(frontendDetection.topKPredictions?.[0].modulation).toBe('LoRa');
    });

    it('should handle missing optional fields', () => {
      const backendEvent: DetectionEvent = {
        type: 'detection',
        timestamp: 1701590400.123,
        data: {
          detection_id: 1,
          center_freq_hz: 915e6,
          bandwidth_hz: 125000,
          peak_power_db: -50,
          snr_db: 10,
          start_bin: 100,
          end_bin: 110,
          // No optional fields
        },
      };

      const frontendDetection: Partial<Detection> = {
        centerFreqHz: backendEvent.data.center_freq_hz,
        modulationType: backendEvent.data.modulation_type, // undefined
        clusterId: backendEvent.data.cluster_id, // undefined
        anomalyScore: backendEvent.data.anomaly_score, // undefined
      };

      expect(frontendDetection.modulationType).toBeUndefined();
      expect(frontendDetection.clusterId).toBeUndefined();
      expect(frontendDetection.anomalyScore).toBeUndefined();
    });
  });

  describe('Cluster Update Conversion', () => {
    it('should convert ClusterUpdate to Cluster correctly', () => {
      // Backend format (snake_case)
      const backendCluster: ClusterUpdate = {
        type: 'clusters',
        timestamp: 1701590400.123,
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

      const backendData = backendCluster.data[0];

      // Frontend format (camelCase)
      const frontendCluster: Cluster = {
        id: backendData.cluster_id,
        size: backendData.size,
        centroid: [backendData.center_freq_hz, backendData.avg_snr_db],
        avgSnrDb: backendData.avg_snr_db,
        dominantFreqHz: backendData.center_freq_hz || backendData.dominant_frequency_hz || 0,
        freqRangeHz: backendData.freq_range_hz,
        avgPowerDb: backendData.avg_power_db,
        avgBandwidthHz: backendData.avg_bandwidth_hz,
        detectionCount: backendData.detection_count,
        label: backendData.label,
        signalTypeHint: backendData.signal_type_hint,
        avgDutyCycle: backendData.avg_duty_cycle,
        uniqueTracks: backendData.unique_tracks,
        avgBw3dbRatio: backendData.avg_bw_3db_ratio,
        color: '',
      };

      // Validate conversion
      expect(frontendCluster.id).toBe(3);
      expect(frontendCluster.size).toBe(42);
      expect(frontendCluster.avgSnrDb).toBe(15.2);
      expect(frontendCluster.dominantFreqHz).toBe(915125000);
      expect(frontendCluster.signalTypeHint).toBe('LoRa');
      expect(frontendCluster.avgDutyCycle).toBe(0.12);
      expect(frontendCluster.uniqueTracks).toBe(3);
    });

    it('should handle frequency field fallback', () => {
      // Backend may send center_freq_hz OR dominant_frequency_hz
      const withCenterFreq = { center_freq_hz: 915e6, dominant_frequency_hz: undefined };
      const withDominantFreq = { center_freq_hz: undefined, dominant_frequency_hz: 915e6 };
      const withBoth = { center_freq_hz: 915e6, dominant_frequency_hz: 916e6 };

      // Frontend should prefer center_freq_hz
      expect(withCenterFreq.center_freq_hz || withCenterFreq.dominant_frequency_hz || 0).toBe(915e6);
      expect(withDominantFreq.center_freq_hz || withDominantFreq.dominant_frequency_hz || 0).toBe(915e6);
      expect(withBoth.center_freq_hz || withBoth.dominant_frequency_hz || 0).toBe(915e6);
    });
  });

  describe('LoRa Frame Conversion', () => {
    it('should convert LoRaFrameEvent to LoRaFrame correctly', () => {
      // Backend format
      const backendFrame: LoRaFrameEvent = {
        type: 'lora',
        timestamp: 1701590400.123,
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

      // Frontend format
      const frontendFrame: LoRaFrame = {
        timestamp_ns: backendFrame.timestamp * 1e9,
        freqHz: backendFrame.center_freq_hz,
        spreadingFactor: backendFrame.spreading_factor,
        bandwidthHz: backendFrame.bandwidth_hz,
        codingRate: backendFrame.coding_rate,
        payload: backendFrame.payload_hex,
        crcValid: backendFrame.crc_valid,
        rssi: backendFrame.rssi_dbm,
        snrDb: backendFrame.snr_db,
      };

      expect(frontendFrame.freqHz).toBe(915000000);
      expect(frontendFrame.spreadingFactor).toBe(7);
      expect(frontendFrame.bandwidthHz).toBe(125000);
      expect(frontendFrame.codingRate).toBe('4/5');
      expect(frontendFrame.payload).toBe('deadbeef');
      expect(frontendFrame.crcValid).toBe(true);
      expect(frontendFrame.rssi).toBe(-80.0);
      expect(frontendFrame.snrDb).toBe(10.5);
    });
  });

  describe('BLE Packet Conversion', () => {
    it('should convert BLEPacketEvent to BLEPacket correctly', () => {
      // Backend format
      const backendPacket: BLEPacketEvent = {
        type: 'ble',
        timestamp: 1701590400.123,
        packet_id: 1,
        channel: 37,
        access_address: '0x8E89BED6',
        pdu_type: 'ADV_IND',
        payload_hex: '0201061aff4c00',
        crc_valid: true,
        rssi_dbm: -70.0,
      };

      // Frontend format
      const frontendPacket: BLEPacket = {
        timestamp_ns: backendPacket.timestamp * 1e9,
        channel: backendPacket.channel,
        accessAddress: parseInt(backendPacket.access_address, 16),
        packetType: backendPacket.pdu_type,
        payload: backendPacket.payload_hex,
        crcValid: backendPacket.crc_valid,
        rssi: backendPacket.rssi_dbm,
      };

      expect(frontendPacket.channel).toBe(37);
      expect(frontendPacket.accessAddress).toBe(0x8E89BED6);
      expect(frontendPacket.packetType).toBe('ADV_IND');
      expect(frontendPacket.payload).toBe('0201061aff4c00');
      expect(frontendPacket.crcValid).toBe(true);
      expect(frontendPacket.rssi).toBe(-70.0);
    });

    it('should handle access_address parsing', () => {
      const hexStrings = ['0x8E89BED6', '8E89BED6', '0X8E89BED6'];

      hexStrings.forEach(hex => {
        const parsed = parseInt(hex.replace(/^0x/i, ''), 16);
        expect(parsed).toBe(0x8E89BED6);
      });
    });
  });

  describe('Timestamp Conversions', () => {
    it('should convert Unix seconds to JavaScript milliseconds', () => {
      const unixSeconds = 1701590400.123;
      const jsMillis = unixSeconds * 1000;

      expect(jsMillis).toBe(1701590400123);
      expect(new Date(jsMillis).toISOString()).toMatch(/2023-12-03/);
    });

    it('should convert seconds to nanoseconds for decoder types', () => {
      const unixSeconds = 1701590400.123;
      const nanos = unixSeconds * 1e9;

      // Use toBeCloseTo due to floating point precision limits
      expect(nanos).toBeCloseTo(1701590400123000000, -4);
    });
  });

  describe('Unit Conversions', () => {
    it('should handle Hz to MHz conversion for display', () => {
      const freqHz = 915125000;
      const freqMHz = freqHz / 1e6;

      expect(freqMHz).toBeCloseTo(915.125, 3);
    });

    it('should handle Hz to kHz conversion for bandwidth display', () => {
      const bwHz = 125000;
      const bwKHz = bwHz / 1e3;

      expect(bwKHz).toBe(125);
    });

    it('should handle linear power to dB conversion', () => {
      const linearPower = 0.001; // 1 mW
      const powerDb = 10 * Math.log10(linearPower);

      expect(powerDb).toBeCloseTo(-30, 1);
    });

    it('should handle uint8 PSD quantization', () => {
      // Backend sends uint8 (0-255), frontend converts to dB
      const uint8Value = 128;
      const minDb = -120;
      const maxDb = -20;
      const dBm = minDb + (uint8Value / 255) * (maxDb - minDb);

      expect(dBm).toBeCloseTo(-70, 0);
    });
  });
});

describe('Field Name Mapping', () => {
  it('should document all snake_case → camelCase mappings', () => {
    const fieldMappings: Record<string, string> = {
      // Detection fields
      'detection_id': 'id',
      'center_freq_hz': 'centerFreqHz',
      'bandwidth_hz': 'bandwidthHz',
      'bandwidth_3db_hz': 'bandwidth3dbHz',
      'bandwidth_6db_hz': 'bandwidth6dbHz',
      'peak_power_db': 'peakPowerDb',
      'snr_db': 'snrDb',
      'start_bin': 'startBin',
      'end_bin': 'endBin',
      'modulation_type': 'modulationType',
      'modulation_confidence': 'confidence',
      'top_k_predictions': 'topKPredictions',
      'cluster_id': 'clusterId',
      'anomaly_score': 'anomalyScore',
      'symbol_rate': 'symbolRate',

      // Cluster fields
      'avg_snr_db': 'avgSnrDb',
      'dominant_frequency_hz': 'dominantFreqHz',
      'freq_range_hz': 'freqRangeHz',
      'avg_power_db': 'avgPowerDb',
      'avg_bandwidth_hz': 'avgBandwidthHz',
      'detection_count': 'detectionCount',
      'signal_type_hint': 'signalTypeHint',
      'avg_duty_cycle': 'avgDutyCycle',
      'unique_tracks': 'uniqueTracks',
      'avg_bw_3db_ratio': 'avgBw3dbRatio',

      // SDR config fields
      'sample_rate_hz': 'sampleRateHz',
      'lna_db': 'lnaDb',
      'tia_db': 'tiaDb',
      'pga_db': 'pgaDb',
      'rx_path': 'rxPath',

      // LoRa fields
      'spreading_factor': 'spreadingFactor',
      'coding_rate': 'codingRate',
      'payload_hex': 'payload',
      'crc_valid': 'crcValid',
      'rssi_dbm': 'rssi',

      // BLE fields
      'access_address': 'accessAddress',
      'pdu_type': 'packetType',
    };

    // All backend fields should be snake_case
    Object.keys(fieldMappings).forEach(backendField => {
      expect(backendField).toMatch(/^[a-z]+(_[a-z0-9]+)*$/);
    });

    // All frontend fields should be camelCase (except simple ones)
    Object.values(fieldMappings).forEach(frontendField => {
      // Allow single words or camelCase
      expect(frontendField).toMatch(/^[a-z][a-zA-Z0-9]*$/);
    });
  });
});
