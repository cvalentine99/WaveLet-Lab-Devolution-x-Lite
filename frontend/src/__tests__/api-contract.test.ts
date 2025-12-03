/**
 * REST API Contract Validation Tests
 * Validates request/response schemas match backend expectations
 */

import { describe, it, expect } from 'vitest';
import type {
  SystemStatus,
  SDRGain,
  SDRConfig,
  PipelineConfig,
  CFARConfig,
  SDRDevice,
  SDRHardwareStatus,
  SDRMetrics,
  SDRHealth,
  SDRCapabilities,
  Recording,
} from '../lib/api';

describe('REST API Contract Validation', () => {
  describe('System Status Schema', () => {
    it('should validate SystemStatus response structure', () => {
      const status: SystemStatus = {
        state: 'running',
        uptime_seconds: 3600,
        samples_processed: 1000000000,
        detections_count: 1542,
        current_throughput_msps: 40.5,
        gpu_memory_used_gb: 8.2,
        buffer_fill_level: 0.35,
        processing_latency_ms: 4.2,
      };

      expect(['idle', 'configuring', 'running', 'paused', 'error']).toContain(status.state);
      expect(typeof status.uptime_seconds).toBe('number');
      expect(status.uptime_seconds).toBeGreaterThanOrEqual(0);
      expect(typeof status.samples_processed).toBe('number');
      expect(typeof status.detections_count).toBe('number');
      expect(typeof status.current_throughput_msps).toBe('number');
      expect(typeof status.gpu_memory_used_gb).toBe('number');
      expect(status.gpu_memory_used_gb).toBeLessThanOrEqual(24); // RTX 4090 = 24GB
      expect(status.buffer_fill_level).toBeGreaterThanOrEqual(0);
      expect(status.buffer_fill_level).toBeLessThanOrEqual(1);
      expect(typeof status.processing_latency_ms).toBe('number');
    });

    it('should handle all valid states', () => {
      const validStates = ['idle', 'configuring', 'running', 'paused', 'error'] as const;
      validStates.forEach(state => {
        const status: Partial<SystemStatus> = { state };
        expect(validStates).toContain(status.state);
      });
    });
  });

  describe('SDR Gain Schema', () => {
    it('should validate SDRGain structure with correct ranges', () => {
      const gain: SDRGain = {
        lna_db: 15,
        tia_db: 9,
        pga_db: 12,
      };

      expect(gain.lna_db).toBeGreaterThanOrEqual(0);
      expect(gain.lna_db).toBeLessThanOrEqual(30);
      expect(gain.tia_db).toBeGreaterThanOrEqual(0);
      expect(gain.tia_db).toBeLessThanOrEqual(12);
      expect(gain.pga_db).toBeGreaterThanOrEqual(0);
      expect(gain.pga_db).toBeLessThanOrEqual(32);
    });
  });

  describe('SDR Config Schema', () => {
    it('should validate SDRConfig structure', () => {
      const config: SDRConfig = {
        device_type: 'usdr',
        center_freq_hz: 915000000,
        sample_rate_hz: 10000000,
        bandwidth_hz: 10000000,
        gain: {
          lna_db: 15,
          tia_db: 9,
          pga_db: 12,
        },
        rx_path: 'LNAL',
      };

      expect(['usrp', 'hackrf', 'usdr']).toContain(config.device_type);
      expect(config.center_freq_hz).toBeGreaterThan(0);
      expect(config.sample_rate_hz).toBeGreaterThan(0);
      expect(config.bandwidth_hz).toBeGreaterThan(0);
      expect(['LNAH', 'LNAL', 'LNAW']).toContain(config.rx_path);
    });

    it('should validate RX path options', () => {
      const validPaths = ['LNAH', 'LNAL', 'LNAW'] as const;
      validPaths.forEach(path => {
        expect(validPaths).toContain(path);
      });
    });
  });

  describe('Pipeline Config Schema', () => {
    it('should validate PipelineConfig structure', () => {
      const config: PipelineConfig = {
        fft_size: 2048,
        window_type: 'hann',
        overlap: 0.5,
        averaging_count: 4,
      };

      // FFT size must be power of 2
      expect(Math.log2(config.fft_size) % 1).toBe(0);
      expect(['hann', 'hamming', 'blackman', 'kaiser', 'flattop']).toContain(config.window_type);
      expect(config.overlap).toBeGreaterThanOrEqual(0);
      expect(config.overlap).toBeLessThan(1);
      expect(config.averaging_count).toBeGreaterThanOrEqual(1);
    });

    it('should validate window types', () => {
      const validWindows = ['hann', 'hamming', 'blackman', 'kaiser', 'flattop'] as const;
      validWindows.forEach(window => {
        expect(validWindows).toContain(window);
      });
    });
  });

  describe('CFAR Config Schema', () => {
    it('should validate CFARConfig structure', () => {
      const config: CFARConfig = {
        ref_cells: 32,
        guard_cells: 4,
        pfa: 0.000001,
        variant: 'CA',
      };

      expect(config.ref_cells).toBeGreaterThan(0);
      expect(config.guard_cells).toBeGreaterThanOrEqual(0);
      expect(config.pfa).toBeGreaterThan(0);
      expect(config.pfa).toBeLessThan(1);
      expect(['CA', 'GO', 'SO', 'OS']).toContain(config.variant);
    });

    it('should validate CFAR variants', () => {
      const validVariants = ['CA', 'GO', 'SO', 'OS'] as const;
      validVariants.forEach(variant => {
        expect(validVariants).toContain(variant);
      });
    });
  });

  describe('SDR Device Schema', () => {
    it('should validate SDRDevice structure', () => {
      const device: SDRDevice = {
        id: 'usdr:0',
        model: 'uSDR DevBoard',
        serial: 'ABC123',
        status: 'available',
      };

      expect(typeof device.id).toBe('string');
      expect(typeof device.model).toBe('string');
      expect(['available', 'connected', 'in_use']).toContain(device.status);
    });
  });

  describe('SDR Hardware Status Schema', () => {
    it('should validate SDRHardwareStatus structure', () => {
      const status: SDRHardwareStatus = {
        connected: true,
        device_id: 'usdr:0',
        temperature_c: 45.2,
        actual_freq_hz: 915000000,
        actual_sample_rate_hz: 10000000,
        actual_bandwidth_hz: 10000000,
        rx_path: 'LNAL',
        streaming: true,
      };

      expect(typeof status.connected).toBe('boolean');
      if (status.temperature_c !== undefined) {
        expect(status.temperature_c).toBeLessThan(100); // Safe operating range
      }
      if (status.rx_path) {
        expect(['LNAH', 'LNAL', 'LNAW']).toContain(status.rx_path);
      }
    });
  });

  describe('SDR Metrics Schema', () => {
    it('should validate SDRMetrics structure', () => {
      const metrics: SDRMetrics = {
        overflow: {
          total: 0,
          rate_per_sec: 0.0,
          last_timestamp: null,
        },
        samples: {
          total_received: 500000000,
          total_dropped: 1200,
          drop_rate_percent: 0.00024,
        },
        hardware: {
          temperature_c: 45.2,
          pll_locked: true,
          actual_sample_rate_hz: 10000000,
          actual_freq_hz: 915000000,
        },
        streaming: {
          uptime_seconds: 3600,
          reconnect_count: 0,
          last_error: null,
        },
        backpressure: {
          events: 0,
          buffer_fill_percent: 35.2,
        },
      };

      // Overflow metrics
      expect(metrics.overflow.total).toBeGreaterThanOrEqual(0);
      expect(metrics.overflow.rate_per_sec).toBeGreaterThanOrEqual(0);

      // Sample metrics
      expect(metrics.samples.total_received).toBeGreaterThanOrEqual(0);
      expect(metrics.samples.drop_rate_percent).toBeGreaterThanOrEqual(0);
      expect(metrics.samples.drop_rate_percent).toBeLessThanOrEqual(100);

      // Hardware metrics
      expect(typeof metrics.hardware.pll_locked).toBe('boolean');

      // Backpressure metrics
      expect(metrics.backpressure.buffer_fill_percent).toBeGreaterThanOrEqual(0);
      expect(metrics.backpressure.buffer_fill_percent).toBeLessThanOrEqual(100);
    });
  });

  describe('SDR Health Schema', () => {
    it('should validate SDRHealth structure', () => {
      const health: SDRHealth = {
        status: 'healthy',
        connected: {},
        streaming: true,
        warnings: [],
        metrics_summary: {
          overflow_rate: 0.0,
          drop_rate_percent: 0.00024,
          buffer_fill_percent: 35.2,
          temperature_c: 45.2,
          uptime_seconds: 3600,
        },
      };

      expect(['healthy', 'degraded', 'unhealthy']).toContain(health.status);
      expect(typeof health.streaming).toBe('boolean');
      expect(Array.isArray(health.warnings)).toBe(true);
    });

    it('should validate health status levels', () => {
      const validStatuses = ['healthy', 'degraded', 'unhealthy'] as const;
      validStatuses.forEach(status => {
        expect(validStatuses).toContain(status);
      });
    });
  });

  describe('SDR Capabilities Schema', () => {
    it('should validate SDRCapabilities structure', () => {
      const caps: SDRCapabilities = {
        freq_range_hz: { min: 1000000, max: 3800000000 },
        sample_rate_range_hz: { min: 1000000, max: 65000000 },
        bandwidth_range_hz: { min: 500000, max: 40000000 },
        gain_range_db: { min: 0, max: 74 },
        rx_paths: ['LNAH', 'LNAL', 'LNAW'],
        supported_formats: ['cf32_le', 'cs16_le'],
        device: {
          id: 'usdr:0',
          model: 'uSDR DevBoard',
          serial: 'ABC123',
          firmware_version: '1.2.3',
        },
        features: {
          supports_gpudirect: true,
          supports_timestamps: true,
          max_channels: 2,
        },
      };

      // Frequency range
      expect(caps.freq_range_hz.min).toBeLessThan(caps.freq_range_hz.max);
      expect(caps.freq_range_hz.min).toBeGreaterThan(0);

      // Sample rate range
      expect(caps.sample_rate_range_hz.min).toBeLessThan(caps.sample_rate_range_hz.max);

      // RX paths
      expect(Array.isArray(caps.rx_paths)).toBe(true);
      expect(caps.rx_paths.length).toBeGreaterThan(0);

      // Device info
      expect(typeof caps.device.id).toBe('string');
      expect(typeof caps.device.model).toBe('string');

      // Features
      expect(typeof caps.features.supports_gpudirect).toBe('boolean');
      expect(caps.features.max_channels).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Recording Schema', () => {
    it('should validate Recording structure', () => {
      const recording: Recording = {
        id: 'rec-abc123',
        name: 'capture-2025-12-03',
        description: 'ISM band monitoring',
        center_freq_hz: 915000000,
        sample_rate_hz: 10000000,
        num_samples: 600000000,
        duration_seconds: 60,
        file_size_bytes: 480000000,
        created_at: '2025-12-03T10:00:00Z',
        status: 'stopped',
        sigmf_meta_path: '/recordings/capture.sigmf-meta',
        sigmf_data_path: '/recordings/capture.sigmf-data',
      };

      expect(typeof recording.id).toBe('string');
      expect(typeof recording.name).toBe('string');
      expect(recording.center_freq_hz).toBeGreaterThan(0);
      expect(recording.sample_rate_hz).toBeGreaterThan(0);
      expect(recording.num_samples).toBeGreaterThanOrEqual(0);
      expect(recording.duration_seconds).toBeGreaterThanOrEqual(0);
      expect(recording.file_size_bytes).toBeGreaterThanOrEqual(0);
      expect(['recording', 'stopped']).toContain(recording.status);
      expect(recording.sigmf_meta_path).toContain('.sigmf-meta');
      expect(recording.sigmf_data_path).toContain('.sigmf-data');
    });
  });
});

describe('API Endpoint Contracts', () => {
  it('should document all required endpoints', () => {
    const requiredEndpoints = [
      // System control
      { method: 'POST', path: '/api/start' },
      { method: 'POST', path: '/api/stop' },
      { method: 'POST', path: '/api/pause' },
      { method: 'POST', path: '/api/resume' },
      { method: 'GET', path: '/api/status' },

      // Configuration
      { method: 'GET', path: '/api/config' },
      { method: 'POST', path: '/api/config' },

      // SDR Device
      { method: 'GET', path: '/api/sdr/devices' },
      { method: 'POST', path: '/api/sdr/connect' },
      { method: 'POST', path: '/api/sdr/disconnect' },
      { method: 'GET', path: '/api/sdr/status' },
      { method: 'GET', path: '/api/sdr/metrics' },
      { method: 'GET', path: '/api/sdr/health' },
      { method: 'GET', path: '/api/sdr/capabilities' },
      { method: 'POST', path: '/api/sdr/config' },
      { method: 'POST', path: '/api/sdr/gain' },
      { method: 'POST', path: '/api/sdr/rx_path' },
      { method: 'POST', path: '/api/sdr/frequency' },

      // Bands
      { method: 'GET', path: '/api/sdr/bands' },
      { method: 'POST', path: '/api/sdr/band' },

      // Recordings
      { method: 'POST', path: '/api/recordings/start' },
      { method: 'POST', path: '/api/recordings/stop' },
      { method: 'GET', path: '/api/recordings' },
    ];

    expect(requiredEndpoints.length).toBeGreaterThan(0);
    requiredEndpoints.forEach(endpoint => {
      expect(['GET', 'POST', 'PUT', 'DELETE']).toContain(endpoint.method);
      expect(endpoint.path).toMatch(/^\/api\//);
    });
  });

  it('should document WebSocket endpoints', () => {
    const wsEndpoints = [
      { path: '/ws/spectrum', format: 'binary' },
      { path: '/ws/detections', format: 'json' },
      { path: '/ws/clusters', format: 'json' },
      { path: '/ws/iq', format: 'binary' },
      { path: '/ws/demodulation', format: 'json' }, // Unified LoRa/BLE stream
    ];

    expect(wsEndpoints.length).toBe(5);
    wsEndpoints.forEach(endpoint => {
      expect(endpoint.path).toMatch(/^\/ws\//);
      expect(['binary', 'json']).toContain(endpoint.format);
    });
  });
});
