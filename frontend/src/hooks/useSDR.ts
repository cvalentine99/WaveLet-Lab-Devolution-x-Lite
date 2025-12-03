import { useState, useEffect, useCallback, useRef } from 'react';
import { api, type SDRMetrics, type SDRHealth, type SDRCapabilities, type SDRHardwareStatus, type GPSStatus, type SystemHealth } from '@/lib/api';

interface UseSDROptions {
  /** Polling interval in ms (default: 1000) */
  pollInterval?: number;
  /** Enable metrics polling (default: true) */
  enableMetrics?: boolean;
  /** Enable health polling (default: true) */
  enableHealth?: boolean;
  /** Enable GPS status polling (default: true) */
  enableGPS?: boolean;
  /** Enable system health polling (default: true) */
  enableSystemHealth?: boolean;
  /** Only poll when connected */
  pollOnlyWhenConnected?: boolean;
}

interface UseSDRReturn {
  // State
  status: SDRHardwareStatus | null;
  metrics: SDRMetrics | null;
  health: SDRHealth | null;
  capabilities: SDRCapabilities | null;
  gpsStatus: GPSStatus | null;
  systemHealth: SystemHealth | null;
  isConnected: boolean;
  isStreaming: boolean;

  // Derived metrics
  dropRatePercent: number;
  bufferFillPercent: number;
  temperatureC: number;
  hasWarnings: boolean;
  warnings: string[];

  // Additional hardware metrics
  pllLocked: boolean;
  uptimeSeconds: number;
  overflowCount: number;
  reconnectCount: number;

  // Actions
  refresh: () => Promise<void>;
  setFrequency: (freqHz: number) => Promise<void>;
  setGain: (lnaDb: number, tiaDb: number, pgaDb: number) => Promise<void>;
  setRxPath: (path: 'LNAH' | 'LNAL' | 'LNAW') => Promise<void>;

  // Loading state
  loading: boolean;
  error: string | null;
}

export function useSDR(options: UseSDROptions = {}): UseSDRReturn {
  const {
    pollInterval = 1000,
    enableMetrics = true,
    enableHealth = true,
    enableGPS = true,
    enableSystemHealth = true,
    pollOnlyWhenConnected = true,
  } = options;

  const [status, setStatus] = useState<SDRHardwareStatus | null>(null);
  const [metrics, setMetrics] = useState<SDRMetrics | null>(null);
  const [health, setHealth] = useState<SDRHealth | null>(null);
  const [capabilities, setCapabilities] = useState<SDRCapabilities | null>(null);
  const [gpsStatus, setGpsStatus] = useState<GPSStatus | null>(null);
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const systemHealthIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const mountedRef = useRef(true);

  // Derived state
  const isConnected = status?.connected ?? false;
  const isStreaming = status?.streaming ?? false;
  const dropRatePercent = metrics?.samples.drop_rate_percent ?? 0;
  const bufferFillPercent = metrics?.backpressure.buffer_fill_percent ?? 0;
  const temperatureC = metrics?.hardware.temperature_c ?? status?.temperature_c ?? 0;
  const hasWarnings = (health?.warnings?.length ?? 0) > 0;
  const warnings = health?.warnings ?? [];

  // Additional hardware metrics
  const pllLocked = metrics?.hardware.pll_locked ?? false;
  const uptimeSeconds = metrics?.streaming.uptime_seconds ?? 0;
  const overflowCount = metrics?.overflow.total ?? 0;
  const reconnectCount = metrics?.streaming.reconnect_count ?? 0;

  // Fetch all SDR data
  const refresh = useCallback(async () => {
    if (!mountedRef.current) return;

    try {
      const [statusRes, metricsRes, healthRes, gpsRes] = await Promise.allSettled([
        api.getSDRStatus(),
        enableMetrics ? api.getSDRMetrics() : Promise.resolve(null),
        enableHealth ? api.getSDRHealth() : Promise.resolve(null),
        enableGPS ? api.getGPSStatus() : Promise.resolve(null),
      ]);

      if (!mountedRef.current) return;

      if (statusRes.status === 'fulfilled') {
        setStatus(statusRes.value);
      }
      if (metricsRes.status === 'fulfilled' && metricsRes.value) {
        setMetrics(metricsRes.value);
      }
      if (healthRes.status === 'fulfilled' && healthRes.value) {
        setHealth(healthRes.value);
      }
      if (gpsRes.status === 'fulfilled' && gpsRes.value) {
        setGpsStatus(gpsRes.value);
      }
      setError(null);
    } catch (err) {
      if (mountedRef.current) {
        setError(err instanceof Error ? err.message : 'Failed to fetch SDR data');
      }
    }
  }, [enableMetrics, enableHealth, enableGPS]);

  // Fetch system health (slower interval - every 10 seconds)
  const refreshSystemHealth = useCallback(async () => {
    if (!mountedRef.current || !enableSystemHealth) return;

    try {
      const health = await api.getSystemHealth();
      if (mountedRef.current) {
        setSystemHealth(health);
      }
    } catch {
      // System health may not be available
    }
  }, [enableSystemHealth]);

  // Fetch capabilities once on mount (they don't change)
  useEffect(() => {
    mountedRef.current = true;

    const fetchCapabilities = async () => {
      try {
        const caps = await api.getSDRCapabilities();
        if (mountedRef.current) {
          setCapabilities(caps);
        }
      } catch {
        // Capabilities may not be available if not connected
      }
    };

    fetchCapabilities();
    refresh();
    refreshSystemHealth();

    return () => {
      mountedRef.current = false;
    };
  }, []);

  // Setup polling for SDR data (fast - every 1s)
  useEffect(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }

    const shouldPoll = !pollOnlyWhenConnected || isConnected;

    if (shouldPoll && pollInterval > 0) {
      intervalRef.current = setInterval(refresh, pollInterval);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [pollInterval, pollOnlyWhenConnected, isConnected, refresh]);

  // Setup polling for system health (slow - every 10s)
  useEffect(() => {
    if (systemHealthIntervalRef.current) {
      clearInterval(systemHealthIntervalRef.current);
    }

    if (enableSystemHealth) {
      systemHealthIntervalRef.current = setInterval(refreshSystemHealth, 10000);
    }

    return () => {
      if (systemHealthIntervalRef.current) {
        clearInterval(systemHealthIntervalRef.current);
      }
    };
  }, [enableSystemHealth, refreshSystemHealth]);

  // Actions
  const setFrequency = useCallback(async (freqHz: number) => {
    setLoading(true);
    try {
      await api.setSDRFrequency(freqHz);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to set frequency');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [refresh]);

  const setGain = useCallback(async (lnaDb: number, tiaDb: number, pgaDb: number) => {
    setLoading(true);
    try {
      await api.setSDRGain({ lna_db: lnaDb, tia_db: tiaDb, pga_db: pgaDb });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to set gain');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [refresh]);

  const setRxPath = useCallback(async (path: 'LNAH' | 'LNAL' | 'LNAW') => {
    setLoading(true);
    try {
      await api.setSDRRxPath(path);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to set RX path');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [refresh]);

  return {
    status,
    metrics,
    health,
    capabilities,
    gpsStatus,
    systemHealth,
    isConnected,
    isStreaming,
    dropRatePercent,
    bufferFillPercent,
    temperatureC,
    hasWarnings,
    warnings,
    pllLocked,
    uptimeSeconds,
    overflowCount,
    reconnectCount,
    refresh,
    setFrequency,
    setGain,
    setRxPath,
    loading,
    error,
  };
}
