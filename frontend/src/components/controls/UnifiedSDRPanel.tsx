import { useState, useEffect, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import {
  Pause, Play, Square, Circle, Radio, Thermometer, RefreshCw,
  ChevronDown, ChevronRight, Antenna,
  Signal, Sliders, Clock
} from 'lucide-react';
import { api, type Config, type SDRDevice, type SDRHardwareStatus, type SDRGain, type DevBoardConfig } from '@/lib/api';
import { FREQUENCY_PRESETS, getPresetById } from '@/lib/frequencyPresets';
import { toast } from 'sonner';
import { BandSelectionDialog } from './BandSelectionDialog';
import { SDRMetricsBar } from './SDRMetricsBar';
import { useSDR } from '@/hooks/useSDR';

export interface UnifiedSDRPanelProps {
  onStart?: () => void;
  onStop?: () => void;
  onPause?: () => void;
  isRunning?: boolean;
  selectedDeviceId?: string;
  onSelectDevice?: (id: string) => void;
}

/**
 * Unified SDR Control Panel - Organized by Signal Flow
 *
 * Signal Path: Antenna → Band/Duplexer → Ext LNA → Atten → LMS7002M → ADC → GPU
 *
 * Sections:
 * 1. TUNING - Frequency, sample rate, bandwidth (most used)
 * 2. RF FRONTEND - DevBoard analog controls (band, ext LNA/PA, attenuator)
 * 3. RECEIVER - LMS7002M chip controls (RX path, internal gains)
 * 4. CLOCK & PERIPHERALS - VCTCXO, GPS, OSC, loopback, UART (rarely used)
 */
export function UnifiedSDRPanel({
  onStart,
  onStop,
  onPause,
  isRunning = false,
  selectedDeviceId: controlledSelectedId,
  onSelectDevice,
}: UnifiedSDRPanelProps) {
  // Config state
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(false);
  const [isPaused, setIsPaused] = useState(false);
  const [isRecording, setIsRecording] = useState(false);

  // Device state
  const [devices, setDevices] = useState<SDRDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>(controlledSelectedId || '');
  const [refreshing, setRefreshing] = useState(false);

  // Use SDR hook for real-time metrics
  const {
    status: hardwareStatus,
    health,
    capabilities,
    isConnected,
    isStreaming,
    dropRatePercent,
    bufferFillPercent,
    temperatureC,
    hasWarnings,
    warnings,
    refresh: refreshSDR,
  } = useSDR({ pollInterval: 1000, pollOnlyWhenConnected: false });

  // LMS7002M Gain controls
  const [lnaGain, setLnaGain] = useState(15);  // 0-30 dB
  const [tiaGain, setTiaGain] = useState(9);   // 0-12 dB
  const [pgaGain, setPgaGain] = useState(12);  // 0-32 dB
  const [rxPath, setRxPath] = useState<'LNAH' | 'LNAL' | 'LNAW'>('LNAL');

  // DevBoard RF Frontend controls
  const [lnaEnable, setLnaEnable] = useState(true);    // lna_: +19.5 dB RX LNA
  const [paEnable, setPaEnable] = useState(false);     // pa_: +19.5 dB TX PA
  const [attenuatorDb, setAttenuatorDb] = useState(0); // attn_: 0-18 dB

  // DevBoard Clock & Peripherals
  const [vctcxoDac, setVctcxoDac] = useState(32768);   // dac_: 0-65535
  const [gpsEnable, setGpsEnable] = useState(false);   // gps_
  const [oscEnable, setOscEnable] = useState(true);    // osc_
  const [loopbackEnable, setLoopbackEnable] = useState(false); // lb_
  const [uartEnable, setUartEnable] = useState(false); // uart_

  // Section visibility
  const [tuningOpen, setTuningOpen] = useState(true);
  const [frontendOpen, setFrontendOpen] = useState(true);
  const [receiverOpen, setReceiverOpen] = useState(true);
  const [peripheralsOpen, setPeripheralsOpen] = useState(false);

  // Band dialog
  const [bandDialogOpen, setBandDialogOpen] = useState(false);
  const [currentBand, setCurrentBand] = useState<string>('');

  // Computed values
  const totalGain = lnaGain + tiaGain + pgaGain + (lnaEnable ? 19.5 : 0) - attenuatorDb;
  const vctcxoOffset = Math.round((vctcxoDac - 32768) * 0.0084); // ~0.0084 Hz per step

  // Load config and devices on mount
  useEffect(() => {
    loadConfig();
    loadDevices();
  }, []);

  // Auto-select uSDR device when devices load
  useEffect(() => {
    if (devices.length > 0 && !selectedDeviceId) {
      // Prefer uSDR device, otherwise select first available
      const usdDevice = devices.find(d =>
        d.model?.toLowerCase().includes('usdr') ||
        d.id?.toLowerCase().includes('usdr')
      );
      const deviceToSelect = usdDevice || devices[0];
      setSelectedDeviceId(deviceToSelect.id);
      onSelectDevice?.(deviceToSelect.id);
    }
  }, [devices, selectedDeviceId, onSelectDevice]);

  // Sync controlled selected device
  useEffect(() => {
    if (controlledSelectedId !== undefined) {
      setSelectedDeviceId(controlledSelectedId);
    }
  }, [controlledSelectedId]);

  // Sync state from hardware when connected
  useEffect(() => {
    if (hardwareStatus?.rx_path) {
      setRxPath(hardwareStatus.rx_path as 'LNAH' | 'LNAL' | 'LNAW');
    }
  }, [hardwareStatus?.rx_path]);

  const loadConfig = async () => {
    try {
      const cfg = await api.getConfig();
      setConfig(cfg);
      // Sync local state from config
      if (cfg.sdr.gain) {
        setLnaGain(cfg.sdr.gain.lna_db || 15);
        setTiaGain(cfg.sdr.gain.tia_db || 9);
        setPgaGain(cfg.sdr.gain.pga_db || 12);
      }
    } catch (error) {
      console.error('Failed to load config:', error);
    }
  };

  const loadDevices = async () => {
    setRefreshing(true);
    try {
      const deviceList = await api.getSDRDevices();
      setDevices(deviceList);
      if (deviceList.length === 0) {
        toast.info('No SDR devices found. Check USB connection.');
      } else if (selectedDeviceId && !deviceList.find(d => d.id === selectedDeviceId)) {
        setSelectedDeviceId('');
        onSelectDevice?.('');
      }
    } catch (error: any) {
      console.error('Failed to load devices:', error);
      toast.error('Failed to scan for SDR devices');
      setDevices([]);
    } finally {
      setRefreshing(false);
    }
  };

  const handleSelectDevice = (deviceId: string) => {
    setSelectedDeviceId(deviceId);
    onSelectDevice?.(deviceId);
  };

  const handlePause = async () => {
    setLoading(true);
    try {
      await api.pause();
      setIsPaused(!isPaused);
      onPause?.();
      toast.success(isPaused ? 'Resumed' : 'Paused');
    } catch (error) {
      toast.error('Failed to pause');
    } finally {
      setLoading(false);
    }
  };

  const handleConnect = async () => {
    if (!selectedDeviceId) {
      toast.error('No device selected. Please select a device first.');
      return;
    }
    if (devices.length === 0) {
      toast.error('No SDR devices found. Click refresh to scan for devices.');
      return;
    }
    setLoading(true);
    try {
      await api.connectSDR(selectedDeviceId);
      toast.success('Connected to SDR');
      await loadDevices();
      await refreshSDR();
    } catch (error: any) {
      const msg = error?.message || 'Failed to connect to SDR';
      toast.error(msg);
      console.error('SDR connection error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDisconnect = async () => {
    setLoading(true);
    try {
      await api.disconnectSDR();
      toast.success('Disconnected');
      await loadDevices();
    } catch (error) {
      toast.error('Failed to disconnect');
    } finally {
      setLoading(false);
    }
  };

  // Start streaming - auto-connect if needed, then start
  const handleStartStream = async () => {
    const deviceId = selectedDeviceId || devices[0]?.id;
    if (!deviceId) {
      toast.error('No SDR device found');
      return;
    }

    setLoading(true);
    try {
      // Auto-connect if not already connected
      if (!isConnected) {
        await api.connectSDR(deviceId);
        await refreshSDR();
      }
      // Call the parent's onStart to start streaming
      onStart?.();
    } catch (error: any) {
      const msg = error?.message || 'Failed to start streaming';
      toast.error(msg);
      console.error('Start stream error:', error);
    } finally {
      setLoading(false);
    }
  };

  // Stop streaming
  const handleStopStream = async () => {
    setLoading(true);
    try {
      // Stop recording if active
      if (isRecording) {
        setIsRecording(false);
      }
      // Call parent's onStop
      onStop?.();
    } catch (error) {
      toast.error('Failed to stop streaming');
    } finally {
      setLoading(false);
    }
  };

  // Toggle recording
  const handleRecord = async () => {
    if (!isRunning) {
      toast.error('Start streaming first before recording');
      return;
    }

    setIsRecording(!isRecording);
    if (!isRecording) {
      toast.success('Recording started');
    } else {
      toast.success('Recording stopped');
    }
  };

  const handleApplyTuning = async () => {
    if (!config) return;
    setLoading(true);
    try {
      // Use dedicated frequency endpoint for efficiency
      await api.setSDRFrequency(config.sdr.center_freq_hz);

      // If sample rate or bandwidth also changed, update via config (REST API uses camelCase)
      const currentConfig = await api.getDirectSDRConfig();
      if (currentConfig.sampleRateHz !== config.sdr.sample_rate_hz ||
          currentConfig.bandwidthHz !== config.sdr.bandwidth_hz) {
        await api.updateDirectSDRConfig({
          sampleRateHz: config.sdr.sample_rate_hz,
          bandwidthHz: config.sdr.bandwidth_hz,
        });
      }

      toast.success('Tuning applied');
      await refreshSDR();
    } catch (error) {
      toast.error('Failed to apply tuning');
    } finally {
      setLoading(false);
    }
  };

  const handleApplyGains = async () => {
    setLoading(true);
    try {
      await api.setSDRGain({ lna_db: lnaGain, tia_db: tiaGain, pga_db: pgaGain });
      await api.setSDRRxPath(rxPath);
      toast.success('Receiver settings applied');
      await refreshSDR();
    } catch (error) {
      toast.error('Failed to apply gains');
    } finally {
      setLoading(false);
    }
  };

  const handleApplyFrontend = async () => {
    setLoading(true);
    try {
      const devboard: DevBoardConfig = {
        lna_enable: lnaEnable,
        pa_enable: paEnable,
        attenuator_db: attenuatorDb,
        vctcxo_dac: vctcxoDac,
        gps_enable: gpsEnable,
        osc_enable: oscEnable,
        loopback_enable: loopbackEnable,
        uart_enable: uartEnable,
      };
      await api.updateConfig({ sdr: { devboard } as any });
      toast.success('Frontend settings applied');
    } catch (error) {
      toast.error('Failed to apply frontend settings');
    } finally {
      setLoading(false);
    }
  };

  const handlePresetChange = (presetId: string) => {
    const preset = getPresetById(presetId);
    if (preset && config) {
      setConfig({
        ...config,
        sdr: {
          ...config.sdr,
          center_freq_hz: preset.centerFreqHz,
          sample_rate_hz: preset.sampleRateHz || config.sdr.sample_rate_hz,
          bandwidth_hz: preset.bandwidth || config.sdr.bandwidth_hz,
        }
      });
      // Auto-select RX path based on frequency
      setRxPath(preset.centerFreqHz > 1.5e9 ? 'LNAH' : 'LNAL');
    }
  };

  const formatFreqMHz = (hz: number) => (hz / 1e6).toFixed(3);
  const formatRateMsps = (hz: number) => (hz / 1e6).toFixed(2);

  return (
    <TooltipProvider>
      <Card className="p-3">
        <div className="space-y-3">
          {/* ═══════════════════════════════════════════════════════════════════
              HEADER - Device status and Start/Stop controls
              ═══════════════════════════════════════════════════════════════════ */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Radio className="w-4 h-4 text-primary" />
              <h3 className="text-sm font-semibold">uSDR DevBoard</h3>
              {isConnected ? (
                <Badge variant="default" className="text-[10px]">Connected</Badge>
              ) : (
                <Badge variant="outline" className="text-[10px]">Disconnected</Badge>
              )}
              {temperatureC > 0 && (
                <span className="text-[10px] text-muted-foreground flex items-center gap-1">
                  <Thermometer className="w-3 h-3" />
                  {temperatureC.toFixed(0)}°C
                </span>
              )}
            </div>
          </div>

          {/* Real-time Metrics Bar */}
          <SDRMetricsBar
            isConnected={isConnected}
            isStreaming={isStreaming}
            dropRatePercent={dropRatePercent}
            bufferFillPercent={bufferFillPercent}
            temperatureC={temperatureC}
            healthStatus={health?.status}
            warnings={warnings}
          />

          {/* Device Selection */}
          <div className="flex gap-2 items-center">
            <Select value={selectedDeviceId} onValueChange={handleSelectDevice}>
              <SelectTrigger className="h-8 text-xs flex-1">
                <SelectValue placeholder="Select device" />
              </SelectTrigger>
              <SelectContent>
                {devices.map((device) => (
                  <SelectItem key={device.id} value={device.id} className="text-xs">
                    {device.model} {device.status === "connected" && "✓"}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
            <Button variant="ghost" size="sm" onClick={loadDevices} disabled={refreshing} className="h-8 w-8 p-0">
              <RefreshCw className={`w-3 h-3 ${refreshing ? 'animate-spin' : ''}`} />
            </Button>
          </div>

          {/* Primary Controls: Start / Stop / Record */}
          <div className="flex gap-2">
            {!isRunning ? (
              <Button
                size="sm"
                onClick={handleStartStream}
                disabled={loading || devices.length === 0}
                className="flex-1 h-9 bg-green-600 hover:bg-green-700 text-white"
              >
                <Play className="w-4 h-4 mr-2" />
                Start
              </Button>
            ) : (
              <Button
                size="sm"
                onClick={handleStopStream}
                disabled={loading}
                variant="destructive"
                className="flex-1 h-9"
              >
                <Square className="w-4 h-4 mr-2" />
                Stop
              </Button>
            )}
            <Button
              size="sm"
              onClick={handleRecord}
              disabled={loading || !isRunning}
              variant={isRecording ? "destructive" : "outline"}
              className={`h-9 px-4 ${isRecording ? 'animate-pulse' : ''}`}
            >
              <Circle className={`w-4 h-4 ${isRecording ? 'fill-current' : ''}`} />
            </Button>
          </div>

          <Separator />

          {/* ═══════════════════════════════════════════════════════════════════
              TUNING - Frequency, Sample Rate, Bandwidth (most used controls)
              ═══════════════════════════════════════════════════════════════════ */}
          <Collapsible open={tuningOpen} onOpenChange={setTuningOpen}>
            <CollapsibleTrigger className="flex items-center gap-2 w-full text-left py-1 hover:bg-muted/50 rounded px-1 -mx-1">
              {tuningOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <Signal className="w-3 h-3 text-blue-500" />
              <span className="text-xs font-medium">Tuning</span>
              <span className="text-[10px] text-muted-foreground ml-auto font-mono">
                {config ? formatFreqMHz(config.sdr.center_freq_hz) : '---'} MHz
              </span>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2 space-y-3">
              {/* Preset Selector */}
              <div className="space-y-1">
                <Label className="text-[10px] text-muted-foreground">Frequency Preset</Label>
                <Select onValueChange={handlePresetChange}>
                  <SelectTrigger className="h-7 text-xs">
                    <SelectValue placeholder="Select preset..." />
                  </SelectTrigger>
                  <SelectContent>
                    {FREQUENCY_PRESETS.map((preset) => (
                      <SelectItem key={preset.id} value={preset.id} className="text-xs">
                        {preset.name} ({formatFreqMHz(preset.centerFreqHz)} MHz)
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Direct Frequency Input */}
              <div className="space-y-1">
                <Label className="text-[10px] text-muted-foreground">Direct Frequency Entry</Label>
                <div className="flex gap-1">
                  <input
                    type="text"
                    placeholder="915.000"
                    className="flex-1 h-7 px-2 text-xs font-mono bg-background border rounded text-center"
                    defaultValue={config ? (config.sdr.center_freq_hz / 1e6).toFixed(3) : ''}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && config) {
                        const value = parseFloat((e.target as HTMLInputElement).value);
                        if (!isNaN(value) && value > 0 && value < 6000) {
                          setConfig({
                            ...config,
                            sdr: { ...config.sdr, center_freq_hz: value * 1e6 }
                          });
                          setRxPath(value > 1500 ? 'LNAH' : 'LNAL');
                          toast.success(`Frequency set to ${value.toFixed(3)} MHz`);
                        } else {
                          toast.error('Invalid frequency (0-6000 MHz)');
                        }
                      }
                    }}
                    onBlur={(e) => {
                      if (config) {
                        const value = parseFloat(e.target.value);
                        if (!isNaN(value) && value > 0 && value < 6000) {
                          setConfig({
                            ...config,
                            sdr: { ...config.sdr, center_freq_hz: value * 1e6 }
                          });
                        }
                      }
                    }}
                  />
                  <span className="text-xs text-muted-foreground self-center">MHz</span>
                </div>
              </div>

              {/* Frequency Controls - Dropdown Selectors */}
              <div className="grid grid-cols-3 gap-2">
                <div className="space-y-1">
                  <Label className="text-[10px] text-muted-foreground">Center (MHz)</Label>
                  <Select
                    value={config ? String(config.sdr.center_freq_hz) : ''}
                    onValueChange={(v) => config && setConfig({
                      ...config,
                      sdr: { ...config.sdr, center_freq_hz: parseFloat(v) }
                    })}
                  >
                    <SelectTrigger className="h-7 text-xs font-mono">
                      <SelectValue placeholder="Center" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="100000000">100 MHz</SelectItem>
                      <SelectItem value="315000000">315 MHz</SelectItem>
                      <SelectItem value="433000000">433 MHz</SelectItem>
                      <SelectItem value="868000000">868 MHz</SelectItem>
                      <SelectItem value="915000000">915 MHz</SelectItem>
                      <SelectItem value="1090000000">1090 MHz</SelectItem>
                      <SelectItem value="2400000000">2400 MHz</SelectItem>
                      <SelectItem value="2450000000">2450 MHz</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px] text-muted-foreground">Rate (MSPS)</Label>
                  <Select
                    value={config ? String(config.sdr.sample_rate_hz) : ''}
                    onValueChange={(v) => config && setConfig({
                      ...config,
                      sdr: { ...config.sdr, sample_rate_hz: parseFloat(v) }
                    })}
                  >
                    <SelectTrigger className="h-7 text-xs font-mono">
                      <SelectValue placeholder="Rate" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1000000">1 MSPS</SelectItem>
                      <SelectItem value="2000000">2 MSPS</SelectItem>
                      <SelectItem value="4000000">4 MSPS</SelectItem>
                      <SelectItem value="8000000">8 MSPS</SelectItem>
                      <SelectItem value="10000000">10 MSPS</SelectItem>
                      <SelectItem value="20000000">20 MSPS</SelectItem>
                      <SelectItem value="30720000">30.72 MSPS</SelectItem>
                      <SelectItem value="40000000">40 MSPS</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label className="text-[10px] text-muted-foreground">BW (MHz)</Label>
                  <Select
                    value={config ? String(config.sdr.bandwidth_hz) : ''}
                    onValueChange={(v) => config && setConfig({
                      ...config,
                      sdr: { ...config.sdr, bandwidth_hz: parseFloat(v) }
                    })}
                  >
                    <SelectTrigger className="h-7 text-xs font-mono">
                      <SelectValue placeholder="BW" />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="1000000">1 MHz</SelectItem>
                      <SelectItem value="2000000">2 MHz</SelectItem>
                      <SelectItem value="4000000">4 MHz</SelectItem>
                      <SelectItem value="8000000">8 MHz</SelectItem>
                      <SelectItem value="10000000">10 MHz</SelectItem>
                      <SelectItem value="20000000">20 MHz</SelectItem>
                      <SelectItem value="30000000">30 MHz</SelectItem>
                      <SelectItem value="40000000">40 MHz</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <Button size="sm" onClick={handleApplyTuning} disabled={loading} className="w-full h-7 text-xs">
                Apply Tuning
              </Button>
            </CollapsibleContent>
          </Collapsible>

          <Separator />

          {/* ═══════════════════════════════════════════════════════════════════
              RF FRONTEND - DevBoard analog controls (before LMS7002M)
              Signal: Antenna → [Band Filter] → [Ext LNA +19.5dB] → [Atten] → LMS7002M
              ═══════════════════════════════════════════════════════════════════ */}
          <Collapsible open={frontendOpen} onOpenChange={setFrontendOpen}>
            <CollapsibleTrigger className="flex items-center gap-2 w-full text-left py-1 hover:bg-muted/50 rounded px-1 -mx-1">
              {frontendOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <Antenna className="w-3 h-3 text-green-500" />
              <span className="text-xs font-medium">RF Frontend</span>
              <span className="text-[10px] text-muted-foreground ml-auto">
                {currentBand && <span className="font-mono mr-2">{currentBand}</span>}
                {lnaEnable && <span className="text-green-400">LNA</span>}
                {paEnable && <span className="text-orange-400 ml-1">PA</span>}
                {attenuatorDb > 0 && <span className="ml-1">-{attenuatorDb}dB</span>}
              </span>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2 space-y-3">
              {/* Band/Path Selector */}
              <div className="p-2 rounded-lg border bg-muted/30 space-y-2">
                <div className="flex items-center justify-between">
                  <Label className="text-[10px] text-muted-foreground">Duplexer Band (path_)</Label>
                  <Button variant="outline" size="sm" onClick={() => setBandDialogOpen(true)} className="h-6 text-[10px]">
                    Select Band
                  </Button>
                </div>
                <div className="flex items-center gap-2">
                  <span className="font-mono text-sm">{currentBand || 'Manual mode'}</span>
                  {currentBand && (
                    <Badge variant="outline" className="text-[10px]">
                      {lnaEnable ? 'LNA' : ''} {paEnable ? 'PA' : ''}
                    </Badge>
                  )}
                </div>
              </div>

              {/* LNA / PA / Attenuator - Horizontal layout */}
              <div className="grid grid-cols-3 gap-3">
                {/* External LNA */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className={`p-2 rounded-lg border text-center cursor-pointer transition-colors ${lnaEnable ? 'bg-green-500/10 border-green-500/50' : 'bg-muted/30'}`}
                         onClick={() => setLnaEnable(!lnaEnable)}>
                      <div className="text-[10px] text-muted-foreground mb-1">RX LNA</div>
                      <div className={`text-lg font-bold ${lnaEnable ? 'text-green-400' : 'text-muted-foreground'}`}>
                        {lnaEnable ? '+19.5' : 'OFF'}
                      </div>
                      <div className="text-[10px] text-muted-foreground">dB</div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>External LNA (QPL9547TR7)</p>
                    <p className="text-[10px] text-muted-foreground">lna_on / lna_off</p>
                  </TooltipContent>
                </Tooltip>

                {/* External PA */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className={`p-2 rounded-lg border text-center cursor-pointer transition-colors ${paEnable ? 'bg-orange-500/10 border-orange-500/50' : 'bg-muted/30'}`}
                         onClick={() => setPaEnable(!paEnable)}>
                      <div className="text-[10px] text-muted-foreground mb-1">TX PA</div>
                      <div className={`text-lg font-bold ${paEnable ? 'text-orange-400' : 'text-muted-foreground'}`}>
                        {paEnable ? '+19.5' : 'OFF'}
                      </div>
                      <div className="text-[10px] text-muted-foreground">dB</div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>External PA (QPL9547TR7)</p>
                    <p className="text-[10px] text-muted-foreground">pa_on / pa_off</p>
                  </TooltipContent>
                </Tooltip>

                {/* Attenuator */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <div className={`p-2 rounded-lg border text-center ${attenuatorDb > 0 ? 'bg-red-500/10 border-red-500/50' : 'bg-muted/30'}`}>
                      <div className="text-[10px] text-muted-foreground mb-1">Atten</div>
                      <div className={`text-lg font-bold ${attenuatorDb > 0 ? 'text-red-400' : 'text-muted-foreground'}`}>
                        -{attenuatorDb}
                      </div>
                      <div className="text-[10px] text-muted-foreground">dB</div>
                    </div>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>RX Attenuator (0 to -18 dB)</p>
                    <p className="text-[10px] text-muted-foreground">attn_{attenuatorDb}</p>
                  </TooltipContent>
                </Tooltip>
              </div>

              {/* Attenuator Slider */}
              <div className="space-y-1">
                <Slider
                  min={0} max={18} step={1}
                  value={[attenuatorDb]}
                  onValueChange={([v]) => setAttenuatorDb(v)}
                  className="h-4"
                />
                <div className="flex justify-between text-[10px] text-muted-foreground">
                  <span>0 dB</span>
                  <span>-18 dB</span>
                </div>
              </div>

              <Button size="sm" onClick={handleApplyFrontend} disabled={loading} className="w-full h-7 text-xs">
                Apply Frontend
              </Button>
            </CollapsibleContent>
          </Collapsible>

          <Separator />

          {/* ═══════════════════════════════════════════════════════════════════
              RECEIVER - LMS7002M internal controls
              Signal: → [RX Path] → [LNA 0-30dB] → [TIA 0-12dB] → [PGA 0-32dB] → ADC
              ═══════════════════════════════════════════════════════════════════ */}
          <Collapsible open={receiverOpen} onOpenChange={setReceiverOpen}>
            <CollapsibleTrigger className="flex items-center gap-2 w-full text-left py-1 hover:bg-muted/50 rounded px-1 -mx-1">
              {receiverOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <Sliders className="w-3 h-3 text-purple-500" />
              <span className="text-xs font-medium">Receiver (LMS7002M)</span>
              <span className="text-[10px] text-muted-foreground ml-auto font-mono">
                {totalGain.toFixed(1)} dB total
              </span>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2 space-y-3">
              {/* RX Path Selection */}
              <div className="space-y-2">
                <Label className="text-[10px] text-muted-foreground">RX Input Path</Label>
                <RadioGroup value={rxPath} onValueChange={(v) => setRxPath(v as 'LNAH' | 'LNAL' | 'LNAW')} className="flex gap-2">
                  <div className="flex items-center space-x-1">
                    <RadioGroupItem value="LNAH" id="lnah" className="h-3 w-3" />
                    <Label htmlFor="lnah" className="text-[10px] cursor-pointer">LNAH (&gt;1.5GHz)</Label>
                  </div>
                  <div className="flex items-center space-x-1">
                    <RadioGroupItem value="LNAL" id="lnal" className="h-3 w-3" />
                    <Label htmlFor="lnal" className="text-[10px] cursor-pointer">LNAL (&lt;1.5GHz)</Label>
                  </div>
                  <div className="flex items-center space-x-1">
                    <RadioGroupItem value="LNAW" id="lnaw" className="h-3 w-3" />
                    <Label htmlFor="lnaw" className="text-[10px] cursor-pointer">LNAW (wide)</Label>
                  </div>
                </RadioGroup>
              </div>

              {/* Gain Sliders */}
              <div className="space-y-2">
                {/* LNA Gain */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <Label className="text-[10px] text-muted-foreground">LNA (0-30 dB)</Label>
                    <span className="text-[10px] font-mono text-purple-400">{lnaGain} dB</span>
                  </div>
                  <Slider min={0} max={30} step={1} value={[lnaGain]} onValueChange={([v]) => setLnaGain(v)} className="h-4" />
                </div>

                {/* TIA Gain */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <Label className="text-[10px] text-muted-foreground">TIA (0-12 dB)</Label>
                    <span className="text-[10px] font-mono text-purple-400">{tiaGain} dB</span>
                  </div>
                  <Slider min={0} max={12} step={3} value={[tiaGain]} onValueChange={([v]) => setTiaGain(v)} className="h-4" />
                </div>

                {/* PGA Gain */}
                <div className="space-y-1">
                  <div className="flex justify-between">
                    <Label className="text-[10px] text-muted-foreground">PGA (0-32 dB)</Label>
                    <span className="text-[10px] font-mono text-purple-400">{pgaGain} dB</span>
                  </div>
                  <Slider min={0} max={32} step={1} value={[pgaGain]} onValueChange={([v]) => setPgaGain(v)} className="h-4" />
                </div>
              </div>

              {/* Total Gain Summary */}
              <div className="p-2 rounded bg-muted/50 text-center">
                <div className="text-[10px] text-muted-foreground">Total Gain (incl. DevBoard)</div>
                <div className="text-lg font-mono font-bold text-purple-400">{totalGain.toFixed(1)} dB</div>
                <div className="text-[10px] text-muted-foreground">
                  {lnaEnable ? '+19.5' : '0'} + {lnaGain} + {tiaGain} + {pgaGain} - {attenuatorDb}
                </div>
              </div>

              <Button size="sm" onClick={handleApplyGains} disabled={loading} className="w-full h-7 text-xs">
                Apply Receiver
              </Button>
            </CollapsibleContent>
          </Collapsible>

          <Separator />

          {/* ═══════════════════════════════════════════════════════════════════
              CLOCK & PERIPHERALS - Rarely used controls
              ═══════════════════════════════════════════════════════════════════ */}
          <Collapsible open={peripheralsOpen} onOpenChange={setPeripheralsOpen}>
            <CollapsibleTrigger className="flex items-center gap-2 w-full text-left py-1 hover:bg-muted/50 rounded px-1 -mx-1">
              {peripheralsOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <Clock className="w-3 h-3 text-yellow-500" />
              <span className="text-xs font-medium">Clock & Peripherals</span>
              <span className="text-[10px] text-muted-foreground ml-auto">
                {vctcxoOffset !== 0 && <span className="font-mono">{vctcxoOffset > 0 ? '+' : ''}{vctcxoOffset}Hz</span>}
                {gpsEnable && <span className="ml-1">GPS</span>}
                {loopbackEnable && <span className="ml-1 text-yellow-400">LB</span>}
              </span>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2 space-y-3">
              {/* VCTCXO Fine Tuning */}
              <div className="space-y-2">
                <div className="flex justify-between">
                  <Label className="text-[10px] text-muted-foreground">VCTCXO DAC (dac_)</Label>
                  <span className="text-[10px] font-mono">
                    {vctcxoDac} ({vctcxoOffset > 0 ? '+' : ''}{vctcxoOffset} Hz)
                  </span>
                </div>
                <Slider min={0} max={65535} step={1} value={[vctcxoDac]} onValueChange={([v]) => setVctcxoDac(v)} className="h-4" />
                <div className="flex justify-between text-[10px] text-muted-foreground">
                  <span>-275 Hz</span>
                  <span>0 Hz</span>
                  <span>+275 Hz</span>
                </div>
              </div>

              {/* Peripheral Toggles */}
              <div className="grid grid-cols-2 gap-2">
                <div className={`flex items-center justify-between p-2 rounded-lg border ${oscEnable ? 'bg-green-500/10 border-green-500/30' : 'bg-muted/30'}`}>
                  <div>
                    <div className="text-[10px] font-medium">OSC</div>
                    <div className="text-[9px] text-muted-foreground">Ref Clock</div>
                  </div>
                  <Switch checked={oscEnable} onCheckedChange={setOscEnable} className="scale-75" />
                </div>

                <div className={`flex items-center justify-between p-2 rounded-lg border ${gpsEnable ? 'bg-blue-500/10 border-blue-500/30' : 'bg-muted/30'}`}>
                  <div>
                    <div className="text-[10px] font-medium">GPS</div>
                    <div className="text-[9px] text-muted-foreground">Module</div>
                  </div>
                  <Switch checked={gpsEnable} onCheckedChange={setGpsEnable} className="scale-75" />
                </div>

                <div className={`flex items-center justify-between p-2 rounded-lg border ${loopbackEnable ? 'bg-yellow-500/10 border-yellow-500/30' : 'bg-muted/30'}`}>
                  <div>
                    <div className="text-[10px] font-medium">Loopback</div>
                    <div className="text-[9px] text-muted-foreground">RX→TX</div>
                  </div>
                  <Switch checked={loopbackEnable} onCheckedChange={setLoopbackEnable} className="scale-75" />
                </div>

                <div className={`flex items-center justify-between p-2 rounded-lg border ${uartEnable ? 'bg-purple-500/10 border-purple-500/30' : 'bg-muted/30'}`}>
                  <div>
                    <div className="text-[10px] font-medium">UART</div>
                    <div className="text-[9px] text-muted-foreground">Interface</div>
                  </div>
                  <Switch checked={uartEnable} onCheckedChange={setUartEnable} className="scale-75" />
                </div>
              </div>

              <Button size="sm" onClick={handleApplyFrontend} disabled={loading} className="w-full h-7 text-xs">
                Apply Peripherals
              </Button>
            </CollapsibleContent>
          </Collapsible>
        </div>

        <BandSelectionDialog
          open={bandDialogOpen}
          onClose={() => setBandDialogOpen(false)}
          onBandSelected={(band) => {
            setCurrentBand(band.name);
            if (band.pa_enable !== undefined) setPaEnable(band.pa_enable);
            if (band.lna_enable !== undefined) setLnaEnable(band.lna_enable);
            if (band.freq_range_mhz && config) {
              const centerHz = ((band.freq_range_mhz[0] + band.freq_range_mhz[1]) / 2) * 1e6;
              setConfig({ ...config, sdr: { ...config.sdr, center_freq_hz: centerHz } });
              setRxPath(centerHz > 1.5e9 ? 'LNAH' : 'LNAL');
            }
            setBandDialogOpen(false);
          }}
        />
      </Card>
    </TooltipProvider>
  );
}
