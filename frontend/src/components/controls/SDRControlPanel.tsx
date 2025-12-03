import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Slider } from '@/components/ui/slider';
import { Play, Square, Pause, Settings2, Zap, Radio, ChevronDown, ChevronUp } from 'lucide-react';
import { api, type Config } from '@/lib/api';
import { FREQUENCY_PRESETS, getPresetById, type FrequencyPreset } from '@/lib/frequencyPresets';
import { toast } from 'sonner';

export interface SDRControlPanelProps {
  onStart?: () => void;
  onStop?: () => void;
  onPause?: () => void;
  isRunning?: boolean;
}

/**
 * SDR Control Panel
 * Configuration controls for the GPU RF Forensics backend
 */
export function SDRControlPanel({
  onStart,
  onStop,
  onPause,
  isRunning = false
}: SDRControlPanelProps) {
  const [config, setConfig] = useState<Config | null>(null);
  const [loading, setLoading] = useState(false);
  const [isCollapsed, setIsCollapsed] = useState(() => {
    const saved = localStorage.getItem('sdr-control-collapsed');
    return saved === 'true';
  });

  // Persist collapse state
  useEffect(() => {
    localStorage.setItem('sdr-control-collapsed', isCollapsed.toString());
  }, [isCollapsed]);

  // Load current configuration
  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const cfg = await api.getConfig();
      setConfig(cfg);
    } catch (error) {
      console.error('Failed to load config:', error);
      // Set default config so UI is still usable
      setConfig({
        sdr: {
          device_type: 'usdr',
          center_freq_hz: 915e6,
          sample_rate_hz: 2.4e6,
          bandwidth_hz: 2e6,
          gain: { lna_db: 20, tia_db: 12, pga_db: 8 },
          rx_path: 'LNAW'
        },
        pipeline: {
          fft_size: 2048,
          window_type: 'hann',
          overlap: 0.5,
          averaging_count: 4
        },
        cfar: {
          ref_cells: 16,
          guard_cells: 4,
          pfa: 1e-6,
          variant: 'CA'
        }
      });
      toast.error('Failed to load configuration - using defaults');
    }
  };

  const handleStart = async () => {
    setLoading(true);
    try {
      await api.start();
      toast.success('Processing started');
      onStart?.();
    } catch (error) {
      console.error('Failed to start:', error);
      toast.error('Failed to start processing');
    } finally {
      setLoading(false);
    }
  };

  const handleStop = async () => {
    setLoading(true);
    try {
      await api.stop();
      toast.success('Processing stopped');
      onStop?.();
    } catch (error) {
      console.error('Failed to stop:', error);
      toast.error('Failed to stop processing');
    } finally {
      setLoading(false);
    }
  };

  const handlePause = async () => {
    setLoading(true);
    try {
      await api.pause();
      toast.success('Processing paused');
      onPause?.();
    } catch (error) {
      console.error('Failed to pause:', error);
      toast.error('Failed to pause processing');
    } finally {
      setLoading(false);
    }
  };

  const handleApplyConfig = async () => {
    if (!config) return;

    setLoading(true);
    try {
      await api.updateConfig(config);
      toast.success('Configuration updated');
    } catch (error) {
      console.error('Failed to update config:', error);
      toast.error('Failed to update configuration');
    } finally {
      setLoading(false);
    }
  };

  if (!config) {
    return (
      <Card className="p-4">
        <div className="text-center py-8 text-muted-foreground">
          Loading configuration...
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-4">
      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Settings2 className="w-4 h-4 text-primary" />
            <h3 className="text-sm font-semibold">SDR Configuration</h3>
            <Button
              size="sm"
              variant="ghost"
              className="h-6 w-6 p-0 ml-1"
              onClick={() => setIsCollapsed(!isCollapsed)}
            >
              {isCollapsed ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />}
            </Button>
          </div>
          <div className="flex items-center gap-2">
            {!isRunning ? (
              <Button
                size="sm"
                onClick={handleStart}
                disabled={loading}
                className="gap-2"
              >
                <Play className="w-3 h-3" />
                Start
              </Button>
            ) : (
              <>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handlePause}
                  disabled={loading}
                  className="gap-2"
                >
                  <Pause className="w-3 h-3" />
                  Pause
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={handleStop}
                  disabled={loading}
                  className="gap-2"
                >
                  <Square className="w-3 h-3" />
                  Stop
                </Button>
              </>
            )}
          </div>
        </div>

        {/* SDR Settings */}
        {!isCollapsed && config && (
        <div className="space-y-4">
          {/* Frequency Preset Dropdown */}
          <div className="space-y-2">
            <Label htmlFor="freq-preset" className="text-xs flex items-center gap-2">
              <Radio className="w-3 h-3 text-primary" />
              Frequency Preset
            </Label>
            <Select
              value=""
              onValueChange={(value) => {
                const preset = getPresetById(value);
                if (preset) {
                  setConfig({
                    ...config,
                    sdr: {
                      ...config.sdr,
                      center_freq_hz: preset.centerFreqHz,
                      sample_rate_hz: preset.sampleRateHz,
                    }
                  });
                  toast.success(`Preset applied: ${preset.name}`);
                }
              }}
            >
              <SelectTrigger id="freq-preset" className="h-8 text-xs">
                <SelectValue placeholder="Select a preset..." />
              </SelectTrigger>
              <SelectContent className="max-h-[400px]">
                {/* ISM Bands */}
                <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground">ISM Bands</div>
                {FREQUENCY_PRESETS.filter(p => p.category === 'ISM').map(preset => (
                  <SelectItem key={preset.id} value={preset.id} className="text-xs">
                    <div className="flex flex-col">
                      <span className="font-medium">{preset.name}</span>
                      <span className="text-[10px] text-muted-foreground">{preset.description}</span>
                    </div>
                  </SelectItem>
                ))}
                
                {/* WiFi */}
                <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mt-2">WiFi</div>
                {FREQUENCY_PRESETS.filter(p => p.category === 'WiFi').map(preset => (
                  <SelectItem key={preset.id} value={preset.id} className="text-xs">
                    <div className="flex flex-col">
                      <span className="font-medium">{preset.name}</span>
                      <span className="text-[10px] text-muted-foreground">{preset.description}</span>
                    </div>
                  </SelectItem>
                ))}
                
                {/* IoT */}
                <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mt-2">IoT Protocols</div>
                {FREQUENCY_PRESETS.filter(p => p.category === 'IoT').map(preset => (
                  <SelectItem key={preset.id} value={preset.id} className="text-xs">
                    <div className="flex flex-col">
                      <span className="font-medium">{preset.name}</span>
                      <span className="text-[10px] text-muted-foreground">{preset.description}</span>
                    </div>
                  </SelectItem>
                ))}
                
                {/* Cellular */}
                <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mt-2">Cellular</div>
                {FREQUENCY_PRESETS.filter(p => p.category === 'Cellular').map(preset => (
                  <SelectItem key={preset.id} value={preset.id} className="text-xs">
                    <div className="flex flex-col">
                      <span className="font-medium">{preset.name}</span>
                      <span className="text-[10px] text-muted-foreground">{preset.description}</span>
                    </div>
                  </SelectItem>
                ))}
                
                {/* Amateur Radio */}
                <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mt-2">Amateur Radio</div>
                {FREQUENCY_PRESETS.filter(p => p.category === 'Amateur').map(preset => (
                  <SelectItem key={preset.id} value={preset.id} className="text-xs">
                    <div className="flex flex-col">
                      <span className="font-medium">{preset.name}</span>
                      <span className="text-[10px] text-muted-foreground">{preset.description}</span>
                    </div>
                  </SelectItem>
                ))}
                
                {/* Satellite */}
                <div className="px-2 py-1.5 text-xs font-semibold text-muted-foreground mt-2">Satellite</div>
                {FREQUENCY_PRESETS.filter(p => p.category === 'Satellite').map(preset => (
                  <SelectItem key={preset.id} value={preset.id} className="text-xs">
                    <div className="flex flex-col">
                      <span className="font-medium">{preset.name}</span>
                      <span className="text-[10px] text-muted-foreground">{preset.description}</span>
                    </div>
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="device-type" className="text-xs">Device Type</Label>
            <Select
              value={config.sdr.device_type}
              onValueChange={(value: "usrp" | "hackrf" | "usdr") =>
                setConfig({
                  ...config,
                  sdr: { ...config.sdr, device_type: value }
                })
              }
            >
              <SelectTrigger id="device-type" className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="usrp">USRP</SelectItem>
                <SelectItem value="hackrf">HackRF</SelectItem>
                <SelectItem value="usdr">uSDR</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="center-freq" className="text-xs">
              Center Frequency (MHz)
            </Label>
            <Input
              id="center-freq"
              type="number"
              className="h-8 text-xs font-mono"
              value={(config.sdr.center_freq_hz / 1e6).toFixed(3)}
              onChange={(e) =>
                setConfig({
                  ...config,
                  sdr: {
                    ...config.sdr,
                    center_freq_hz: parseFloat(e.target.value) * 1e6
                  }
                })
              }
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="sample-rate" className="text-xs">
              Sample Rate (MHz)
            </Label>
            <Input
              id="sample-rate"
              type="number"
              className="h-8 text-xs font-mono"
              value={(config.sdr.sample_rate_hz / 1e6).toFixed(3)}
              onChange={(e) =>
                setConfig({
                  ...config,
                  sdr: {
                    ...config.sdr,
                    sample_rate_hz: parseFloat(e.target.value) * 1e6
                  }
                })
              }
            />
          </div>

          {/* Gain controls moved to Advanced SDR Dialog */}
          <div className="space-y-2">
            <Label className="text-xs">Gain Controls</Label>
            <p className="text-xs text-muted-foreground">
              Use Advanced Settings for LNA/TIA/PGA gain control
            </p>
          </div>
        </div>
        )}

        {/* Pipeline Settings */}
        {!isCollapsed && (
        <div className="space-y-4 pt-4 border-t border-border">
          <div className="flex items-center gap-2">
            <Zap className="w-3 h-3 text-yellow-500" />
            <h4 className="text-xs font-semibold">Processing Pipeline</h4>
          </div>

          <div className="space-y-2">
            <Label htmlFor="fft-size" className="text-xs">FFT Size</Label>
            <Select
              value={config.pipeline.fft_size.toString()}
              onValueChange={(value) =>
                setConfig({
                  ...config,
                  pipeline: { ...config.pipeline, fft_size: parseInt(value) }
                })
              }
            >
              <SelectTrigger id="fft-size" className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="512">512</SelectItem>
                <SelectItem value="1024">1024</SelectItem>
                <SelectItem value="2048">2048</SelectItem>
                <SelectItem value="4096">4096</SelectItem>
                <SelectItem value="8192">8192</SelectItem>
              </SelectContent>
            </Select>
          </div>

          <div className="space-y-2">
            <Label htmlFor="window-type" className="text-xs">Window Type</Label>
            <Select
              value={config.pipeline.window_type}
              onValueChange={(value: "hann" | "hamming" | "blackman" | "kaiser" | "flattop") =>
                setConfig({
                  ...config,
                  pipeline: { ...config.pipeline, window_type: value }
                })
              }
            >
              <SelectTrigger id="window-type" className="h-8 text-xs">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="hann">Hann</SelectItem>
                <SelectItem value="hamming">Hamming</SelectItem>
                <SelectItem value="blackman">Blackman</SelectItem>
                <SelectItem value="kaiser">Kaiser</SelectItem>
                <SelectItem value="flattop">Flat Top</SelectItem>
              </SelectContent>
            </Select>
          </div>
        </div>
        )}

        {/* Loading state when config is not yet loaded */}
        {!isCollapsed && !config && (
          <div className="text-center py-4 text-muted-foreground text-sm">
            Loading configuration...
          </div>
        )}

        {/* Apply Button */}
        {!isCollapsed && config && (
        <Button
          className="w-full gap-2"
          onClick={handleApplyConfig}
          disabled={loading || isRunning}
        >
          <Settings2 className="w-3 h-3" />
          Apply Configuration
        </Button>
        )}
      </div>
    </Card>
  );
}
