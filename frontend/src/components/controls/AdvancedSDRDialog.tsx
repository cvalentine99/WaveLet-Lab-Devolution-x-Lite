import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Label } from '@/components/ui/label';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { RadioGroup, RadioGroupItem } from '@/components/ui/radio-group';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { api, type SDRDevice, type SDRHardwareStatus, type SDRGain, type DevBoardConfig, type Band } from '@/lib/api';
import { toast } from 'sonner';
import { Radio, Cpu, Zap, Thermometer, RefreshCw, AlertTriangle } from 'lucide-react';
import { BandSelectionDialog } from './BandSelectionDialog';
import { validateBand, getRecommendedRxPath } from '@/lib/bandValidation';

export interface AdvancedSDRDialogProps {
  open: boolean;
  onClose: () => void;
}

/**
 * Advanced SDR Settings Dialog
 * Device discovery, connection, multi-gain control, RX path, and DevBoard settings
 */
export function AdvancedSDRDialog({ open, onClose }: AdvancedSDRDialogProps) {
  const [devices, setDevices] = useState<SDRDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');
  const [hardwareStatus, setHardwareStatus] = useState<SDRHardwareStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);

  // Gain controls
  const [lnaGain, setLnaGain] = useState(15);
  const [tiaGain, setTiaGain] = useState(9);
  const [pgaGain, setPgaGain] = useState(12);

  // RX path
  const [rxPath, setRxPath] = useState<'LNAH' | 'LNAL' | 'LNAW'>('LNAL');  // Default to LNAL (low band)

  // DevBoard controls
  const [lnaEnable, setLnaEnable] = useState(true);
  const [paEnable, setPaEnable] = useState(false);
  const [attenuatorDb, setAttenuatorDb] = useState(0);
  const [vctcxoDac, setVctcxoDac] = useState(32768);
  
  // Band selection dialog
  const [bandDialogOpen, setBandDialogOpen] = useState(false);

  // Load devices on open
  useEffect(() => {
    if (open) {
      loadDevices();
      loadHardwareStatus();
    }
  }, [open]);

  const loadDevices = async () => {
    try {
      setRefreshing(true);
      const deviceList = await api.getSDRDevices();
      setDevices(deviceList);
      
      // Auto-select connected device
      const connected = deviceList.find(d => d.status === "connected");
      if (connected) {
        setSelectedDeviceId(connected.id);
      }
    } catch (error) {
      console.error('Failed to load devices:', error);
      toast.error('Failed to load SDR devices');
    } finally {
      setRefreshing(false);
    }
  };

  const loadHardwareStatus = async () => {
    try {
      const status = await api.getSDRStatus();
      setHardwareStatus(status);
      
      // Populate controls from hardware status
      if (status.rx_path) {
        setRxPath(status.rx_path as any);
      }
      // Gain and DevBoard config come from /api/sdr/config, not /api/sdr/status
      // Would need to fetch /api/sdr/config separately to populate those controls
    } catch (error) {
      console.error('Failed to load hardware status:', error);
    }
  };

  const handleConnect = async () => {
    if (!selectedDeviceId) {
      toast.error('Please select a device');
      return;
    }

    try {
      setLoading(true);
      await api.connectSDR(selectedDeviceId);
      toast.success('Connected to SDR device');
      await loadDevices();
      await loadHardwareStatus();
    } catch (error) {
      console.error('Failed to connect:', error);
      toast.error('Failed to connect to SDR device');
    } finally {
      setLoading(false);
    }
  };

  const handleDisconnect = async () => {
    try {
      setLoading(true);
      await api.disconnectSDR();
      toast.success('Disconnected from SDR device');
      await loadDevices();
      setHardwareStatus(null);
    } catch (error) {
      console.error('Failed to disconnect:', error);
      toast.error('Failed to disconnect from SDR device');
    } finally {
      setLoading(false);
    }
  };

  const handleApplyGains = async () => {
    try {
      setLoading(true);
      const gain: SDRGain = {
        lna_db: lnaGain,
        tia_db: tiaGain,
        pga_db: pgaGain,
      };
      
      await api.updateConfig({
        sdr: {
          gain,
          rx_path: rxPath,
        } as any,
      });
      
      toast.success('Gain settings applied');
    } catch (error) {
      console.error('Failed to apply gains:', error);
      toast.error('Failed to apply gain settings');
    } finally {
      setLoading(false);
    }
  };

  const handleApplyDevBoard = async () => {
    try {
      setLoading(true);
      const devboard: DevBoardConfig = {
        lna_enable: lnaEnable,
        pa_enable: paEnable,
        attenuator_db: attenuatorDb,
        vctcxo_dac: vctcxoDac,
      };
      
      await api.updateConfig({
        sdr: {
          devboard,
        } as any,
      });
      
      toast.success('DevBoard settings applied');
    } catch (error) {
      console.error('Failed to apply DevBoard settings:', error);
      toast.error('Failed to apply DevBoard settings');
    } finally {
      setLoading(false);
    }
  };

  const handleBandSelected = async (band: Band) => {
    // Validate band configuration
    const validation = validateBand(band);
    
    // Show warnings if any
    if (!validation.valid) {
      validation.warnings.forEach(warning => {
        toast.warning(warning, {
          duration: 5000,
          icon: <AlertTriangle className="w-4 h-4" />,
        });
      });
      
      // Show suggestions
      if (validation.suggestions.length > 0) {
        toast.info(validation.suggestions.join(' • '), {
          duration: 6000,
        });
      }
    }
    
    // Auto-populate frequency and sample rate from band specs
    if (band.center_freq_hz || band.sample_rate_hz || band.bandwidth_hz) {
      try {
        setLoading(true);
        
        const updates: any = {};
        if (band.center_freq_hz) {
          updates.center_freq_hz = band.center_freq_hz;
          
          // Auto-suggest RX path based on frequency
          const recommendedPath = getRecommendedRxPath(band.center_freq_hz);
          if (recommendedPath !== rxPath) {
            toast.info(`Consider switching RX path to ${recommendedPath} for optimal performance`, {
              duration: 5000,
            });
          }
        }
        if (band.sample_rate_hz) {
          updates.sample_rate_hz = band.sample_rate_hz;
        }
        if (band.bandwidth_hz) {
          updates.bandwidth_hz = band.bandwidth_hz;
        }
        
        await api.updateConfig({ sdr: updates });
        
        if (validation.valid) {
          toast.success(`Band ${band.name} applied with frequency ${(band.center_freq_hz! / 1e6).toFixed(1)} MHz`);
        } else {
          toast.warning(`Band ${band.name} applied with warnings - check settings`, {
            duration: 5000,
          });
        }
      } catch (error) {
        console.error('Failed to apply band config:', error);
        toast.error('Failed to apply band configuration');
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-3xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-center gap-2">
            <Radio className="w-5 h-5 text-primary" />
            <DialogTitle>Advanced SDR Settings</DialogTitle>
          </div>
          <DialogDescription>
            Device management, gain control, and hardware configuration
          </DialogDescription>
        </DialogHeader>

        <Tabs defaultValue="device" className="mt-4">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="device">Device</TabsTrigger>
            <TabsTrigger value="gains">Gains & Path</TabsTrigger>
            <TabsTrigger value="devboard">DevBoard</TabsTrigger>
          </TabsList>

          {/* Device Tab */}
          <TabsContent value="device" className="space-y-4 mt-4">
            <Card className="p-4">
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <Label>SDR Device</Label>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={loadDevices}
                    disabled={refreshing}
                  >
                    <RefreshCw className={`w-4 h-4 ${refreshing ? 'animate-spin' : ''}`} />
                  </Button>
                </div>

                <Select value={selectedDeviceId} onValueChange={setSelectedDeviceId}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select SDR device" />
                  </SelectTrigger>
                  <SelectContent>
                    {devices.map((device) => (
                      <SelectItem key={device.id} value={device.id}>
                        <div className="flex items-center gap-2">
                          <span>{device.model}</span>
                          {device.status === "connected" && (
                            <Badge variant="default" className="text-xs">Connected</Badge>
                          )}
                          {device.serial && (
                            <span className="text-xs text-muted-foreground">({device.serial})</span>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>

                <div className="flex gap-2">
                  <Button
                    onClick={handleConnect}
                    disabled={loading || !selectedDeviceId || devices.find(d => d.id === selectedDeviceId)?.status === "connected"}
                    className="flex-1"
                  >
                    Connect
                  </Button>
                  <Button
                    variant="outline"
                    onClick={handleDisconnect}
                    disabled={loading || !hardwareStatus?.connected}
                    className="flex-1"
                  >
                    Disconnect
                  </Button>
                </div>
              </div>
            </Card>

            {/* Hardware Status */}
            {hardwareStatus?.connected && (
              <Card className="p-4">
                <h3 className="text-sm font-medium mb-3 flex items-center gap-2">
                  <Cpu className="w-4 h-4" />
                  Hardware Status
                </h3>
                <div className="grid grid-cols-2 gap-3 text-sm">
                  <div>
                    <div className="text-muted-foreground">Device</div>
                    <div className="font-medium">{hardwareStatus.device_id || 'Unknown'}</div>
                  </div>
                  {hardwareStatus.temperature_c !== undefined && (
                    <div>
                      <div className="text-muted-foreground flex items-center gap-1">
                        <Thermometer className="w-3 h-3" />
                        Temperature
                      </div>
                      <div className="font-medium">{hardwareStatus.temperature_c.toFixed(1)}°C</div>
                    </div>
                  )}
                  {hardwareStatus.actual_freq_hz && (
                    <div>
                      <div className="text-muted-foreground">RX Frequency</div>
                      <div className="font-medium font-mono">
                        {(hardwareStatus.actual_freq_hz / 1e6).toFixed(3)} MHz
                      </div>
                    </div>
                  )}
                  {hardwareStatus.actual_sample_rate_hz && (
                    <div>
                      <div className="text-muted-foreground">Sample Rate</div>
                      <div className="font-medium font-mono">
                        {(hardwareStatus.actual_sample_rate_hz / 1e6).toFixed(1)} MSPS
                      </div>
                    </div>
                  )}
                </div>
              </Card>
            )}
          </TabsContent>

          {/* Gains & Path Tab */}
          <TabsContent value="gains" className="space-y-4 mt-4">
            <Card className="p-4">
              <h3 className="text-sm font-medium mb-4 flex items-center gap-2">
                <Zap className="w-4 h-4" />
                Gain Control
              </h3>
              
              <div className="space-y-6">
                {/* LNA Gain */}
                <div className="space-y-2">
                  <Label className="text-xs">
                    LNA Gain: {lnaGain} dB
                  </Label>
                  <Slider
                    min={0}
                    max={30}
                    step={1}
                    value={[lnaGain]}
                    onValueChange={([value]) => setLnaGain(value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Low Noise Amplifier (0-30 dB)
                  </p>
                </div>

                {/* TIA Gain */}
                <div className="space-y-2">
                  <Label className="text-xs">
                    TIA Gain: {tiaGain} dB
                  </Label>
                  <Slider
                    min={0}
                    max={12}
                    step={1}
                    value={[tiaGain]}
                    onValueChange={([value]) => setTiaGain(value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Transimpedance Amplifier (0-12 dB)
                  </p>
                </div>

                {/* PGA Gain */}
                <div className="space-y-2">
                  <Label className="text-xs">
                    PGA Gain: {pgaGain} dB
                  </Label>
                  <Slider
                    min={0}
                    max={32}
                    step={1}
                    value={[pgaGain]}
                    onValueChange={([value]) => setPgaGain(value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    Programmable Gain Amplifier (0-32 dB)
                  </p>
                </div>

                <div className="pt-2">
                  <div className="text-sm font-medium">
                    Total Gain: {lnaGain + tiaGain + pgaGain} dB
                  </div>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <h3 className="text-sm font-medium mb-4">RX Antenna Path</h3>
              
              <RadioGroup value={rxPath} onValueChange={(value: any) => setRxPath(value)}>
                <div className="space-y-3">
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="LNAH" id="rx-lnah" />
                    <Label htmlFor="rx-lnah" className="text-sm cursor-pointer">
                      LNAH - High Band (1.5 - 3.8 GHz)
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="LNAL" id="rx-lnal" />
                    <Label htmlFor="rx-lnal" className="text-sm cursor-pointer">
                      LNAL - Low Band (0.3 - 2.2 GHz) [Default]
                    </Label>
                  </div>
                  <div className="flex items-center space-x-2">
                    <RadioGroupItem value="LNAW" id="rx-lnaw" />
                    <Label htmlFor="rx-lnaw" className="text-sm cursor-pointer">
                      LNAW - Wideband (Full Range)
                    </Label>
                  </div>
                </div>
              </RadioGroup>
            </Card>

            <Button onClick={handleApplyGains} disabled={loading} className="w-full">
              Apply Gain & Path Settings
            </Button>
          </TabsContent>

          {/* DevBoard Tab */}
          <TabsContent value="devboard" className="space-y-4 mt-4">
            <Card className="p-4">
              <h3 className="text-sm font-medium mb-4">Front-End Controls</h3>
              
              <div className="space-y-4">
                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="lna-enable">LNA Enable</Label>
                    <p className="text-xs text-muted-foreground">
                      Enable Low Noise Amplifier (+19.5 dB)
                    </p>
                  </div>
                  <Switch
                    id="lna-enable"
                    checked={lnaEnable}
                    onCheckedChange={setLnaEnable}
                  />
                </div>

                <Separator />

                <div className="flex items-center justify-between">
                  <div>
                    <Label htmlFor="pa-enable">PA Enable</Label>
                    <p className="text-xs text-muted-foreground">
                      Enable Power Amplifier (+19.5 dB)
                    </p>
                  </div>
                  <Switch
                    id="pa-enable"
                    checked={paEnable}
                    onCheckedChange={setPaEnable}
                  />
                </div>

                <Separator />

                <div className="space-y-2">
                  <Label className="text-xs">
                    Attenuator: {attenuatorDb} dB
                  </Label>
                  <Slider
                    min={0}
                    max={18}
                    step={1}
                    value={[attenuatorDb]}
                    onValueChange={([value]) => setAttenuatorDb(value)}
                  />
                  <p className="text-xs text-muted-foreground">
                    RX attenuator (0-18 dB in 1 dB steps)
                  </p>
                </div>
              </div>
            </Card>

            <Card className="p-4">
              <h3 className="text-sm font-medium mb-4">Clock Control</h3>
              
              <div className="space-y-2">
                <Label className="text-xs">
                  VCTCXO DAC: {vctcxoDac}
                </Label>
                <Slider
                  min={0}
                  max={65535}
                  step={1}
                  value={[vctcxoDac]}
                  onValueChange={([value]) => setVctcxoDac(value)}
                />
                <p className="text-xs text-muted-foreground">
                  Fine frequency tuning (±275 Hz range)
                </p>
              </div>
            </Card>

            <Card className="p-4 bg-muted/50">
              <h3 className="text-sm font-medium mb-2">Duplexer Band Selection</h3>
              <p className="text-xs text-muted-foreground mb-3">
                Configure FDD, TX-only, RX-only, or TDD band settings
              </p>
              <Button 
                onClick={() => setBandDialogOpen(true)} 
                variant="outline" 
                className="w-full"
              >
                <Radio className="w-4 h-4 mr-2" />
                Select Band
              </Button>
            </Card>

            <Button onClick={handleApplyDevBoard} disabled={loading} className="w-full">
              Apply DevBoard Settings
            </Button>
          </TabsContent>
        </Tabs>
      </DialogContent>
      
      <BandSelectionDialog 
        open={bandDialogOpen} 
        onClose={() => setBandDialogOpen(false)}
        onBandSelected={handleBandSelected}
      />
    </Dialog>
  );
}
