import { useEffect, useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { EnhancedSpectrumDisplay } from '@/components/spectrum/EnhancedSpectrumDisplay';
import { DetectionList } from '@/components/detections/DetectionList';
import { ClusterView } from '@/components/clusters/ClusterView';
import { UnifiedSDRPanel } from '@/components/controls/UnifiedSDRPanel';
import { DecoderPanel } from '@/components/demod/DecoderPanel';
import { FileManager } from '@/components/file-manager/FileManager';
import { ReportGenerator } from '@/components/reports/ReportGenerator';
import { PerformanceDashboard } from '@/components/performance/PerformanceDashboard';
import { KeyboardShortcutsHelp } from '@/components/KeyboardShortcutsHelp';
import { useKeyboardShortcuts, type KeyboardShortcut } from '@/hooks/useKeyboardShortcuts';
import { AdvancedSDRDialog } from '@/components/controls/AdvancedSDRDialog';
import { RSSIMeter } from '@/components/monitoring/RSSIMeter';
import { DetectionTimeline } from '@/components/monitoring/DetectionTimeline';
import { GPUMetricsPanel } from '@/components/gpu';
import { SystemStatusPanel } from '@/components/status/SystemStatusPanel';
import { useSDR } from '@/hooks/useSDR';
import { MLClassificationDashboard } from '@/components/ml';
import { Activity, Cpu, Gauge, Zap, WifiOff, Wifi, Settings, FileText, Bell, Bookmark, Brain } from 'lucide-react';
import {
  startSpectrumStream,
  startDetectionsStream,
  startClustersStream,
  startDemodulationStream,
  setLoRaFrameCallback,
  setBLEPacketCallback,
  stopAllStreams,
  areStreamsConnected,
  fetchInitialData,
  type LoRaFrameEvent,
  type BLEPacketEvent
} from '@/services/websocket';
import type { LoRaFrame, BLEPacket } from '@/types';
import { api, type SystemStatus } from '@/lib/api';
import { useDetectionStore } from '@/stores/detectionStore';
import { useClusterStore } from '@/stores/clusterStore';
import { useConfigStore } from '@/stores/configStore';
import { useSpectrumStore } from '@/stores/spectrumStore';
import { toast } from 'sonner';
import { trpc } from '@/lib/trpc';
import { ThresholdAlertsDialog } from '@/components/alerts/ThresholdAlertsDialog';
import { BookmarkDialog } from '@/components/bookmarks/BookmarkDialog';
import { useThresholdMonitor } from '@/hooks/useThresholdMonitor';

/**
 * Live Monitoring Page
 * Real-time GPU-accelerated RF signal analysis with WebSocket streaming
 */
export default function LiveMonitoring() {
  const [isRunning, setIsRunning] = useState(false);
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [connections, setConnections] = useState({
    spectrum: false,
    detections: false,
    clusters: false
  });
  const [showSettings, setShowSettings] = useState(true);  // Show SDR panel by default
  const [showReportGenerator, setShowReportGenerator] = useState(false);
  const [showKeyboardHelp, setShowKeyboardHelp] = useState(false);
  const [showAdvancedSDR, setShowAdvancedSDR] = useState(false);
  const [activeTab, setActiveTab] = useState('detections');
  const [currentRSSI, setCurrentRSSI] = useState<number | null>(null); // null = no signal
  const [showThresholdAlerts, setShowThresholdAlerts] = useState(false);
  const [showBookmarkDialog, setShowBookmarkDialog] = useState(false);
  const [bookmarkData, setBookmarkData] = useState<any>(null);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>('');

  // Decoder panel state
  const [loraFrames, setLoraFrames] = useState<LoRaFrame[]>([]);
  const [blePackets, setBlePackets] = useState<BLEPacket[]>([]);

  // Subscribe to size for re-render triggers, get data via getState() for stable reference
  const detectionCount = useDetectionStore((state) => state.detections.size);
  const clusterCount = useClusterStore((state) => state.clusters.size);
  const currentPsd = useSpectrumStore((state) => state.currentPsd);
  
  // Get detections array via getState() - avoids unstable selector snapshot issue
  // detectionCount subscription ensures we re-render when detections change
  const detections = useDetectionStore.getState().getActiveDetections();

  // Connect threshold monitoring with real-time detections
  useThresholdMonitor(detections.map(d => ({
    center_freq_hz: d.centerFreqHz,
    power_dbm: d.snrDb, // Using SNR as power indicator
    snr_db: d.snrDb
  })));

  // SDR status and metrics
  const {
    gpsStatus,
    systemHealth,
    pllLocked,
    uptimeSeconds,
    overflowCount,
    reconnectCount,
    temperatureC,
  } = useSDR({ pollInterval: 1000, pollOnlyWhenConnected: false });

  // Recordings
  const { data: recordings = [], refetch: refetchRecordings } = trpc.recordings.list.useQuery();
  const createRecordingMutation = trpc.recordings.create.useMutation({
    onSuccess: () => {
      refetchRecordings();
      toast.success('Recording saved successfully');
    },
  });
  const deleteRecordingMutation = trpc.recordings.delete.useMutation({
    onSuccess: () => {
      refetchRecordings();
      toast.success('Recording deleted');
    },
  });

  // Fetch initial detections/clusters on mount
  useEffect(() => {
    fetchInitialData();
  }, []);

  // Poll system status
  useEffect(() => {
    let backendOffline = false;
    const pollStatus = async () => {
      try {
        const s = await api.getStatus();
        setStatus(s);
        setIsRunning(s.state === 'running');
        if (backendOffline) {
          backendOffline = false;
          console.log('[Backend] Connection restored');
        }
      } catch (error) {
        // Suppress repeated errors when backend is offline
        if (!backendOffline) {
          backendOffline = true;
          console.warn('[Backend] Offline - waiting for connection...');
        }
        // Set default offline status
        setStatus({
          state: 'idle',
          uptime_seconds: 0,
          samples_processed: 0,
          detections_count: 0,
          current_throughput_msps: 0,
          gpu_memory_used_gb: 0,
          buffer_fill_level: 0,
          processing_latency_ms: 0,
        });
        setIsRunning(false);
      }
    };

    pollStatus();
    const interval = setInterval(pollStatus, 2000);

    return () => clearInterval(interval);
  }, []);

  // Check WebSocket connections
  useEffect(() => {
    const checkConnections = () => {
      setConnections(areStreamsConnected());
    };

    const interval = setInterval(checkConnections, 1000);
    return () => clearInterval(interval);
  }, []);

  // Calculate RSSI from spectrum data (proper dB averaging via linear domain)
  useEffect(() => {
    if (!currentPsd || currentPsd.length === 0) {
      // No PSD data - show "No Signal" state
      setCurrentRSSI(null);
      return;
    }

    // Convert dB to linear power, average, then convert back to dB
    // P_linear = 10^(P_dB/10)
    // P_avg_linear = sum(P_linear) / N
    // P_avg_dB = 10 * log10(P_avg_linear)
    let sumLinearPower = 0;
    for (let i = 0; i < currentPsd.length; i++) {
      // Clamp to prevent underflow with very low dB values
      const dbValue = Math.max(currentPsd[i], -150);
      sumLinearPower += Math.pow(10, dbValue / 10);
    }
    const avgLinearPower = sumLinearPower / currentPsd.length;
    const avgPowerDb = 10 * Math.log10(avgLinearPower);

    // Update RSSI state (clamp to reasonable range)
    setCurrentRSSI(Math.max(-120, Math.min(-10, avgPowerDb)));
  }, [currentPsd]);

  // Set up LoRa/BLE decoder callbacks
  useEffect(() => {
    const MAX_FRAMES = 100; // Keep last 100 frames

    // LoRa frame callback - convert from WebSocket format to frontend format
    setLoRaFrameCallback((event: LoRaFrameEvent) => {
      const frame: LoRaFrame = {
        timestamp_ns: event.timestamp * 1e9,
        freqHz: event.center_freq_hz,
        spreadingFactor: event.spreading_factor,
        bandwidthHz: event.bandwidth_hz,
        codingRate: event.coding_rate,
        payload: event.payload_hex,
        crcValid: event.crc_valid,
        rssi: event.rssi_dbm,
        snrDb: event.snr_db,
      };
      setLoraFrames(prev => [frame, ...prev].slice(0, MAX_FRAMES));
    });

    // BLE packet callback - convert from WebSocket format to frontend format
    setBLEPacketCallback((event: BLEPacketEvent) => {
      const packet: BLEPacket = {
        timestamp_ns: event.timestamp * 1e9,
        channel: event.channel,
        accessAddress: parseInt(event.access_address, 16),
        packetType: event.pdu_type,
        payload: event.payload_hex,
        crcValid: event.crc_valid,
        rssi: event.rssi_dbm,
      };
      setBlePackets(prev => [packet, ...prev].slice(0, MAX_FRAMES));
    });

    // Cleanup callbacks on unmount
    return () => {
      setLoRaFrameCallback(() => {});
      setBLEPacketCallback(() => {});
    };
  }, []);

  const handleStart = async () => {
    if (!selectedDeviceId) {
      toast.error('Select an SDR before starting the stream');
      return;
    }
    try {
      // Start backend stream for the selected device
      await api.startDeviceStream(selectedDeviceId);
      // Then start WebSocket streams to receive data
      startSpectrumStream();
      startDetectionsStream();
      startClustersStream();
      startDemodulationStream(); // Unified LoRa/BLE stream
      setIsRunning(true);
      toast.success('Acquisition started');
    } catch (error) {
      console.error('Failed to start:', error);
      toast.error('Failed to start acquisition');
    }
  };

  const handleStop = async () => {
    try {
      if (selectedDeviceId) {
        await api.stopDeviceStream(selectedDeviceId);
      }
      // Stop WebSocket streams
      stopAllStreams();
      setIsRunning(false);
      toast.success('Acquisition stopped');
    } catch (error) {
      console.error('Failed to stop:', error);
      toast.error('Failed to stop acquisition');
    }
  };

  // Spectrum control actions from store
  const togglePeakHold = useSpectrumStore((state) => state.togglePeakHold);
  const toggleAverage = useSpectrumStore((state) => state.toggleAverage);
  const clearMarkers = useSpectrumStore((state) => state.clearMarkers);
  const addMarker = useSpectrumStore((state) => state.addMarker);
  const centerFreqHz = useSpectrumStore((state) => state.centerFreqHz);

  // Keyboard shortcuts
  const shortcuts: KeyboardShortcut[] = [
    {
      key: ' ',
      description: 'Start/Stop pipeline',
      action: () => {
        if (isRunning) {
          handleStop();
        } else {
          handleStart();
        }
      },
    },
    {
      key: 'r',
      description: 'Toggle recording',
      action: () => {
        // Recording controls handle their own state
        document.querySelector<HTMLButtonElement>('[data-recording-toggle]')?.click();
      },
    },
    {
      key: 's',
      description: 'Toggle settings panel',
      action: () => setShowSettings((prev) => !prev),
    },
    {
      key: 'p',
      description: 'Switch to Performance tab',
      action: () => setActiveTab('performance'),
    },
    {
      key: 'd',
      description: 'Switch to Detections tab',
      action: () => setActiveTab('detections'),
    },
    {
      key: 'c',
      description: 'Switch to Clusters tab',
      action: () => setActiveTab('clusters'),
    },
    {
      key: '?',
      description: 'Show keyboard shortcuts',
      action: () => setShowKeyboardHelp((prev) => !prev),
    },
    // Spectrum trace shortcuts (Shift modifier to avoid conflicts)
    {
      key: 'P',
      shift: true,
      description: 'Toggle Peak Hold trace',
      action: () => {
        togglePeakHold();
        toast.success('Peak Hold toggled', { duration: 1500 });
      },
    },
    {
      key: 'A',
      shift: true,
      description: 'Toggle Average trace',
      action: () => {
        toggleAverage();
        toast.success('Average toggled', { duration: 1500 });
      },
    },
    {
      key: 'm',
      description: 'Add marker at center freq',
      action: () => {
        addMarker(centerFreqHz, `${(centerFreqHz / 1e6).toFixed(1)}`);
        toast.success('Marker added', { duration: 1500 });
      },
    },
    {
      key: 'M',
      shift: true,
      description: 'Clear all markers',
      action: () => {
        clearMarkers();
        toast.success('Markers cleared', { duration: 1500 });
      },
    },
  ];

  useKeyboardShortcuts({ shortcuts });

  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 sticky top-0 z-50">
        <div className="container mx-auto px-4 py-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div className="flex items-center gap-3">
                <img src="/wsdr-logo.svg" alt="wSDR" className="h-8" />
                <h1 className="text-xl font-bold">RF Signal Forensics</h1>
              </div>
              
              {/* Status Badges */}
              <div className="flex items-center gap-2">
                <Badge 
                  variant={isRunning ? "default" : "secondary"}
                  className="gap-1"
                >
                  {isRunning ? (
                    <>
                      <Zap className="w-3 h-3" />
                      Running
                    </>
                  ) : (
                    <>
                      <WifiOff className="w-3 h-3" />
                      Stopped
                    </>
                  )}
                </Badge>

                <Badge 
                  variant={connections.spectrum && connections.detections ? "default" : "secondary"}
                  className="gap-1"
                >
                  {connections.spectrum && connections.detections ? (
                    <>
                      <Wifi className="w-3 h-3" />
                      Connected
                    </>
                  ) : (
                    <>
                      <WifiOff className="w-3 h-3" />
                      Disconnected
                    </>
                  )}
                </Badge>
              </div>
            </div>

            {/* GPU Metrics */}
            <div className="flex items-center gap-4">
              {status && (
                <div className="flex items-center gap-4 text-xs">
                  <div className="flex items-center gap-2">
                    <Cpu className="w-3 h-3 text-muted-foreground" />
                    <span className="text-muted-foreground">GPU:</span>
                    <span className="font-mono font-medium">
                      {status.gpu_memory_used_gb.toFixed(2)} GB
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Gauge className="w-3 h-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Throughput:</span>
                    <span className="font-mono font-medium">
                      {status.current_throughput_msps.toFixed(1)} Msps
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Zap className="w-3 h-3 text-muted-foreground" />
                    <span className="text-muted-foreground">Latency:</span>
                    <span className="font-mono font-medium">
                      {status.processing_latency_ms.toFixed(1)} ms
                    </span>
                  </div>
                </div>
              )}

              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowThresholdAlerts(true)}
                title="Power Threshold Alerts"
              >
                <Bell className="w-4 h-4 mr-2" />
                Alerts
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowReportGenerator(true)}
              >
                <FileText className="w-4 h-4 mr-2" />
                Report
              </Button>

              <Button
                variant="outline"
                size="sm"
                onClick={() => setShowSettings(!showSettings)}
              >
                <Settings className="w-4 h-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-6">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
          {/* Left Sidebar - Decoder & File Manager */}
          {showSettings && (
            <div className="lg:col-span-3 space-y-4">
              <DecoderPanel
                loraFrames={loraFrames}
                blePackets={blePackets}
                maxHeight={400}
              />

              <FileManager />
            </div>
          )}

          {/* Center - Spectrum Display */}
          <div className={showSettings ? "lg:col-span-6 overflow-hidden" : "lg:col-span-9 overflow-hidden"}>
            <EnhancedSpectrumDisplay />
            
            {/* Detection Timeline */}
            <DetectionTimeline className="mt-4" timeWindowSeconds={60} />

            {/* GPU Pipeline Metrics */}
            <GPUMetricsPanel className="mt-6" pollInterval={1000} />
          </div>

          {/* Right Sidebar - SDR Controls, Metrics & Detections */}
          <div className="lg:col-span-3 space-y-4">
            <UnifiedSDRPanel
              onStart={handleStart}
              onStop={handleStop}
              isRunning={isRunning}
              selectedDeviceId={selectedDeviceId}
              onSelectDevice={setSelectedDeviceId}
            />
            <RSSIMeter currentRSSI={currentRSSI} />

            <SystemStatusPanel
              gpsStatus={gpsStatus}
              systemHealth={systemHealth}
              pllLocked={pllLocked}
              uptimeSeconds={uptimeSeconds}
              overflowCount={overflowCount}
              reconnectCount={reconnectCount}
              temperatureC={temperatureC}
            />

            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="detections" className="text-xs px-2">
                  Detections
                  {detectionCount > 0 && (
                    <Badge variant="secondary" className="ml-1 text-[10px] px-1">
                      {detectionCount}
                    </Badge>
                  )}
                </TabsTrigger>
                <TabsTrigger value="clusters" className="text-xs px-2">
                  Clusters
                  {clusterCount > 0 && (
                    <Badge variant="secondary" className="ml-1 text-[10px] px-1">
                      {clusterCount}
                    </Badge>
                  )}
                </TabsTrigger>
                <TabsTrigger value="ml" className="text-xs px-2">
                  <Brain className="w-3 h-3 mr-1" />
                  ML
                </TabsTrigger>
                <TabsTrigger value="performance" className="text-xs px-2">
                  Perf
                </TabsTrigger>
              </TabsList>

              <TabsContent value="detections" className="mt-4">
                <DetectionList
                  maxHeight={600}
                  onBookmark={(detection) => {
                    setBookmarkData({
                      centerFreqHz: detection.centerFreqHz,
                      powerDbm: detection.peakPowerDb,
                      snrDb: detection.snrDb,
                      bandwidthHz: detection.bandwidthHz,
                      modulationType: detection.modulationType,
                      detectionId: detection.id
                    });
                    setShowBookmarkDialog(true);
                  }}
                />
              </TabsContent>

              <TabsContent value="clusters" className="mt-4">
                <ClusterView maxHeight={600} />
              </TabsContent>

              <TabsContent value="ml" className="mt-4">
                <MLClassificationDashboard maxHeight={600} />
              </TabsContent>

              <TabsContent value="performance" className="mt-4">
                <PerformanceDashboard />
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </main>

      {/* Report Generator Dialog */}
      <ReportGenerator
        open={showReportGenerator}
        onClose={() => setShowReportGenerator(false)}
      />

      {/* Keyboard Shortcuts Help Dialog */}
      <KeyboardShortcutsHelp
        open={showKeyboardHelp}
        onClose={() => setShowKeyboardHelp(false)}
        shortcuts={shortcuts}
      />

      {/* Advanced SDR Dialog */}
      <AdvancedSDRDialog
        open={showAdvancedSDR}
        onClose={() => setShowAdvancedSDR(false)}
      />

      {/* Threshold Alerts Dialog */}
      <ThresholdAlertsDialog
        open={showThresholdAlerts}
        onOpenChange={setShowThresholdAlerts}
      />

      {/* Bookmark Dialog */}
      <BookmarkDialog
        open={showBookmarkDialog}
        onOpenChange={setShowBookmarkDialog}
        initialData={bookmarkData}
      />
    </div>
  );
}
