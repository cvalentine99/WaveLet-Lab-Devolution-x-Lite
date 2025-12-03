import { useEffect, useRef, useState } from 'react';
import { useWebSocket } from '@/hooks/useWebSocket';
import { parseSpectrumFrame, parseDetectionEvent, type DetectionEvent } from '@/lib/spectrumParser';
import { WaterfallDisplay, type WaterfallDisplayRef } from '@/components/spectrum/WaterfallDisplay';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Activity, Wifi, Zap, AlertTriangle } from 'lucide-react';

// Backend WebSocket URL - use env var or default to localhost:8765
const WS_BASE = import.meta.env.VITE_BACKEND_WS_URL || 'ws://localhost:8765';

/**
 * Live RF Monitoring Page
 * Real-time spectrum waterfall with detection overlays
 */
export default function Live() {
  const waterfallRef = useRef<WaterfallDisplayRef>(null);
  const [detections, setDetections] = useState<DetectionEvent[]>([]);
  const [metrics, setMetrics] = useState({
    fps: 0,
    latency: 0,
    gpuLoad: 0,
    detectionCount: 0,
  });
  const [isStreaming, setIsStreaming] = useState(false);

  // WebSocket for spectrum data
  const spectrumWs = useWebSocket({
    url: `${WS_BASE}/ws/spectrum`,
    onMessage: (data: ArrayBuffer) => {
      try {
        const frame = parseSpectrumFrame(data);
        waterfallRef.current?.renderSpectrumLine(frame);
      } catch (error) {
        console.error('[Spectrum] Parse error:', error);
      }
    },
    reconnect: true,
  });

  // WebSocket for detection events
  const detectionWs = useWebSocket({
    url: `${WS_BASE}/ws/detections`,
    onMessage: (data: ArrayBuffer) => {
      try {
        const text = new TextDecoder().decode(data);
        const detection = parseDetectionEvent(text);
        
        setDetections((prev) => {
          const updated = [detection, ...prev].slice(0, 100); // Keep last 100
          return updated;
        });

        setMetrics((prev) => ({
          ...prev,
          detectionCount: prev.detectionCount + 1,
        }));
      } catch (error) {
        console.error('[Detection] Parse error:', error);
      }
    },
    reconnect: true,
  });

  useEffect(() => {
    setIsStreaming(
      spectrumWs.status === 'connected' && detectionWs.status === 'connected'
    );
  }, [spectrumWs.status, detectionWs.status]);

  return (
    <div className="min-h-screen bg-background p-4">
      {/* Header */}
      <div className="mb-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Activity className="h-6 w-6 text-primary" />
          <h1 className="text-2xl font-bold">Live RF Monitoring</h1>
          <Badge variant={isStreaming ? 'default' : 'secondary'}>
            {isStreaming ? 'STREAMING' : 'OFFLINE'}
          </Badge>
        </div>

        <div className="flex items-center gap-4 text-sm font-mono">
          <div className="flex items-center gap-2">
            <Zap className="h-4 w-4 text-green-500" />
            <span>{metrics.fps} FPS</span>
          </div>
          <div className="flex items-center gap-2">
            <Activity className="h-4 w-4 text-blue-500" />
            <span>{metrics.latency}ms</span>
          </div>
          <div className="flex items-center gap-2">
            <Wifi className="h-4 w-4 text-purple-500" />
            <span>{metrics.gpuLoad}% GPU</span>
          </div>
        </div>
      </div>

      {/* Main Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-4">
        {/* Waterfall Display - Takes 3 columns */}
        <div className="lg:col-span-3">
          <Card className="p-4">
            <div className="mb-2 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Spectrum Waterfall</h2>
              <div className="flex gap-2">
                <Button size="sm" variant="outline">
                  Settings
                </Button>
                <Button size="sm" variant="outline">
                  Export
                </Button>
              </div>
            </div>
            <WaterfallDisplay
              ref={waterfallRef}
              width={1024}
              height={600}
              colormap="viridis"
              showGrid={true}
              showFrequencyAxis={true}
              showTimeAxis={true}
            />
          </Card>
        </div>

        {/* Detection Sidebar - Takes 1 column */}
        <div className="lg:col-span-1">
          <Card className="p-4 h-full">
            <div className="mb-3 flex items-center justify-between">
              <h2 className="text-lg font-semibold">Detections</h2>
              <Badge variant="secondary">{detections.length}</Badge>
            </div>

            <ScrollArea className="h-[600px]">
              <div className="space-y-2">
                {detections.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    <AlertTriangle className="h-8 w-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm">No detections yet</p>
                  </div>
                ) : (
                  detections.map((det) => (
                    <DetectionCard key={det.id} detection={det} />
                  ))
                )}
              </div>
            </ScrollArea>
          </Card>
        </div>
      </div>

      {/* Bottom Metrics Panel */}
      <div className="mt-4 grid grid-cols-1 md:grid-cols-4 gap-4">
        <MetricCard
          title="Total Detections"
          value={metrics.detectionCount.toString()}
          icon={<Activity className="h-5 w-5" />}
          color="text-green-500"
        />
        <MetricCard
          title="Active Clusters"
          value="0"
          icon={<Wifi className="h-5 w-5" />}
          color="text-blue-500"
        />
        <MetricCard
          title="Anomalies"
          value="0"
          icon={<AlertTriangle className="h-5 w-5" />}
          color="text-yellow-500"
        />
        <MetricCard
          title="Avg SNR"
          value="--"
          icon={<Zap className="h-5 w-5" />}
          color="text-purple-500"
        />
      </div>
    </div>
  );
}

function DetectionCard({ detection }: { detection: DetectionEvent }) {
  const freqMHz = (detection.frequencyStart / 1e6).toFixed(3);
  const bwKHz = (detection.bandwidth / 1e3).toFixed(1);
  const ageSeconds = Math.floor((Date.now() - detection.timestamp) / 1000);

  return (
    <div className="border border-border rounded-lg p-3 bg-card hover:bg-accent/50 transition-colors cursor-pointer">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs">
            {detection.signalClass}
          </Badge>
          <span className="text-xs text-muted-foreground">{ageSeconds}s ago</span>
        </div>
        <Badge variant={detection.confidence > 0.8 ? 'default' : 'secondary'}>
          {(detection.confidence * 100).toFixed(0)}%
        </Badge>
      </div>

      <div className="space-y-1 text-sm font-mono">
        <div className="flex justify-between">
          <span className="text-muted-foreground">Freq:</span>
          <span className="text-primary">{freqMHz} MHz</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">BW:</span>
          <span>{bwKHz} kHz</span>
        </div>
        <div className="flex justify-between">
          <span className="text-muted-foreground">Duration:</span>
          <span>{detection.duration.toFixed(1)} ms</span>
        </div>
      </div>
    </div>
  );
}

function MetricCard({
  title,
  value,
  icon,
  color,
}: {
  title: string;
  value: string;
  icon: React.ReactNode;
  color: string;
}) {
  return (
    <Card className="p-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-muted-foreground mb-1">{title}</p>
          <p className="text-2xl font-bold font-mono">{value}</p>
        </div>
        <div className={color}>{icon}</div>
      </div>
    </Card>
  );
}
