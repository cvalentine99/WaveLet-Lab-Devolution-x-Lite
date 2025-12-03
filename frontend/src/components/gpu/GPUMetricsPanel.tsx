import { useState, useEffect, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import {
  Cpu,
  Thermometer,
  HardDrive,
  Activity,
  Zap,
  Clock,
  TrendingUp,
  AlertTriangle,
  CheckCircle2
} from 'lucide-react';
import { api, type SystemStatus } from '@/lib/api';
import { cn } from '@/lib/utils';

export interface GPUMetricsPanelProps {
  /** Polling interval in ms */
  pollInterval?: number;
  /** Compact mode for sidebar */
  compact?: boolean;
  className?: string;
}

interface GPUState {
  status: SystemStatus | null;
  loading: boolean;
  error: string | null;
}

/**
 * GPU Metrics Panel
 * Real-time RTX 4090 performance monitoring for RF forensics pipeline
 */
export function GPUMetricsPanel({
  pollInterval = 1000,
  compact = false,
  className
}: GPUMetricsPanelProps) {
  const [state, setState] = useState<GPUState>({
    status: null,
    loading: true,
    error: null,
  });

  const fetchStatus = useCallback(async () => {
    try {
      const status = await api.getStatus();
      setState({ status, loading: false, error: null });
    } catch (err) {
      setState(prev => ({
        ...prev,
        loading: false,
        error: err instanceof Error ? err.message : 'Failed to fetch GPU status'
      }));
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, pollInterval);
    return () => clearInterval(interval);
  }, [fetchStatus, pollInterval]);

  const { status, loading, error } = state;

  // Derive health indicators
  const gpuMemoryPercent = status ? (status.gpu_memory_used_gb / 24) * 100 : 0; // RTX 4090 = 24GB
  const bufferHealth = status ? status.buffer_fill_level * 100 : 0;
  const latencyHealth = status ? (status.processing_latency_ms < 10 ? 'good' : status.processing_latency_ms < 50 ? 'warn' : 'critical') : 'unknown';

  const getStateColor = (state: string) => {
    switch (state) {
      case 'running': return 'bg-green-500';
      case 'paused': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      case 'configuring': return 'bg-blue-500';
      default: return 'bg-gray-500';
    }
  };

  const getStateIcon = (state: string) => {
    switch (state) {
      case 'running': return <CheckCircle2 className="w-3 h-3" />;
      case 'error': return <AlertTriangle className="w-3 h-3" />;
      default: return null;
    }
  };

  if (loading && !status) {
    return (
      <Card className={cn('p-4', className)}>
        <div className="flex items-center gap-2 text-muted-foreground">
          <Cpu className="w-4 h-4 animate-pulse" />
          <span className="text-sm">Loading GPU metrics...</span>
        </div>
      </Card>
    );
  }

  if (error && !status) {
    return (
      <Card className={cn('p-4', className)}>
        <div className="flex items-center gap-2 text-red-400">
          <AlertTriangle className="w-4 h-4" />
          <span className="text-sm">{error}</span>
        </div>
      </Card>
    );
  }

  if (compact) {
    return (
      <TooltipProvider>
        <Card className={cn('p-3', className)}>
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-green-500" />
              <span className="text-xs font-semibold">RTX 4090</span>
            </div>
            <div className="flex items-center gap-1">
              <div className={cn('w-2 h-2 rounded-full', getStateColor(status?.state || 'idle'))} />
              <span className="text-[10px] text-muted-foreground capitalize">{status?.state || 'idle'}</span>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-[10px]">
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1">
                  <HardDrive className="w-3 h-3 text-muted-foreground" />
                  <span className="font-mono">{status?.gpu_memory_used_gb.toFixed(1)}GB</span>
                </div>
              </TooltipTrigger>
              <TooltipContent>VRAM Usage ({gpuMemoryPercent.toFixed(0)}% of 24GB)</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1">
                  <TrendingUp className="w-3 h-3 text-muted-foreground" />
                  <span className="font-mono">{status?.current_throughput_msps.toFixed(1)} Msps</span>
                </div>
              </TooltipTrigger>
              <TooltipContent>Processing Throughput</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1">
                  <Clock className="w-3 h-3 text-muted-foreground" />
                  <span className={cn('font-mono',
                    latencyHealth === 'good' ? 'text-green-400' :
                    latencyHealth === 'warn' ? 'text-yellow-400' : 'text-red-400'
                  )}>
                    {status?.processing_latency_ms.toFixed(1)}ms
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>Processing Latency</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <div className="flex items-center gap-1">
                  <Activity className="w-3 h-3 text-muted-foreground" />
                  <span className="font-mono">{status?.detections_count.toLocaleString()}</span>
                </div>
              </TooltipTrigger>
              <TooltipContent>Total Detections</TooltipContent>
            </Tooltip>
          </div>
        </Card>
      </TooltipProvider>
    );
  }

  // Full panel view
  return (
    <TooltipProvider>
      <Card className={cn('p-4', className)}>
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-green-500" />
            <h3 className="text-sm font-semibold">GPU Pipeline</h3>
            <Badge variant="outline" className="text-[10px] bg-green-500/10 text-green-400 border-green-500/30">
              RTX 4090
            </Badge>
          </div>
          <div className="flex items-center gap-2">
            {getStateIcon(status?.state || 'idle')}
            <Badge
              variant="secondary"
              className={cn('text-xs capitalize',
                status?.state === 'running' ? 'bg-green-500/20 text-green-400' :
                status?.state === 'error' ? 'bg-red-500/20 text-red-400' :
                'bg-muted'
              )}
            >
              {status?.state || 'idle'}
            </Badge>
          </div>
        </div>

        {/* Metrics Grid */}
        <div className="space-y-4">
          {/* VRAM Usage */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <HardDrive className="w-3.5 h-3.5" />
                <span>VRAM Usage</span>
              </div>
              <span className="font-mono">
                {status?.gpu_memory_used_gb.toFixed(1)} / 24 GB
              </span>
            </div>
            <Progress
              value={gpuMemoryPercent}
              className="h-2"
            />
          </div>

          {/* Buffer Fill */}
          <div className="space-y-1.5">
            <div className="flex items-center justify-between text-xs">
              <div className="flex items-center gap-1.5 text-muted-foreground">
                <Activity className="w-3.5 h-3.5" />
                <span>Ring Buffer</span>
              </div>
              <span className={cn('font-mono',
                bufferHealth > 80 ? 'text-red-400' :
                bufferHealth > 60 ? 'text-yellow-400' : 'text-green-400'
              )}>
                {bufferHealth.toFixed(0)}%
              </span>
            </div>
            <Progress
              value={bufferHealth}
              className={cn('h-2',
                bufferHealth > 80 ? '[&>div]:bg-red-500' :
                bufferHealth > 60 ? '[&>div]:bg-yellow-500' : ''
              )}
            />
          </div>

          {/* Stats Grid */}
          <div className="grid grid-cols-2 gap-3 pt-2">
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="p-2.5 rounded-lg bg-muted/50 border">
                  <div className="flex items-center gap-1.5 text-muted-foreground text-[10px] mb-1">
                    <TrendingUp className="w-3 h-3" />
                    <span>Throughput</span>
                  </div>
                  <div className="text-lg font-mono font-semibold text-green-400">
                    {status?.current_throughput_msps.toFixed(1)}
                    <span className="text-xs text-muted-foreground ml-1">Msps</span>
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent>Real-time sample processing rate</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <div className="p-2.5 rounded-lg bg-muted/50 border">
                  <div className="flex items-center gap-1.5 text-muted-foreground text-[10px] mb-1">
                    <Clock className="w-3 h-3" />
                    <span>Latency</span>
                  </div>
                  <div className={cn('text-lg font-mono font-semibold',
                    latencyHealth === 'good' ? 'text-green-400' :
                    latencyHealth === 'warn' ? 'text-yellow-400' : 'text-red-400'
                  )}>
                    {status?.processing_latency_ms.toFixed(1)}
                    <span className="text-xs text-muted-foreground ml-1">ms</span>
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent>End-to-end processing latency</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <div className="p-2.5 rounded-lg bg-muted/50 border">
                  <div className="flex items-center gap-1.5 text-muted-foreground text-[10px] mb-1">
                    <Zap className="w-3 h-3" />
                    <span>Detections</span>
                  </div>
                  <div className="text-lg font-mono font-semibold text-blue-400">
                    {status?.detections_count.toLocaleString()}
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent>Total signals detected this session</TooltipContent>
            </Tooltip>

            <Tooltip>
              <TooltipTrigger asChild>
                <div className="p-2.5 rounded-lg bg-muted/50 border">
                  <div className="flex items-center gap-1.5 text-muted-foreground text-[10px] mb-1">
                    <Activity className="w-3 h-3" />
                    <span>Samples</span>
                  </div>
                  <div className="text-lg font-mono font-semibold text-purple-400">
                    {((status?.samples_processed ?? 0) / 1e9).toFixed(2)}
                    <span className="text-xs text-muted-foreground ml-1">G</span>
                  </div>
                </div>
              </TooltipTrigger>
              <TooltipContent>Total samples processed (billions)</TooltipContent>
            </Tooltip>
          </div>

          {/* Uptime */}
          <div className="flex items-center justify-between pt-2 border-t text-xs text-muted-foreground">
            <span>Uptime</span>
            <span className="font-mono">
              {formatUptime(status?.uptime_seconds || 0)}
            </span>
          </div>
        </div>
      </Card>
    </TooltipProvider>
  );
}

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = Math.floor(seconds % 60);

  if (h > 0) {
    return `${h}h ${m}m ${s}s`;
  } else if (m > 0) {
    return `${m}m ${s}s`;
  }
  return `${s}s`;
}
