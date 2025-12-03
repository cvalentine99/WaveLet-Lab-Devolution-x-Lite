import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import { AlertTriangle, Thermometer, Activity, Database, Waves } from 'lucide-react';
import { cn } from '@/lib/utils';

interface SDRMetricsBarProps {
  isConnected: boolean;
  isStreaming: boolean;
  dropRatePercent: number;
  bufferFillPercent: number;
  temperatureC: number;
  healthStatus?: 'healthy' | 'degraded' | 'unhealthy';
  warnings?: string[];
  className?: string;
}

export function SDRMetricsBar({
  isConnected,
  isStreaming,
  dropRatePercent,
  bufferFillPercent,
  temperatureC,
  healthStatus = 'healthy',
  warnings = [],
  className,
}: SDRMetricsBarProps) {
  const getHealthColor = () => {
    switch (healthStatus) {
      case 'healthy': return 'bg-green-500';
      case 'degraded': return 'bg-yellow-500';
      case 'unhealthy': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const getDropRateColor = () => {
    if (dropRatePercent > 5) return 'text-red-500';
    if (dropRatePercent > 1) return 'text-yellow-500';
    return 'text-green-500';
  };

  const getBufferColor = () => {
    if (bufferFillPercent > 80) return 'text-red-500';
    if (bufferFillPercent > 60) return 'text-yellow-500';
    return 'text-emerald-500';
  };

  const getTempColor = () => {
    if (temperatureC > 70) return 'text-red-500';
    if (temperatureC > 55) return 'text-yellow-500';
    return 'text-blue-400';
  };

  if (!isConnected) {
    return (
      <div className={cn('flex items-center gap-2 text-xs text-muted-foreground', className)}>
        <Badge variant="outline" className="text-[10px]">Disconnected</Badge>
      </div>
    );
  }

  return (
    <TooltipProvider>
      <div className={cn('flex items-center gap-3 text-[10px]', className)}>
        {/* Health Status */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1">
              <div className={cn('w-2 h-2 rounded-full', getHealthColor())} />
              <span className="text-muted-foreground capitalize">{healthStatus}</span>
              {warnings.length > 0 && (
                <AlertTriangle className="w-3 h-3 text-yellow-500" />
              )}
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="max-w-[250px]">
            <p className="font-medium">SDR Health: {healthStatus}</p>
            {warnings.length > 0 && (
              <ul className="mt-1 text-[10px] text-yellow-400">
                {warnings.map((w, i) => <li key={i}>{w}</li>)}
              </ul>
            )}
          </TooltipContent>
        </Tooltip>

        {/* Streaming indicator */}
        {isStreaming && (
          <div className="flex items-center gap-1">
            <Waves className="w-3 h-3 text-green-500 animate-pulse" />
            <span className="text-green-500">Live</span>
          </div>
        )}

        {/* Drop Rate */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className={cn('flex items-center gap-1 font-mono', getDropRateColor())}>
              <Activity className="w-3 h-3" />
              <span>{dropRatePercent.toFixed(1)}%</span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>Sample Drop Rate</p>
            <p className="text-[10px] text-muted-foreground">Target: &lt;1%</p>
          </TooltipContent>
        </Tooltip>

        {/* Buffer Fill */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className="flex items-center gap-1">
              <Database className="w-3 h-3 text-muted-foreground" />
              <div className="w-12 h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                  className={cn('h-full transition-all',
                    bufferFillPercent > 80 ? 'bg-red-500' :
                    bufferFillPercent > 60 ? 'bg-yellow-500' : 'bg-emerald-500'
                  )}
                  style={{ width: `${Math.min(bufferFillPercent, 100)}%` }}
                />
              </div>
              <span className={cn('font-mono', getBufferColor())}>
                {bufferFillPercent.toFixed(0)}%
              </span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>Ring Buffer Fill Level</p>
            <p className="text-[10px] text-muted-foreground">Backpressure at 75%</p>
          </TooltipContent>
        </Tooltip>

        {/* Temperature */}
        <Tooltip>
          <TooltipTrigger asChild>
            <div className={cn('flex items-center gap-1 font-mono', getTempColor())}>
              <Thermometer className="w-3 h-3" />
              <span>{temperatureC.toFixed(0)}°C</span>
            </div>
          </TooltipTrigger>
          <TooltipContent side="bottom">
            <p>LMS7002M Temperature</p>
            <p className="text-[10px] text-muted-foreground">Safe: &lt;70°C</p>
          </TooltipContent>
        </Tooltip>
      </div>
    </TooltipProvider>
  );
}
