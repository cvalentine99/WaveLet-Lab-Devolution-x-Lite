import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Collapsible, CollapsibleContent, CollapsibleTrigger } from '@/components/ui/collapsible';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import {
  Satellite, MapPin, Clock, Lock, Unlock, Activity, AlertTriangle,
  Cpu, Server, ChevronDown, ChevronRight, RefreshCw, Gauge
} from 'lucide-react';
import { useState } from 'react';
import type { GPSStatus, SystemHealth } from '@/lib/api';

export interface SystemStatusPanelProps {
  gpsStatus: GPSStatus | null;
  systemHealth: SystemHealth | null;
  pllLocked: boolean;
  uptimeSeconds: number;
  overflowCount: number;
  reconnectCount: number;
  temperatureC: number;
  className?: string;
}

function formatUptime(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`;
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.floor(seconds % 60)}s`;
  const hours = Math.floor(seconds / 3600);
  const mins = Math.floor((seconds % 3600) / 60);
  return `${hours}h ${mins}m`;
}

function formatCoordinate(value: number | null, type: 'lat' | 'lon'): string {
  if (value === null) return '--';
  const abs = Math.abs(value);
  const deg = Math.floor(abs);
  const min = ((abs - deg) * 60).toFixed(4);
  const dir = type === 'lat'
    ? (value >= 0 ? 'N' : 'S')
    : (value >= 0 ? 'E' : 'W');
  return `${deg}°${min}'${dir}`;
}

/**
 * System Status Panel
 * Displays GPS, Hardware, and System health information
 */
export function SystemStatusPanel({
  gpsStatus,
  systemHealth,
  pllLocked,
  uptimeSeconds,
  overflowCount,
  reconnectCount,
  temperatureC,
  className = '',
}: SystemStatusPanelProps) {
  const [gpsOpen, setGpsOpen] = useState(true);
  const [hardwareOpen, setHardwareOpen] = useState(true);
  const [systemOpen, setSystemOpen] = useState(false);

  const gpsLocked = gpsStatus?.locked ?? false;
  const gpsSatellites = gpsStatus?.satellites ?? 0;
  const gpsEnabled = gpsStatus?.enabled ?? false;

  return (
    <TooltipProvider>
      <Card className={`p-3 ${className}`}>
        <div className="space-y-2">
          {/* Header */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Server className="w-4 h-4 text-primary" />
              <h3 className="text-sm font-semibold">System Status</h3>
            </div>
            {systemHealth && (
              <Badge
                variant={systemHealth.status === 'healthy' ? 'default' :
                         systemHealth.status === 'degraded' ? 'secondary' : 'destructive'}
                className="text-[10px]"
              >
                {systemHealth.status}
              </Badge>
            )}
          </div>

          {/* GPS Section */}
          <Collapsible open={gpsOpen} onOpenChange={setGpsOpen}>
            <CollapsibleTrigger className="flex items-center gap-2 w-full text-left py-1 hover:bg-muted/50 rounded px-1 -mx-1">
              {gpsOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <Satellite className={`w-3 h-3 ${gpsLocked ? 'text-green-500' : 'text-muted-foreground'}`} />
              <span className="text-xs font-medium">GPS</span>
              <span className="ml-auto flex items-center gap-1">
                {gpsEnabled ? (
                  gpsLocked ? (
                    <Badge variant="default" className="text-[9px] h-4 px-1">
                      {gpsSatellites} sats
                    </Badge>
                  ) : (
                    <Badge variant="secondary" className="text-[9px] h-4 px-1">
                      Searching
                    </Badge>
                  )
                ) : (
                  <Badge variant="outline" className="text-[9px] h-4 px-1">
                    Disabled
                  </Badge>
                )}
              </span>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2 pl-5 space-y-2">
              {gpsStatus ? (
                <>
                  {/* Lock Status */}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Lock</span>
                    <span className={`flex items-center gap-1 font-mono ${gpsLocked ? 'text-green-400' : 'text-yellow-400'}`}>
                      {gpsLocked ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                      {gpsStatus.fix_type.toUpperCase()}
                    </span>
                  </div>

                  {/* Position */}
                  {gpsLocked && gpsStatus.latitude !== null && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground flex items-center gap-1">
                        <MapPin className="w-3 h-3" /> Position
                      </span>
                      <span className="font-mono text-[10px]">
                        {formatCoordinate(gpsStatus.latitude, 'lat')}, {formatCoordinate(gpsStatus.longitude, 'lon')}
                      </span>
                    </div>
                  )}

                  {/* Satellites */}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Satellites</span>
                    <span className="font-mono">{gpsSatellites}</span>
                  </div>

                  {/* Time */}
                  {gpsStatus.time_utc && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground flex items-center gap-1">
                        <Clock className="w-3 h-3" /> UTC
                      </span>
                      <span className="font-mono text-[10px]">{gpsStatus.time_utc}</span>
                    </div>
                  )}

                  {/* HDOP */}
                  {gpsStatus.hdop != null && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">HDOP</span>
                      <span className={`font-mono ${gpsStatus.hdop < 2 ? 'text-green-400' : gpsStatus.hdop < 5 ? 'text-yellow-400' : 'text-red-400'}`}>
                        {gpsStatus.hdop.toFixed(1)}
                      </span>
                    </div>
                  )}
                </>
              ) : (
                <div className="text-xs text-muted-foreground">GPS data not available</div>
              )}
            </CollapsibleContent>
          </Collapsible>

          {/* Hardware Section */}
          <Collapsible open={hardwareOpen} onOpenChange={setHardwareOpen}>
            <CollapsibleTrigger className="flex items-center gap-2 w-full text-left py-1 hover:bg-muted/50 rounded px-1 -mx-1">
              {hardwareOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <Activity className={`w-3 h-3 ${pllLocked ? 'text-green-500' : 'text-red-500'}`} />
              <span className="text-xs font-medium">Hardware</span>
              <span className="ml-auto">
                {pllLocked ? (
                  <Badge variant="default" className="text-[9px] h-4 px-1">PLL OK</Badge>
                ) : (
                  <Badge variant="destructive" className="text-[9px] h-4 px-1">PLL UNLOCK</Badge>
                )}
              </span>
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2 pl-5 space-y-2">
              {/* PLL Lock */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">PLL Lock</span>
                <span className={`flex items-center gap-1 font-mono ${pllLocked ? 'text-green-400' : 'text-red-400'}`}>
                  {pllLocked ? <Lock className="w-3 h-3" /> : <Unlock className="w-3 h-3" />}
                  {pllLocked ? 'Locked' : 'Unlocked'}
                </span>
              </div>

              {/* Uptime */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground flex items-center gap-1">
                  <Clock className="w-3 h-3" /> Uptime
                </span>
                <span className="font-mono">{formatUptime(uptimeSeconds)}</span>
              </div>

              {/* Temperature */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Temperature</span>
                <span className={`font-mono ${
                  (temperatureC ?? 0) < 55 ? 'text-blue-400' :
                  (temperatureC ?? 0) < 70 ? 'text-yellow-400' : 'text-red-400'
                }`}>
                  {(temperatureC ?? 0).toFixed(0)}°C
                </span>
              </div>

              {/* Overflow Events */}
              <div className="flex items-center justify-between text-xs">
                <span className="text-muted-foreground">Overflow Events</span>
                <span className={`font-mono ${overflowCount > 0 ? 'text-yellow-400' : 'text-green-400'}`}>
                  {overflowCount}
                </span>
              </div>

              {/* Reconnects */}
              {reconnectCount > 0 && (
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground flex items-center gap-1">
                    <RefreshCw className="w-3 h-3" /> Reconnects
                  </span>
                  <span className="font-mono text-yellow-400">{reconnectCount}</span>
                </div>
              )}
            </CollapsibleContent>
          </Collapsible>

          {/* System Section */}
          <Collapsible open={systemOpen} onOpenChange={setSystemOpen}>
            <CollapsibleTrigger className="flex items-center gap-2 w-full text-left py-1 hover:bg-muted/50 rounded px-1 -mx-1">
              {systemOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
              <Cpu className="w-3 h-3 text-purple-500" />
              <span className="text-xs font-medium">System</span>
              {systemHealth?.gpu_name && (
                <span className="ml-auto text-[10px] text-muted-foreground font-mono truncate max-w-[120px]">
                  {systemHealth.gpu_name.replace('NVIDIA ', '').replace('GeForce ', '')}
                </span>
              )}
            </CollapsibleTrigger>
            <CollapsibleContent className="pt-2 pl-5 space-y-2">
              {systemHealth ? (
                <>
                  {/* GPU */}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">GPU</span>
                    <span className="font-mono text-[10px] truncate max-w-[140px]">{systemHealth.gpu_name}</span>
                  </div>

                  {/* GPU Memory */}
                  {systemHealth.gpu_memory_used_gb != null && systemHealth.gpu_memory_total_gb != null && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground">GPU Memory</span>
                      <span className="font-mono">
                        {systemHealth.gpu_memory_used_gb.toFixed(1)} / {systemHealth.gpu_memory_total_gb.toFixed(0)} GB
                      </span>
                    </div>
                  )}

                  {/* GPU Utilization */}
                  {systemHealth.gpu_utilization_percent != null && (
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-muted-foreground flex items-center gap-1">
                        <Gauge className="w-3 h-3" /> GPU Util
                      </span>
                      <span className={`font-mono ${
                        systemHealth.gpu_utilization_percent < 50 ? 'text-green-400' :
                        systemHealth.gpu_utilization_percent < 80 ? 'text-yellow-400' : 'text-red-400'
                      }`}>
                        {systemHealth.gpu_utilization_percent.toFixed(0)}%
                      </span>
                    </div>
                  )}

                  {/* CUDA Version */}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">CUDA</span>
                    <span className="font-mono text-[10px]">{systemHealth.cuda_version}</span>
                  </div>

                  {/* Version */}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Version</span>
                    <span className="font-mono text-[10px]">{systemHealth.version}</span>
                  </div>

                  {/* Pipeline State */}
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Pipeline</span>
                    <Badge
                      variant={systemHealth.pipeline_state === 'running' ? 'default' :
                               systemHealth.pipeline_state === 'idle' ? 'secondary' :
                               systemHealth.pipeline_state === 'paused' ? 'outline' : 'destructive'}
                      className="text-[9px] h-4"
                    >
                      {systemHealth.pipeline_state}
                    </Badge>
                  </div>
                </>
              ) : (
                <div className="text-xs text-muted-foreground">System data not available</div>
              )}
            </CollapsibleContent>
          </Collapsible>
        </div>
      </Card>
    </TooltipProvider>
  );
}
