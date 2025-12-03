import { useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { Progress } from '@/components/ui/progress';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from '@/components/ui/tooltip';
import {
  Brain,
  Activity,
  AlertTriangle,
  TrendingUp,
  BarChart3,
  Sparkles,
  Target,
  Layers
} from 'lucide-react';
import { useDetectionStore } from '@/stores/detectionStore';
import { useClusterStore } from '@/stores/clusterStore';
import { useShallow } from 'zustand/react/shallow';
import { cn } from '@/lib/utils';
import type { Detection } from '@/types';

export interface MLClassificationDashboardProps {
  maxHeight?: number;
  className?: string;
}

// Color mapping for modulation types
const MODULATION_COLORS: Record<string, string> = {
  'BPSK': '#3b82f6',
  'QPSK': '#8b5cf6',
  '8PSK': '#a855f7',
  '16QAM': '#ec4899',
  '64QAM': '#f43f5e',
  'FSK': '#f59e0b',
  'GFSK': '#eab308',
  'MSK': '#84cc16',
  'OFDM': '#22c55e',
  'LoRa': '#14b8a6',
  'BLE': '#06b6d4',
  'WiFi': '#0ea5e9',
  'LTE': '#6366f1',
  '5G-NR': '#8b5cf6',
  'Unknown': '#6b7280',
};

function getModulationColor(modType: string): string {
  return MODULATION_COLORS[modType] || MODULATION_COLORS['Unknown'];
}

/**
 * ML Classification Dashboard
 * Real-time view of AMC (Automatic Modulation Classification) results
 */
export function MLClassificationDashboard({
  maxHeight = 500,
  className
}: MLClassificationDashboardProps) {
  // Get detections from store
  const detections = useDetectionStore(
    useShallow(state => Array.from(state.detections.values()))
  );

  const clusters = useClusterStore(
    useShallow(state => Array.from(state.clusters.values()))
  );

  // Compute classification statistics
  const stats = useMemo(() => {
    const classified = detections.filter(d => d.modulationType);
    const withConfidence = classified.filter(d => d.confidence !== undefined);
    const anomalous = detections.filter(d => d.anomalyScore !== undefined && d.anomalyScore > 0.5);
    const highConfidence = withConfidence.filter(d => (d.confidence || 0) > 0.8);

    // Count by modulation type
    const modCounts: Record<string, number> = {};
    classified.forEach(d => {
      const mod = d.modulationType || 'Unknown';
      modCounts[mod] = (modCounts[mod] || 0) + 1;
    });

    // Sort by count
    const sortedMods = Object.entries(modCounts)
      .sort((a, b) => b[1] - a[1]);

    // Average confidence
    const avgConfidence = withConfidence.length > 0
      ? withConfidence.reduce((sum, d) => sum + (d.confidence || 0), 0) / withConfidence.length
      : 0;

    // Average anomaly score
    const avgAnomalyScore = anomalous.length > 0
      ? anomalous.reduce((sum, d) => sum + (d.anomalyScore || 0), 0) / anomalous.length
      : 0;

    return {
      total: detections.length,
      classified: classified.length,
      unclassified: detections.length - classified.length,
      anomalous: anomalous.length,
      highConfidence: highConfidence.length,
      avgConfidence,
      avgAnomalyScore,
      modCounts: sortedMods,
      classificationRate: detections.length > 0 ? (classified.length / detections.length) * 100 : 0,
    };
  }, [detections]);

  // Recent high-confidence classifications
  const recentClassifications = useMemo(() => {
    return detections
      .filter(d => d.modulationType && d.confidence !== undefined)
      .sort((a, b) => b.timestamp - a.timestamp)
      .slice(0, 8);
  }, [detections]);

  // Recent anomalies
  const recentAnomalies = useMemo(() => {
    return detections
      .filter(d => d.anomalyScore !== undefined && d.anomalyScore > 0.5)
      .sort((a, b) => (b.anomalyScore || 0) - (a.anomalyScore || 0))
      .slice(0, 5);
  }, [detections]);

  return (
    <TooltipProvider>
      <Card className={cn('p-4', className)}>
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Brain className="w-5 h-5 text-purple-500" />
            <h3 className="text-sm font-semibold">ML Classification</h3>
          </div>
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-[10px]">
              AMC Engine
            </Badge>
            {stats.anomalous > 0 && (
              <Badge variant="destructive" className="text-[10px] gap-1">
                <AlertTriangle className="w-3 h-3" />
                {stats.anomalous} anomalies
              </Badge>
            )}
          </div>
        </div>

        <ScrollArea style={{ maxHeight }}>
          <div className="space-y-4">
            {/* Summary Stats */}
            <div className="grid grid-cols-4 gap-2">
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-2 rounded-lg bg-muted/50 border text-center">
                    <div className="text-lg font-mono font-semibold text-blue-400">
                      {stats.classified}
                    </div>
                    <div className="text-[10px] text-muted-foreground">Classified</div>
                  </div>
                </TooltipTrigger>
                <TooltipContent>Signals with modulation classification</TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-2 rounded-lg bg-muted/50 border text-center">
                    <div className="text-lg font-mono font-semibold text-green-400">
                      {(stats.avgConfidence * 100).toFixed(0)}%
                    </div>
                    <div className="text-[10px] text-muted-foreground">Avg Conf</div>
                  </div>
                </TooltipTrigger>
                <TooltipContent>Average classification confidence</TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-2 rounded-lg bg-muted/50 border text-center">
                    <div className="text-lg font-mono font-semibold text-purple-400">
                      {stats.modCounts.length}
                    </div>
                    <div className="text-[10px] text-muted-foreground">Types</div>
                  </div>
                </TooltipTrigger>
                <TooltipContent>Unique modulation types detected</TooltipContent>
              </Tooltip>

              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="p-2 rounded-lg bg-muted/50 border text-center">
                    <div className={cn('text-lg font-mono font-semibold',
                      stats.anomalous > 0 ? 'text-red-400' : 'text-green-400'
                    )}>
                      {stats.anomalous}
                    </div>
                    <div className="text-[10px] text-muted-foreground">Anomalies</div>
                  </div>
                </TooltipTrigger>
                <TooltipContent>High anomaly score detections</TooltipContent>
              </Tooltip>
            </div>

            {/* Classification Rate */}
            <div className="space-y-1.5">
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center gap-1.5 text-muted-foreground">
                  <Target className="w-3.5 h-3.5" />
                  <span>Classification Rate</span>
                </div>
                <span className="font-mono">{stats.classificationRate.toFixed(1)}%</span>
              </div>
              <Progress value={stats.classificationRate} className="h-2" />
            </div>

            {/* Modulation Distribution */}
            <div className="space-y-2">
              <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <BarChart3 className="w-3.5 h-3.5" />
                <span>Modulation Distribution</span>
              </div>

              {stats.modCounts.length === 0 ? (
                <div className="text-center py-4 text-muted-foreground text-xs">
                  <Layers className="w-6 h-6 mx-auto mb-1 opacity-50" />
                  No classifications yet
                </div>
              ) : (
                <div className="space-y-1.5">
                  {stats.modCounts.slice(0, 6).map(([mod, count]) => {
                    const percent = (count / stats.classified) * 100;
                    const color = getModulationColor(mod);

                    return (
                      <div key={mod} className="space-y-1">
                        <div className="flex items-center justify-between text-xs">
                          <div className="flex items-center gap-2">
                            <div
                              className="w-2.5 h-2.5 rounded-sm"
                              style={{ backgroundColor: color }}
                            />
                            <span className="font-medium">{mod}</span>
                          </div>
                          <span className="font-mono text-muted-foreground">
                            {count} ({percent.toFixed(1)}%)
                          </span>
                        </div>
                        <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full rounded-full transition-all duration-300"
                            style={{
                              width: `${percent}%`,
                              backgroundColor: color,
                            }}
                          />
                        </div>
                      </div>
                    );
                  })}
                  {stats.modCounts.length > 6 && (
                    <div className="text-[10px] text-muted-foreground text-center pt-1">
                      +{stats.modCounts.length - 6} more types
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Recent Classifications */}
            {recentClassifications.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Recent Classifications</span>
                </div>

                <div className="space-y-1">
                  {recentClassifications.map((det) => (
                    <RecentClassificationRow key={det.id} detection={det} />
                  ))}
                </div>
              </div>
            )}

            {/* Anomaly Alerts */}
            {recentAnomalies.length > 0 && (
              <div className="space-y-2">
                <div className="flex items-center gap-1.5 text-xs text-red-400">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  <span>Anomaly Alerts</span>
                </div>

                <div className="space-y-1">
                  {recentAnomalies.map((det) => (
                    <AnomalyAlertRow key={det.id} detection={det} />
                  ))}
                </div>
              </div>
            )}

            {/* Cluster Signal Types */}
            {clusters.some(c => c.signalTypeHint) && (
              <div className="space-y-2 pt-2 border-t">
                <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
                  <Layers className="w-3.5 h-3.5" />
                  <span>Cluster Signal Types</span>
                </div>

                <div className="flex flex-wrap gap-1.5">
                  {clusters
                    .filter(c => c.signalTypeHint)
                    .slice(0, 8)
                    .map(cluster => (
                      <Badge
                        key={cluster.id}
                        variant="outline"
                        className="text-[10px] gap-1"
                        style={{
                          borderColor: cluster.color,
                          color: cluster.color,
                        }}
                      >
                        {cluster.signalTypeHint}
                        <span className="text-muted-foreground">({cluster.size})</span>
                      </Badge>
                    ))}
                </div>
              </div>
            )}
          </div>
        </ScrollArea>
      </Card>
    </TooltipProvider>
  );
}

/**
 * Recent Classification Row
 */
function RecentClassificationRow({ detection }: { detection: Detection }) {
  const color = getModulationColor(detection.modulationType || 'Unknown');
  const confidence = (detection.confidence || 0) * 100;

  return (
    <div className="flex items-center justify-between p-1.5 rounded bg-muted/30 text-xs">
      <div className="flex items-center gap-2">
        <div
          className="w-2 h-2 rounded-full"
          style={{ backgroundColor: color }}
        />
        <span className="font-mono text-[11px]">
          {(detection.centerFreqHz / 1e6).toFixed(3)} MHz
        </span>
        <Badge
          variant="outline"
          className="text-[9px] px-1 py-0 h-4"
          style={{ borderColor: color, color }}
        >
          {detection.modulationType}
        </Badge>
      </div>
      <div className="flex items-center gap-1.5">
        <div className="w-10 h-1 bg-muted rounded-full overflow-hidden">
          <div
            className="h-full rounded-full"
            style={{
              width: `${confidence}%`,
              backgroundColor: confidence > 80 ? '#22c55e' : confidence > 60 ? '#eab308' : '#ef4444'
            }}
          />
        </div>
        <span className={cn('font-mono text-[10px] w-8 text-right',
          confidence > 80 ? 'text-green-400' :
          confidence > 60 ? 'text-yellow-400' : 'text-red-400'
        )}>
          {confidence.toFixed(0)}%
        </span>
      </div>
    </div>
  );
}

/**
 * Anomaly Alert Row
 */
function AnomalyAlertRow({ detection }: { detection: Detection }) {
  const score = (detection.anomalyScore || 0) * 100;
  const severity = score > 80 ? 'critical' : score > 60 ? 'high' : 'medium';

  return (
    <div className={cn('flex items-center justify-between p-1.5 rounded text-xs',
      severity === 'critical' ? 'bg-red-500/10' :
      severity === 'high' ? 'bg-orange-500/10' : 'bg-yellow-500/10'
    )}>
      <div className="flex items-center gap-2">
        <AlertTriangle className={cn('w-3 h-3',
          severity === 'critical' ? 'text-red-400' :
          severity === 'high' ? 'text-orange-400' : 'text-yellow-400'
        )} />
        <span className="font-mono text-[11px]">
          {(detection.centerFreqHz / 1e6).toFixed(3)} MHz
        </span>
        {detection.modulationType && (
          <span className="text-muted-foreground text-[10px]">
            {detection.modulationType}
          </span>
        )}
      </div>
      <Badge
        variant="outline"
        className={cn('text-[9px] px-1.5 py-0 h-4',
          severity === 'critical' ? 'border-red-500/50 text-red-400' :
          severity === 'high' ? 'border-orange-500/50 text-orange-400' :
          'border-yellow-500/50 text-yellow-400'
        )}
      >
        {score.toFixed(0)}% anomaly
      </Badge>
    </div>
  );
}
