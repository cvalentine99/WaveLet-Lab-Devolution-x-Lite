import { useMemo, useCallback } from 'react';
import { useDetectionStore } from '@/stores/detectionStore';
import { PerformanceProfiler } from '@/components/performance/PerformanceProfiler';
import { useClusterStore } from '@/stores/clusterStore';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import {
  HoverCard,
  HoverCardContent,
  HoverCardTrigger,
} from '@/components/ui/hover-card';
import { Activity, Radio, Signal, Zap, ChevronDown, Bookmark, AlertTriangle } from 'lucide-react';
import type { Detection } from '@/types';
import { useShallow } from 'zustand/react/shallow';

export interface DetectionListProps {
  maxHeight?: number;
  showFilters?: boolean;
  onBookmark?: (detection: Detection) => void;
}

/**
 * Detection List Component
 * Displays live RF signal detections with filtering
 */
export function DetectionList({ maxHeight = 600, showFilters = true, onBookmark }: DetectionListProps) {
  return (
    <PerformanceProfiler id="DetectionList" slowThresholdMs={16}>
      <DetectionListInner maxHeight={maxHeight} showFilters={showFilters} onBookmark={onBookmark} />
    </PerformanceProfiler>
  );
}

// Stable selector for filtered detections - returns the data directly
const selectFilteredDetections = (state: {
  detections: Map<string, Detection>;
  minSnrDb: number;
  minFreqHz: number | null;
  maxFreqHz: number | null;
  modulationFilter: Set<string>;
  showInactive: boolean;
}): Detection[] => {
  let detections = Array.from(state.detections.values());

  // Apply SNR filter
  if (state.minSnrDb > -Infinity) {
    detections = detections.filter((det) => det.snrDb >= state.minSnrDb);
  }

  // Apply frequency range filter
  if (state.minFreqHz !== null) {
    detections = detections.filter((det) => det.centerFreqHz >= state.minFreqHz!);
  }
  if (state.maxFreqHz !== null) {
    detections = detections.filter((det) => det.centerFreqHz <= state.maxFreqHz!);
  }

  // Apply modulation filter
  if (state.modulationFilter.size > 0) {
    detections = detections.filter(
      (det) => det.modulationType && state.modulationFilter.has(det.modulationType)
    );
  }

  // Filter inactive if needed
  if (!state.showInactive) {
    const now = Date.now();
    const threshold = 5000;
    detections = detections.filter((det) => now - det.timestamp < threshold);
  }

  return detections;
};

function DetectionListInner({ maxHeight = 600, showFilters = true, onBookmark }: DetectionListProps) {
  // Use useShallow for array results to prevent infinite loops
  const detections = useDetectionStore(useShallow(selectFilteredDetections));
  const selectedDetectionId = useDetectionStore((state) => state.selectedDetectionId);
  const selectDetection = useDetectionStore((state) => state.selectDetection);
  
  // Select the colors map directly, not the getter function
  const clusterColors = useClusterStore(useShallow((state) => state.clusterColors));

  // Create a stable color getter from the map
  const getClusterColor = useCallback(
    (id: number): string => {
      const CLUSTER_COLORS = [
        '#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6',
        '#ec4899', '#06b6d4', '#f97316', '#84cc16', '#6366f1',
      ];
      return clusterColors.get(id) || CLUSTER_COLORS[id % CLUSTER_COLORS.length];
    },
    [clusterColors]
  );

  // Sort by timestamp (newest first)
  const sortedDetections = useMemo(() => {
    return [...detections].sort((a, b) => b.timestamp - a.timestamp);
  }, [detections]);

  const handleDetectionClick = useCallback(
    (detection: Detection) => {
      selectDetection(detection.id === selectedDetectionId ? null : detection.id);
    },
    [selectDetection, selectedDetectionId]
  );

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold">Live Detections</h3>
          <Badge variant="secondary" className="text-xs">
            {detections.length}
          </Badge>
        </div>
      </div>

      <ScrollArea style={{ maxHeight }}>
        <div className="space-y-2">
          {sortedDetections.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-sm">
              <Radio className="w-8 h-8 mx-auto mb-2 opacity-50" />
              No detections yet
            </div>
          ) : (
            sortedDetections.map((detection) => (
              <DetectionCard
                key={detection.id}
                detection={detection}
                isSelected={detection.id === selectedDetectionId}
                clusterColor={
                  detection.clusterId !== undefined
                    ? getClusterColor(detection.clusterId)
                    : undefined
                }
                onClick={() => handleDetectionClick(detection)}
                onBookmark={onBookmark ? () => onBookmark(detection) : undefined}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </Card>
  );
}

/**
 * Detection Card
 */
interface DetectionCardProps {
  detection: Detection;
  isSelected: boolean;
  clusterColor?: string;
  onClick: () => void;
  onBookmark?: () => void;
}

function DetectionCard({ detection, isSelected, clusterColor, onClick, onBookmark }: DetectionCardProps) {
  const age = Date.now() - detection.timestamp;
  const isRecent = age < 2000;

  return (
    <div
      className={`
        p-3 rounded-lg border cursor-pointer transition-all
        ${isSelected ? 'border-primary bg-primary/10' : 'border-border hover:border-primary/50'}
        ${isRecent ? 'animate-pulse-subtle' : ''}
      `}
      onClick={onClick}
      style={{
        borderLeftWidth: clusterColor ? '4px' : '1px',
        borderLeftColor: clusterColor || undefined,
      }}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <Signal className="w-4 h-4 text-primary" />
          <span className="font-mono text-sm font-medium">
            {(detection.centerFreqHz / 1e6).toFixed(3)} MHz
          </span>
        </div>
        <div className="flex items-center gap-2">
          <div className="flex items-center gap-1">
            <Zap className="w-3 h-3 text-yellow-500" />
            <span className="text-xs font-mono text-muted-foreground">
              {detection.snrDb.toFixed(1)} dB
            </span>
          </div>
          {onBookmark && (
            <button
              onClick={(e) => {
                e.stopPropagation();
                onBookmark();
              }}
              className="p-1 hover:bg-primary/20 rounded transition-colors"
              title="Bookmark this detection"
            >
              <Bookmark className="w-3 h-3 text-muted-foreground hover:text-primary" />
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-muted-foreground">BW:</span>{' '}
          <span className="font-mono">{(detection.bandwidthHz / 1e3).toFixed(1)} kHz</span>
        </div>
        <div>
          <span className="text-muted-foreground">Power:</span>{' '}
          <span className="font-mono">{detection.peakPowerDb.toFixed(1)} dBm</span>
        </div>
      </div>

      {detection.modulationType && (
        <div className="mt-2 flex items-center gap-2">
          {detection.topKPredictions && detection.topKPredictions.length > 1 ? (
            <HoverCard>
              <HoverCardTrigger asChild>
                <div className="flex items-center gap-1 cursor-pointer">
                  <Badge variant="outline" className="text-xs">
                    {detection.modulationType}
                  </Badge>
                  {detection.confidence !== undefined && (
                    <span className="text-xs text-muted-foreground">
                      {(detection.confidence * 100).toFixed(0)}%
                    </span>
                  )}
                  <ChevronDown className="w-3 h-3 text-muted-foreground" />
                </div>
              </HoverCardTrigger>
              <HoverCardContent className="w-48 p-2" side="bottom">
                <div className="space-y-1">
                  <p className="text-xs font-semibold text-muted-foreground mb-2">
                    Classification Results
                  </p>
                  {detection.topKPredictions.map((pred, idx) => (
                    <div
                      key={pred.modulation}
                      className="flex items-center justify-between text-xs"
                    >
                      <span className={idx === 0 ? 'font-medium' : 'text-muted-foreground'}>
                        {pred.modulation}
                      </span>
                      <div className="flex items-center gap-1">
                        <div
                          className="h-1.5 bg-primary rounded-full"
                          style={{ width: `${pred.confidence * 40}px` }}
                        />
                        <span className="text-muted-foreground w-8 text-right">
                          {(pred.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </HoverCardContent>
            </HoverCard>
          ) : (
            <>
              <Badge variant="outline" className="text-xs">
                {detection.modulationType}
              </Badge>
              {detection.confidence !== undefined && (
                <span className="text-xs text-muted-foreground">
                  {(detection.confidence * 100).toFixed(0)}%
                </span>
              )}
            </>
          )}
        </div>
      )}

      {/* Cluster & Anomaly Row */}
      {(detection.clusterId !== undefined || detection.anomalyScore !== undefined) && (
        <div className="mt-2 flex items-center gap-2 flex-wrap">
          {detection.clusterId !== undefined && (
            <Badge
              variant="secondary"
              className="text-xs"
              style={{ backgroundColor: clusterColor, color: 'white' }}
            >
              Cluster {detection.clusterId}
            </Badge>
          )}
          {detection.anomalyScore !== undefined && detection.anomalyScore > 0.5 && (
            <div
              className={`flex items-center gap-1 px-1.5 py-0.5 rounded text-[10px] font-medium ${
                detection.anomalyScore > 0.8
                  ? 'bg-red-500/20 text-red-400'
                  : detection.anomalyScore > 0.6
                    ? 'bg-orange-500/20 text-orange-400'
                    : 'bg-yellow-500/20 text-yellow-400'
              }`}
              title={`Anomaly Score: ${(detection.anomalyScore * 100).toFixed(0)}%`}
            >
              <AlertTriangle className="w-3 h-3" />
              <span>{(detection.anomalyScore * 100).toFixed(0)}%</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
