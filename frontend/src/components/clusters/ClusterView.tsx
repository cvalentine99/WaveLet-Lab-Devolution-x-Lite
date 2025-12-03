import { useMemo, useCallback } from 'react';
import { useClusterStore } from '@/stores/clusterStore';
import { PerformanceProfiler } from '@/components/performance/PerformanceProfiler';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Network, Users } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import type { Cluster } from '@/types';

export interface ClusterViewProps {
  maxHeight?: number;
}

/**
 * Cluster View Component
 * Displays cuML DBSCAN clustering results
 */
export function ClusterView({ maxHeight = 600 }: ClusterViewProps) {
  return (
    <PerformanceProfiler id="ClusterView" slowThresholdMs={16}>
      <ClusterViewInner maxHeight={maxHeight} />
    </PerformanceProfiler>
  );
}

// Stable selector for sorted clusters - returns the data directly sorted by size
const selectClustersSortedBySize = (state: { clusters: Map<number, Cluster> }): Cluster[] => {
  return Array.from(state.clusters.values()).sort((a, b) => b.size - a.size);
};

function ClusterViewInner({ maxHeight = 600 }: ClusterViewProps) {
  // Use useShallow for array results to prevent infinite loops
  const clusters = useClusterStore(useShallow(selectClustersSortedBySize));
  const selectedClusterId = useClusterStore((state) => state.selectedClusterId);
  const selectCluster = useClusterStore((state) => state.selectCluster);
  
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

  const totalSignals = useMemo(() => {
    return clusters.reduce((sum, cluster) => sum + cluster.size, 0);
  }, [clusters]);

  const handleClusterClick = useCallback(
    (clusterId: number) => {
      selectCluster(clusterId === selectedClusterId ? null : clusterId);
    },
    [selectCluster, selectedClusterId]
  );

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Network className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold">Emitter Clusters</h3>
          <Badge variant="secondary" className="text-xs">
            {clusters.length}
          </Badge>
        </div>
        <div className="text-xs text-muted-foreground">
          <Users className="w-3 h-3 inline mr-1" />
          {totalSignals} signals
        </div>
      </div>

      <ScrollArea style={{ maxHeight }}>
        <div className="space-y-2">
          {clusters.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-sm">
              <Network className="w-8 h-8 mx-auto mb-2 opacity-50" />
              No clusters detected
            </div>
          ) : (
            clusters.map((cluster) => (
              <ClusterCard
                key={cluster.id}
                cluster={cluster}
                isSelected={cluster.id === selectedClusterId}
                color={getClusterColor(cluster.id)}
                onClick={() => handleClusterClick(cluster.id)}
              />
            ))
          )}
        </div>
      </ScrollArea>
    </Card>
  );
}

/**
 * Cluster Card
 */
interface ClusterCardProps {
  cluster: {
    id: number;
    size: number;
    avgSnrDb: number;
    dominantFreqHz: number;
    label?: string;
    signalTypeHint?: string;
    avgDutyCycle?: number;
    uniqueTracks?: number;
  };
  isSelected: boolean;
  color: string;
  onClick: () => void;
}

function ClusterCard({ cluster, isSelected, color, onClick }: ClusterCardProps) {
  return (
    <div
      className={`
        p-3 rounded-lg border cursor-pointer transition-all
        ${isSelected ? 'border-primary bg-primary/10' : 'border-border hover:border-primary/50'}
      `}
      onClick={onClick}
      style={{
        borderLeftWidth: '4px',
        borderLeftColor: color,
      }}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <div
            className="w-3 h-3 rounded-full"
            style={{ backgroundColor: color }}
          />
          <span className="font-medium text-sm">
            {cluster.label || `Cluster ${cluster.id}`}
          </span>
          {cluster.signalTypeHint && (
            <Badge variant="outline" className="text-[10px] px-1.5 py-0 h-4 bg-purple-500/10 text-purple-400 border-purple-500/30">
              {cluster.signalTypeHint}
            </Badge>
          )}
        </div>
        <Badge variant="secondary" className="text-xs">
          {cluster.size} signals
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs">
        <div>
          <span className="text-muted-foreground">Freq:</span>{' '}
          <span className="font-mono">{(cluster.dominantFreqHz / 1e6).toFixed(3)} MHz</span>
        </div>
        <div>
          <span className="text-muted-foreground">Avg SNR:</span>{' '}
          <span className="font-mono">{cluster.avgSnrDb.toFixed(1)} dB</span>
        </div>
        {cluster.avgDutyCycle !== undefined && (
          <div>
            <span className="text-muted-foreground">Duty:</span>{' '}
            <span className="font-mono">{(cluster.avgDutyCycle * 100).toFixed(0)}%</span>
          </div>
        )}
        {cluster.uniqueTracks !== undefined && cluster.uniqueTracks > 0 && (
          <div>
            <span className="text-muted-foreground">Tracks:</span>{' '}
            <span className="font-mono">{cluster.uniqueTracks}</span>
          </div>
        )}
      </div>
    </div>
  );
}
