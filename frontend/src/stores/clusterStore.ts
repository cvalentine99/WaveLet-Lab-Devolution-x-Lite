import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { Cluster } from '@/types';

/**
 * Cluster Store
 * Manages emitter clusters from cuML DBSCAN
 */

interface ClusterState {
  // Cluster storage
  clusters: Map<number, Cluster>;
  
  // Selection
  selectedClusterId: number | null;
  
  // Cluster colors (auto-assigned)
  clusterColors: Map<number, string>;
  
  // Actions
  updateCluster: (cluster: Cluster) => void;
  updateClusters: (clusters: Cluster[]) => void;
  removeCluster: (id: number) => void;
  selectCluster: (id: number | null) => void;
  labelCluster: (id: number, label: string) => void;
  mergeClusters: (ids: number[], newLabel?: string) => void;
  clearClusters: () => void;
  
  // Getters
  getCluster: (id: number) => Cluster | undefined;
  getSortedClusters: (sortBy: 'size' | 'frequency' | 'snr') => Cluster[];
  getClusterColor: (id: number) => string;
}

// Predefined color palette for clusters
const CLUSTER_COLORS = [
  '#10b981', // green
  '#3b82f6', // blue
  '#f59e0b', // amber
  '#ef4444', // red
  '#8b5cf6', // purple
  '#ec4899', // pink
  '#06b6d4', // cyan
  '#f97316', // orange
  '#84cc16', // lime
  '#6366f1', // indigo
];

function generateClusterColor(id: number): string {
  return CLUSTER_COLORS[id % CLUSTER_COLORS.length];
}

export const useClusterStore = create<ClusterState>()(
  immer((set, get) => ({
    // Initial state
    clusters: new Map(),
    selectedClusterId: null,
    clusterColors: new Map(),
    
    // Update single cluster
    updateCluster: (cluster: Cluster) => {
      set((state) => {
        state.clusters.set(cluster.id, cluster);
        
        // Assign color if not exists
        if (!state.clusterColors.has(cluster.id)) {
          state.clusterColors.set(cluster.id, cluster.color || generateClusterColor(cluster.id));
        }
      });
    },
    
    // Update multiple clusters (bulk update)
    updateClusters: (clusters: Cluster[]) => {
      set((state) => {
        clusters.forEach((cluster) => {
          state.clusters.set(cluster.id, cluster);
          
          if (!state.clusterColors.has(cluster.id)) {
            state.clusterColors.set(cluster.id, cluster.color || generateClusterColor(cluster.id));
          }
        });
      });
    },
    
    // Remove cluster
    removeCluster: (id: number) => {
      set((state) => {
        state.clusters.delete(id);
        state.clusterColors.delete(id);
        
        if (state.selectedClusterId === id) {
          state.selectedClusterId = null;
        }
      });
    },
    
    // Select cluster
    selectCluster: (id: number | null) => {
      set((state) => {
        state.selectedClusterId = id;
      });
    },
    
    // Label cluster
    labelCluster: (id: number, label: string) => {
      set((state) => {
        const cluster = state.clusters.get(id);
        if (cluster) {
          cluster.label = label;
        }
      });
    },
    
    // Merge clusters
    mergeClusters: (ids: number[], newLabel?: string) => {
      set((state) => {
        if (ids.length < 2) return;
        
        // Get all clusters to merge
        const clustersToMerge = ids
          .map((id) => state.clusters.get(id))
          .filter((c): c is Cluster => c !== undefined);
        
        if (clustersToMerge.length < 2) return;
        
        // Create merged cluster
        const mergedId = Math.min(...ids);
        const totalSize = clustersToMerge.reduce((sum, c) => sum + c.size, 0);
        const avgSnr = clustersToMerge.reduce((sum, c) => sum + c.avgSnrDb * c.size, 0) / totalSize;
        
        // Compute weighted centroid
        const centroidDim = clustersToMerge[0].centroid.length;
        const mergedCentroid = new Array(centroidDim).fill(0);
        
        clustersToMerge.forEach((cluster) => {
          cluster.centroid.forEach((val, i) => {
            mergedCentroid[i] += val * cluster.size;
          });
        });
        
        mergedCentroid.forEach((val, i) => {
          mergedCentroid[i] = val / totalSize;
        });
        
        const mergedCluster: Cluster = {
          id: mergedId,
          size: totalSize,
          centroid: mergedCentroid,
          avgSnrDb: avgSnr,
          dominantFreqHz: clustersToMerge[0].dominantFreqHz,
          label: newLabel || `Merged Cluster ${mergedId}`,
          color: state.clusterColors.get(mergedId) || generateClusterColor(mergedId),
        };
        
        // Remove old clusters and add merged
        ids.forEach((id) => {
          state.clusters.delete(id);
          if (id !== mergedId) {
            state.clusterColors.delete(id);
          }
        });
        
        state.clusters.set(mergedId, mergedCluster);
      });
    },
    
    // Clear all clusters
    clearClusters: () => {
      set((state) => {
        state.clusters.clear();
        state.clusterColors.clear();
        state.selectedClusterId = null;
      });
    },
    
    // Get cluster by ID
    getCluster: (id: number) => {
      return get().clusters.get(id);
    },
    
    // Get sorted clusters
    getSortedClusters: (sortBy: 'size' | 'frequency' | 'snr') => {
      const clusters = Array.from(get().clusters.values());
      
      switch (sortBy) {
        case 'size':
          return clusters.sort((a, b) => b.size - a.size);
        case 'frequency':
          return clusters.sort((a, b) => a.dominantFreqHz - b.dominantFreqHz);
        case 'snr':
          return clusters.sort((a, b) => b.avgSnrDb - a.avgSnrDb);
        default:
          return clusters;
      }
    },
    
    // Get cluster color
    getClusterColor: (id: number) => {
      const state = get();
      return state.clusterColors.get(id) || generateClusterColor(id);
    },
  }))
);

// Selectors
export const selectClusters = (state: ClusterState) => state.clusters;
export const selectSelectedCluster = (state: ClusterState) =>
  state.selectedClusterId !== null
    ? state.clusters.get(state.selectedClusterId)
    : null;
export const selectClusterCount = (state: ClusterState) => state.clusters.size;
