import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import type { Detection, TrackedSignal } from '@/types';

/**
 * Detection Store
 * Manages detections, tracked signals, and selection state
 */

interface DetectionState {
  // Detection storage
  detections: Map<string, Detection>;
  trackedSignals: Map<string, TrackedSignal>;
  
  // Selection
  selectedDetectionId: string | null;
  selectedSignalId: string | null;
  
  // Filters
  minSnrDb: number;
  minFreqHz: number | null;
  maxFreqHz: number | null;
  modulationFilter: Set<string>;
  showInactive: boolean;
  
  // Actions
  addDetection: (detection: Detection) => void;
  removeDetection: (id: string) => void;
  selectDetection: (id: string | null) => void;
  updateTrackedSignal: (signal: TrackedSignal) => void;
  clearDetections: () => void;
  
  // Filters
  setMinSnr: (snr: number) => void;
  setFrequencyRange: (minHz: number | null, maxHz: number | null) => void;
  toggleModulationFilter: (modulation: string) => void;
  clearFilters: () => void;
  
  // Getters
  getDetection: (id: string) => Detection | undefined;
  getSortedDetections: (sortBy: 'frequency' | 'power' | 'snr' | 'time') => Detection[];
  getActiveDetections: () => Detection[];
  getFilteredDetections: () => Detection[];
}

export const useDetectionStore = create<DetectionState>()(
  immer((set, get) => ({
    // Initial state
    detections: new Map(),
    trackedSignals: new Map(),
    selectedDetectionId: null,
    selectedSignalId: null,
    
    minSnrDb: -Infinity,
    minFreqHz: null,
    maxFreqHz: null,
    modulationFilter: new Set(),
    showInactive: true,
    
    // Add detection
    addDetection: (detection: Detection) => {
      set((state) => {
        state.detections.set(detection.id, detection);
      });
    },
    
    // Remove detection
    removeDetection: (id: string) => {
      set((state) => {
        state.detections.delete(id);
        if (state.selectedDetectionId === id) {
          state.selectedDetectionId = null;
        }
      });
    },
    
    // Select detection
    selectDetection: (id: string | null) => {
      set((state) => {
        state.selectedDetectionId = id;
      });
    },
    
    // Update tracked signal
    updateTrackedSignal: (signal: TrackedSignal) => {
      set((state) => {
        state.trackedSignals.set(signal.id, signal);
      });
    },
    
    // Clear all detections
    clearDetections: () => {
      set((state) => {
        state.detections.clear();
        state.trackedSignals.clear();
        state.selectedDetectionId = null;
        state.selectedSignalId = null;
      });
    },
    
    // Set minimum SNR filter
    setMinSnr: (snr: number) => {
      set((state) => {
        state.minSnrDb = snr;
      });
    },
    
    // Set frequency range filter
    setFrequencyRange: (minHz: number | null, maxHz: number | null) => {
      set((state) => {
        state.minFreqHz = minHz;
        state.maxFreqHz = maxHz;
      });
    },
    
    // Toggle modulation filter
    toggleModulationFilter: (modulation: string) => {
      set((state) => {
        if (state.modulationFilter.has(modulation)) {
          state.modulationFilter.delete(modulation);
        } else {
          state.modulationFilter.add(modulation);
        }
      });
    },
    
    // Clear all filters
    clearFilters: () => {
      set((state) => {
        state.minSnrDb = -Infinity;
        state.minFreqHz = null;
        state.maxFreqHz = null;
        state.modulationFilter.clear();
      });
    },
    
    // Get detection by ID
    getDetection: (id: string) => {
      return get().detections.get(id);
    },
    
    // Get sorted detections
    getSortedDetections: (sortBy: 'frequency' | 'power' | 'snr' | 'time') => {
      const detections = Array.from(get().detections.values());
      
      switch (sortBy) {
        case 'frequency':
          return detections.sort((a, b) => a.centerFreqHz - b.centerFreqHz);
        case 'power':
          return detections.sort((a, b) => b.peakPowerDb - a.peakPowerDb);
        case 'snr':
          return detections.sort((a, b) => b.snrDb - a.snrDb);
        case 'time':
          return detections.sort((a, b) => b.timestamp - a.timestamp);
        default:
          return detections;
      }
    },
    
    // Get active detections (within last 5 seconds)
    getActiveDetections: () => {
      const now = Date.now();
      const threshold = 5000; // 5 seconds
      
      return Array.from(get().detections.values()).filter(
        (det) => now - det.timestamp < threshold
      );
    },
    
    // Get filtered detections
    getFilteredDetections: () => {
      const state = get();
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
          (det) => det.modulationType != null && state.modulationFilter.has(det.modulationType)
        );
      }
      
      // Filter inactive if needed
      if (!state.showInactive) {
        const now = Date.now();
        const threshold = 5000;
        detections = detections.filter((det) => now - det.timestamp < threshold);
      }
      
      return detections;
    },
  }))
);

// Selectors
export const selectDetections = (state: DetectionState) => state.detections;
export const selectSelectedDetection = (state: DetectionState) =>
  state.selectedDetectionId
    ? state.detections.get(state.selectedDetectionId)
    : null;
export const selectDetectionCount = (state: DetectionState) => state.detections.size;
