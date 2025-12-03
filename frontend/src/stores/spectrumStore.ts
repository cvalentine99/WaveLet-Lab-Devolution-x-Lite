import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';

/**
 * Spectrum Store
 * Manages PSD data, waterfall buffer, and frequency range
 */

interface SpectrumState {
  // Current PSD data
  currentPsd: Float32Array | null;

  // Waterfall circular buffer
  waterfallBuffer: Float32Array[];
  waterfallWriteIndex: number;
  waterfallHistoryDepth: number;

  // Frequency configuration
  centerFreqHz: number;
  spanHz: number;
  sampleRateHz: number;
  numBins: number;

  // Computed values
  frequencyAxis: Float32Array | null;
  binWidthHz: number;

  // IQ data for constellation display
  currentIQ: { i: Float32Array; q: Float32Array } | null;
  iqTimestamp: number;

  // Peak hold trace (shows max values with decay)
  peakHoldBuffer: Float32Array | null;
  peakHoldEnabled: boolean;
  peakHoldDecayDbPerFrame: number;

  // Average trace (exponential moving average)
  avgBuffer: Float32Array | null;
  avgEnabled: boolean;
  avgAlpha: number; // EMA smoothing factor (0.0-1.0)

  // Cursor state for crosshair
  cursorFreqHz: number | null;
  cursorPowerDb: number | null;

  // Markers
  markers: Array<{ freqHz: number; label: string; color?: string }>;

  // Actions
  updatePsd: (psd: Float32Array, timestamp: number) => void;
  updateIQ: (i: Float32Array, q: Float32Array, timestamp: number) => void;
  clearWaterfall: () => void;
  setFrequencyRange: (centerHz: number, spanHz: number, sampleRateHz: number) => void;
  setWaterfallDepth: (depth: number) => void;

  // Peak/Avg controls
  togglePeakHold: () => void;
  toggleAverage: () => void;
  setPeakHoldDecay: (dbPerFrame: number) => void;
  setAvgAlpha: (alpha: number) => void;
  clearPeakHold: () => void;

  // Cursor controls
  setCursor: (freqHz: number | null, powerDb: number | null) => void;

  // Marker controls
  addMarker: (freqHz: number, label?: string) => void;
  removeMarker: (freqHz: number) => void;
  clearMarkers: () => void;

  // Getters
  getFrequencyAxis: () => Float32Array;
  getWaterfallLine: (index: number) => Float32Array | null;
}

export const useSpectrumStore = create<SpectrumState>()(
  immer((set, get) => ({
    // Initial state
    currentPsd: null,
    waterfallBuffer: [],
    waterfallWriteIndex: 0,
    waterfallHistoryDepth: 512,

    centerFreqHz: 915e6, // Default 915 MHz
    spanHz: 10e6, // Default 10 MHz span
    sampleRateHz: 10e6,
    numBins: 1024,

    frequencyAxis: null,
    binWidthHz: 0,

    // IQ data
    currentIQ: null,
    iqTimestamp: 0,

    // Peak hold (orange trace)
    peakHoldBuffer: null,
    peakHoldEnabled: false,
    peakHoldDecayDbPerFrame: 0.1,

    // Average (purple trace)
    avgBuffer: null,
    avgEnabled: false,
    avgAlpha: 0.1,

    // Cursor
    cursorFreqHz: null,
    cursorPowerDb: null,

    // Markers
    markers: [],
    
    // Update PSD and add to waterfall
    updatePsd: (psd: Float32Array, timestamp: number) => {
      set((state) => {
        state.currentPsd = psd;
        state.numBins = psd.length;

        // Add to waterfall buffer (circular)
        if (state.waterfallBuffer.length < state.waterfallHistoryDepth) {
          state.waterfallBuffer.push(new Float32Array(psd));
        } else {
          state.waterfallBuffer[state.waterfallWriteIndex] = new Float32Array(psd);
        }

        state.waterfallWriteIndex =
          (state.waterfallWriteIndex + 1) % state.waterfallHistoryDepth;

        // Update computed values
        state.binWidthHz = state.spanHz / psd.length;

        // Update peak hold buffer (with decay)
        if (state.peakHoldEnabled) {
          if (!state.peakHoldBuffer || state.peakHoldBuffer.length !== psd.length) {
            state.peakHoldBuffer = new Float32Array(psd);
          } else {
            const decay = state.peakHoldDecayDbPerFrame;
            for (let i = 0; i < psd.length; i++) {
              // Decay existing peak, then take max with new value
              state.peakHoldBuffer[i] = Math.max(
                state.peakHoldBuffer[i] - decay,
                psd[i]
              );
            }
          }
        }

        // Update average buffer (exponential moving average)
        if (state.avgEnabled) {
          if (!state.avgBuffer || state.avgBuffer.length !== psd.length) {
            state.avgBuffer = new Float32Array(psd);
          } else {
            const alpha = state.avgAlpha;
            const oneMinusAlpha = 1 - alpha;
            for (let i = 0; i < psd.length; i++) {
              state.avgBuffer[i] = alpha * psd[i] + oneMinusAlpha * state.avgBuffer[i];
            }
          }
        }
      });
    },

    // Update IQ samples for constellation display
    updateIQ: (i: Float32Array, q: Float32Array, timestamp: number) => {
      set((state) => {
        state.currentIQ = { i, q };
        state.iqTimestamp = timestamp;
      });
    },

    // Clear waterfall history
    clearWaterfall: () => {
      set((state) => {
        state.waterfallBuffer = [];
        state.waterfallWriteIndex = 0;
      });
    },
    
    // Set frequency range
    setFrequencyRange: (centerHz: number, spanHz: number, sampleRateHz: number) => {
      set((state) => {
        state.centerFreqHz = centerHz;
        state.spanHz = spanHz;
        state.sampleRateHz = sampleRateHz;
        state.binWidthHz = spanHz / state.numBins;
        
        // Recompute frequency axis
        const startFreq = centerHz - spanHz / 2;
        const axis = new Float32Array(state.numBins);
        for (let i = 0; i < state.numBins; i++) {
          axis[i] = startFreq + i * state.binWidthHz;
        }
        state.frequencyAxis = axis;
      });
    },
    
    // Set waterfall history depth
    setWaterfallDepth: (depth: number) => {
      set((state) => {
        const oldDepth = state.waterfallHistoryDepth;
        state.waterfallHistoryDepth = depth;
        
        // Trim buffer if needed - maintain proper circular order
        if (state.waterfallBuffer.length > depth) {
          // Reorder buffer so oldest data is at index 0, then trim
          const writeIdx = state.waterfallWriteIndex;
          const oldBuffer = state.waterfallBuffer;
          const reordered: Float32Array[] = [];
          
          // Start from oldest (writeIndex) and wrap around
          for (let i = 0; i < oldBuffer.length; i++) {
            const idx = (writeIdx + i) % oldBuffer.length;
            reordered.push(oldBuffer[idx]);
          }
          
          // Keep only the most recent 'depth' entries (end of reordered array)
          state.waterfallBuffer = reordered.slice(-depth);
          state.waterfallWriteIndex = 0; // Now correct - buffer is full and ordered
        } else if (depth > oldDepth) {
          // Growing buffer - writeIndex stays valid, just allow more entries
          state.waterfallWriteIndex = state.waterfallWriteIndex % Math.max(state.waterfallBuffer.length, 1);
        }
      });
    },
    
    // Get frequency axis (computed)
    getFrequencyAxis: () => {
      const state = get();
      if (state.frequencyAxis) {
        return state.frequencyAxis;
      }
      
      const startFreq = state.centerFreqHz - state.spanHz / 2;
      const axis = new Float32Array(state.numBins);
      const binWidth = state.spanHz / state.numBins;
      
      for (let i = 0; i < state.numBins; i++) {
        axis[i] = startFreq + i * binWidth;
      }
      
      return axis;
    },
    
    // Get waterfall line by index
    getWaterfallLine: (index: number) => {
      const state = get();
      if (index < 0 || index >= state.waterfallBuffer.length) {
        return null;
      }
      return state.waterfallBuffer[index];
    },

    // Peak hold controls
    togglePeakHold: () => {
      set((state) => {
        state.peakHoldEnabled = !state.peakHoldEnabled;
        if (!state.peakHoldEnabled) {
          state.peakHoldBuffer = null;
        }
      });
    },

    clearPeakHold: () => {
      set((state) => {
        state.peakHoldBuffer = null;
      });
    },

    setPeakHoldDecay: (dbPerFrame: number) => {
      set((state) => {
        state.peakHoldDecayDbPerFrame = dbPerFrame;
      });
    },

    // Average controls
    toggleAverage: () => {
      set((state) => {
        state.avgEnabled = !state.avgEnabled;
        if (!state.avgEnabled) {
          state.avgBuffer = null;
        }
      });
    },

    setAvgAlpha: (alpha: number) => {
      set((state) => {
        state.avgAlpha = Math.max(0.01, Math.min(1.0, alpha));
      });
    },

    // Cursor controls
    setCursor: (freqHz: number | null, powerDb: number | null) => {
      set((state) => {
        state.cursorFreqHz = freqHz;
        state.cursorPowerDb = powerDb;
      });
    },

    // Marker controls
    addMarker: (freqHz: number, label?: string) => {
      set((state) => {
        const newLabel = label || `M${state.markers.length + 1}`;
        state.markers.push({ freqHz, label: newLabel });
      });
    },

    removeMarker: (freqHz: number) => {
      set((state) => {
        state.markers = state.markers.filter(m => Math.abs(m.freqHz - freqHz) > 1000);
      });
    },

    clearMarkers: () => {
      set((state) => {
        state.markers = [];
      });
    },
  }))
);

// Selectors for optimized re-renders
export const selectCurrentPsd = (state: SpectrumState) => state.currentPsd;
export const selectFrequencyRange = (state: SpectrumState) => ({
  centerHz: state.centerFreqHz,
  spanHz: state.spanHz,
  sampleRateHz: state.sampleRateHz,
});
export const selectWaterfallBuffer = (state: SpectrumState) => state.waterfallBuffer;
export const selectNumBins = (state: SpectrumState) => state.numBins;
export const selectCurrentIQ = (state: SpectrumState) => state.currentIQ;
export const selectPeakHold = (state: SpectrumState) => ({
  buffer: state.peakHoldBuffer,
  enabled: state.peakHoldEnabled,
});
export const selectAverage = (state: SpectrumState) => ({
  buffer: state.avgBuffer,
  enabled: state.avgEnabled,
});
export const selectCursor = (state: SpectrumState) => ({
  freqHz: state.cursorFreqHz,
  powerDb: state.cursorPowerDb,
});
export const selectMarkers = (state: SpectrumState) => state.markers;
