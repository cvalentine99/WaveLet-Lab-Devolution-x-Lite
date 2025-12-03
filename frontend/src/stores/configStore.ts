import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { persist } from 'zustand/middleware';
import type { SDRConfig, PipelineStatus, DisplaySettings, Preset } from '@/types';

/**
 * Config Store
 * Manages SDR configuration, pipeline status, and display settings
 */

interface ConfigState {
  // SDR configuration
  sdrConfig: SDRConfig;
  
  // Pipeline status
  pipelineStatus: PipelineStatus;
  
  // Display settings
  displaySettings: DisplaySettings;
  
  // Presets
  presets: Map<string, Preset>;
  currentPresetName: string | null;
  
  // Actions
  updateSdrConfig: (config: Partial<SDRConfig>) => void;
  updatePipelineStatus: (status: Partial<PipelineStatus>) => void;
  updateDisplaySettings: (settings: Partial<DisplaySettings>) => void;
  
  // Presets
  savePreset: (name: string, description?: string) => void;
  loadPreset: (name: string) => void;
  deletePreset: (name: string) => void;
  applyPreset: (preset: Preset) => void;
}

const DEFAULT_SDR_CONFIG: SDRConfig = {
  centerFreqHz: 915e6, // 915 MHz ISM band
  sampleRateHz: 10e6, // 10 Msps
  bandwidthHz: 10e6,
  gain: 40,
  deviceType: 'Simulated',
};

const DEFAULT_PIPELINE_STATUS: PipelineStatus = {
  state: 'IDLE',
  uptimeSeconds: 0,
  samplesProcessed: 0,
  detectionsCount: 0,
  currentThroughputMsps: 0,
  gpuMemoryUsedGb: 0,
  gpuUtilizationPercent: 0,
};

const DEFAULT_DISPLAY_SETTINGS: DisplaySettings = {
  colorMap: 'viridis',
  dynamicRangeMin: -120,
  dynamicRangeMax: -20,
  showGrid: true,
  showFrequencyAxis: true,
  showPowerAxis: true,
  showTimeAxis: true,
  showDetections: true,
  showBandMarkers: false,
  showCfarThreshold: true,
  waterfallHistoryDepth: 512,
  persistenceMode: 'none',
  fftSize: 1024,
  windowFunction: 'hann',
};

export const useConfigStore = create<ConfigState>()(
  persist(
    immer((set, get) => ({
      // Initial state
      sdrConfig: DEFAULT_SDR_CONFIG,
      pipelineStatus: DEFAULT_PIPELINE_STATUS,
      displaySettings: DEFAULT_DISPLAY_SETTINGS,
      presets: new Map(),
      currentPresetName: null,
      
      // Update SDR config
      updateSdrConfig: (config: Partial<SDRConfig>) => {
        set((state) => {
          Object.assign(state.sdrConfig, config);
          state.currentPresetName = null; // Clear preset when manually changed
        });
      },
      
      // Update pipeline status
      updatePipelineStatus: (status: Partial<PipelineStatus>) => {
        set((state) => {
          Object.assign(state.pipelineStatus, status);
        });
      },
      
      // Update display settings
      updateDisplaySettings: (settings: Partial<DisplaySettings>) => {
        set((state) => {
          Object.assign(state.displaySettings, settings);
        });
      },
      
      // Save current config as preset
      savePreset: (name: string, description?: string) => {
        set((state) => {
          const preset: Preset = {
            name,
            sdrConfig: { ...state.sdrConfig },
            displaySettings: { ...state.displaySettings },
            description,
            createdAt: Date.now(),
          };
          
          state.presets.set(name, preset);
          state.currentPresetName = name;
        });
      },
      
      // Load preset by name
      loadPreset: (name: string) => {
        const preset = get().presets.get(name);
        if (preset) {
          get().applyPreset(preset);
        }
      },
      
      // Delete preset
      deletePreset: (name: string) => {
        set((state) => {
          state.presets.delete(name);
          if (state.currentPresetName === name) {
            state.currentPresetName = null;
          }
        });
      },
      
      // Apply preset
      applyPreset: (preset: Preset) => {
        set((state) => {
          state.sdrConfig = { ...preset.sdrConfig };
          state.displaySettings = { ...preset.displaySettings };
          state.currentPresetName = preset.name;
        });
      },
    })),
    {
      name: 'rf-forensics-config',
      partialize: (state) => ({
        sdrConfig: state.sdrConfig, // Fixed: now persists SDR settings
        displaySettings: state.displaySettings,
        presets: Array.from(state.presets.entries()),
      }),
      onRehydrateStorage: () => (state) => {
        if (state && Array.isArray(state.presets)) {
          state.presets = new Map(state.presets as any);
        }
      },
    }
  )
);

// Selectors
export const selectSdrConfig = (state: ConfigState) => state.sdrConfig;
export const selectPipelineStatus = (state: ConfigState) => state.pipelineStatus;
export const selectDisplaySettings = (state: ConfigState) => state.displaySettings;
export const selectPresets = (state: ConfigState) => Array.from(state.presets.values());
