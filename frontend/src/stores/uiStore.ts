import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { persist } from 'zustand/middleware';
import type { ColorMapType } from '@/types';

/**
 * UI Store
 * Manages UI state (sidebar, panels, modals, etc.)
 */

interface UIState {
  // Layout
  sidebarOpen: boolean;
  rightPanelTab: 'detections' | 'clusters' | 'details';
  bottomDrawerOpen: boolean;
  bottomDrawerTab: 'demod' | 'logs' | 'export';
  
  // Visualization
  colorMap: ColorMapType;
  showCursor: boolean;
  showPeakMarker: boolean;
  showDeltaMarker: boolean;
  
  // Modals
  settingsModalOpen: boolean;
  presetModalOpen: boolean;
  exportModalOpen: boolean;
  
  // Actions
  toggleSidebar: () => void;
  setRightPanelTab: (tab: 'detections' | 'clusters' | 'details') => void;
  toggleBottomDrawer: () => void;
  setBottomDrawerTab: (tab: 'demod' | 'logs' | 'export') => void;
  
  setColorMap: (colorMap: ColorMapType) => void;
  toggleCursor: () => void;
  togglePeakMarker: () => void;
  toggleDeltaMarker: () => void;
  
  openSettingsModal: () => void;
  closeSettingsModal: () => void;
  openPresetModal: () => void;
  closePresetModal: () => void;
  openExportModal: () => void;
  closeExportModal: () => void;
}

export const useUIStore = create<UIState>()(
  persist(
    immer((set) => ({
      // Initial state
      sidebarOpen: true,
      rightPanelTab: 'detections',
      bottomDrawerOpen: false,
      bottomDrawerTab: 'demod',
      
      colorMap: 'viridis',
      showCursor: true,
      showPeakMarker: true,
      showDeltaMarker: false,
      
      settingsModalOpen: false,
      presetModalOpen: false,
      exportModalOpen: false,
      
      // Toggle sidebar
      toggleSidebar: () => {
        set((state) => {
          state.sidebarOpen = !state.sidebarOpen;
        });
      },
      
      // Set right panel tab
      setRightPanelTab: (tab: 'detections' | 'clusters' | 'details') => {
        set((state) => {
          state.rightPanelTab = tab;
        });
      },
      
      // Toggle bottom drawer
      toggleBottomDrawer: () => {
        set((state) => {
          state.bottomDrawerOpen = !state.bottomDrawerOpen;
        });
      },
      
      // Set bottom drawer tab
      setBottomDrawerTab: (tab: 'demod' | 'logs' | 'export') => {
        set((state) => {
          state.bottomDrawerTab = tab;
        });
      },
      
      // Set colormap
      setColorMap: (colorMap: ColorMapType) => {
        set((state) => {
          state.colorMap = colorMap;
        });
      },
      
      // Toggle cursor
      toggleCursor: () => {
        set((state) => {
          state.showCursor = !state.showCursor;
        });
      },
      
      // Toggle peak marker
      togglePeakMarker: () => {
        set((state) => {
          state.showPeakMarker = !state.showPeakMarker;
        });
      },
      
      // Toggle delta marker
      toggleDeltaMarker: () => {
        set((state) => {
          state.showDeltaMarker = !state.showDeltaMarker;
        });
      },
      
      // Modal actions
      openSettingsModal: () => {
        set((state) => {
          state.settingsModalOpen = true;
        });
      },
      
      closeSettingsModal: () => {
        set((state) => {
          state.settingsModalOpen = false;
        });
      },
      
      openPresetModal: () => {
        set((state) => {
          state.presetModalOpen = true;
        });
      },
      
      closePresetModal: () => {
        set((state) => {
          state.presetModalOpen = false;
        });
      },
      
      openExportModal: () => {
        set((state) => {
          state.exportModalOpen = true;
        });
      },
      
      closeExportModal: () => {
        set((state) => {
          state.exportModalOpen = false;
        });
      },
    })),
    {
      name: 'rf-forensics-ui',
      partialize: (state) => ({
        sidebarOpen: state.sidebarOpen,
        colorMap: state.colorMap,
        showCursor: state.showCursor,
        showPeakMarker: state.showPeakMarker,
      }),
    }
  )
);

// Selectors
export const selectSidebarOpen = (state: UIState) => state.sidebarOpen;
export const selectRightPanelTab = (state: UIState) => state.rightPanelTab;
export const selectBottomDrawerOpen = (state: UIState) => state.bottomDrawerOpen;
export const selectColorMap = (state: UIState) => state.colorMap;
