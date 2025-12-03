import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface ThresholdAlert {
  id: string;
  name: string;
  enabled: boolean;
  freqStartHz: number;
  freqEndHz: number;
  thresholdDbm: number;
  alertType: 'toast' | 'audio' | 'both';
  audioFile?: string;
  lastTriggeredAt?: Date;
  triggerCount: number;
  createdAt: Date;
}

interface ThresholdState {
  thresholds: Map<string, ThresholdAlert>;
  addThreshold: (threshold: Omit<ThresholdAlert, 'id' | 'createdAt' | 'triggerCount' | 'lastTriggeredAt'>) => void;
  updateThreshold: (id: string, updates: Partial<ThresholdAlert>) => void;
  deleteThreshold: (id: string) => void;
  incrementTrigger: (id: string) => void;
  getEnabledThresholds: () => ThresholdAlert[];
}

export const useThresholdStore = create<ThresholdState>()(
  persist(
    (set, get) => ({
      thresholds: new Map(),

      addThreshold: (threshold) => {
        const id = `threshold_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const newThreshold: ThresholdAlert = {
          ...threshold,
          id,
          triggerCount: 0,
          createdAt: new Date(),
        };

        set((state) => {
          const newThresholds = new Map(state.thresholds);
          newThresholds.set(id, newThreshold);
          return { thresholds: newThresholds };
        });
      },

      updateThreshold: (id, updates) => {
        set((state) => {
          const threshold = state.thresholds.get(id);
          if (!threshold) return state;

          const newThresholds = new Map(state.thresholds);
          newThresholds.set(id, { ...threshold, ...updates });
          return { thresholds: newThresholds };
        });
      },

      deleteThreshold: (id) => {
        set((state) => {
          const newThresholds = new Map(state.thresholds);
          newThresholds.delete(id);
          return { thresholds: newThresholds };
        });
      },

      incrementTrigger: (id) => {
        set((state) => {
          const threshold = state.thresholds.get(id);
          if (!threshold) return state;

          const newThresholds = new Map(state.thresholds);
          newThresholds.set(id, {
            ...threshold,
            triggerCount: threshold.triggerCount + 1,
            lastTriggeredAt: new Date(),
          });
          return { thresholds: newThresholds };
        });
      },

      getEnabledThresholds: () => {
        return Array.from(get().thresholds.values()).filter((t) => t.enabled);
      },
    }),
    {
      name: 'rf-forensics-thresholds',
      storage: {
        getItem: (name) => {
          const str = localStorage.getItem(name);
          if (!str) return null;
          const parsed = JSON.parse(str);
          
          // Reconstruct Map with proper Date objects
          const thresholdEntries = (parsed.state.thresholds || []).map(
            ([key, value]: [string, any]) => {
              return [key, {
                ...value,
                // Restore Date objects from ISO strings
                createdAt: value.createdAt ? new Date(value.createdAt) : new Date(),
                lastTriggeredAt: value.lastTriggeredAt ? new Date(value.lastTriggeredAt) : undefined,
              }];
            }
          );
          
          return {
            state: {
              ...parsed.state,
              thresholds: new Map(thresholdEntries),
            },
          };
        },
        setItem: (name, value) => {
          const serialized = {
            state: {
              ...value.state,
              thresholds: Array.from((value.state.thresholds as Map<string, ThresholdAlert>).entries()),
            },
          };
          localStorage.setItem(name, JSON.stringify(serialized));
        },
        removeItem: (name) => localStorage.removeItem(name),
      },
    }
  )
);
