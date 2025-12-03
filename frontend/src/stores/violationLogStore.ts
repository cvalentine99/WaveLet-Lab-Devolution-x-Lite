import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export interface ViolationLog {
  id: string;
  thresholdId: string;
  thresholdName: string;
  timestamp: Date;
  centerFreqHz: number;
  powerDbm: number;
  thresholdDbm: number;
  exceedanceDbm: number; // How much it exceeded the threshold
  detectionId?: string;
}

interface ViolationLogState {
  violations: Map<string, ViolationLog>;
  addViolation: (violation: Omit<ViolationLog, 'id' | 'timestamp'>) => void;
  getViolationsByThreshold: (thresholdId: string) => ViolationLog[];
  getRecentViolations: (limit?: number) => ViolationLog[];
  clearViolations: () => void;
  deleteViolation: (id: string) => void;
  searchViolations: (query: string) => ViolationLog[];
}

export const useViolationLogStore = create<ViolationLogState>()(
  persist(
    (set, get) => ({
      violations: new Map(),

      addViolation: (violation) => {
        const id = `viol_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        const newViolation: ViolationLog = {
          ...violation,
          id,
          timestamp: new Date(),
        };

        set((state) => {
          const newViolations = new Map(state.violations);
          newViolations.set(id, newViolation);
          return { violations: newViolations };
        });
      },

      getViolationsByThreshold: (thresholdId) => {
        const violations = Array.from(get().violations.values());
        return violations
          .filter((v) => v.thresholdId === thresholdId)
          .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
      },

      getRecentViolations: (limit = 100) => {
        const violations = Array.from(get().violations.values());
        return violations
          .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
          .slice(0, limit);
      },

      clearViolations: () => {
        set({ violations: new Map() });
      },

      deleteViolation: (id) => {
        set((state) => {
          const newViolations = new Map(state.violations);
          newViolations.delete(id);
          return { violations: newViolations };
        });
      },

      searchViolations: (query) => {
        const violations = Array.from(get().violations.values());
        const lowerQuery = query.toLowerCase();
        return violations
          .filter(
            (v) =>
              v.thresholdName.toLowerCase().includes(lowerQuery) ||
              (v.centerFreqHz / 1e6).toFixed(2).includes(lowerQuery)
          )
          .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime());
      },
    }),
    {
      name: 'violation-log-storage',
      storage: {
        getItem: (name) => {
          const str = localStorage.getItem(name);
          if (!str) return null;
          const { state } = JSON.parse(str);
          return {
            state: {
              ...state,
              violations: new Map(
                Object.entries(state.violations || {}).map(([k, v]: [string, any]) => [
                  k,
                  { ...v, timestamp: new Date(v.timestamp) },
                ])
              ),
            },
          };
        },
        setItem: (name, value) => {
          const str = JSON.stringify({
            state: {
              ...value.state,
              violations: Object.fromEntries(value.state.violations),
            },
          });
          localStorage.setItem(name, str);
        },
        removeItem: (name) => localStorage.removeItem(name),
      },
    }
  )
);
