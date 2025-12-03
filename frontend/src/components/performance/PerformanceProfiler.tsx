import { Profiler, ProfilerOnRenderCallback, ReactNode } from 'react';

/**
 * Performance Profiler Wrapper
 * Tracks component render times and logs slow renders in development
 */

export interface PerformanceProfilerProps {
  id: string;
  children: ReactNode;
  logSlowRenders?: boolean;
  slowThresholdMs?: number;
}

// Store performance metrics
const performanceMetrics = new Map<string, {
  renderCount: number;
  totalDuration: number;
  maxDuration: number;
  minDuration: number;
  lastRender: number;
}>();

export function PerformanceProfiler({
  id,
  children,
  logSlowRenders = true,
  slowThresholdMs = 16, // 60fps = 16.67ms per frame
}: PerformanceProfilerProps) {
  const onRender: ProfilerOnRenderCallback = (
    id,
    phase,
    actualDuration,
    baseDuration
  ) => {
    // Get or create metrics for this component
    const metrics = performanceMetrics.get(id) || {
      renderCount: 0,
      totalDuration: 0,
      maxDuration: 0,
      minDuration: Infinity,
      lastRender: 0,
    };

    // Update metrics
    metrics.renderCount++;
    metrics.totalDuration += actualDuration;
    metrics.maxDuration = Math.max(metrics.maxDuration, actualDuration);
    metrics.minDuration = Math.min(metrics.minDuration, actualDuration);
    metrics.lastRender = actualDuration;

    performanceMetrics.set(id, metrics);

    // Log slow renders in development
    if (logSlowRenders && actualDuration > slowThresholdMs && import.meta.env.DEV) {
      console.warn(
        `[Performance] Slow render detected in "${id}"`,
        `\n  Phase: ${phase}`,
        `\n  Actual duration: ${actualDuration.toFixed(2)}ms`,
        `\n  Base duration: ${baseDuration.toFixed(2)}ms`,
        `\n  Threshold: ${slowThresholdMs}ms`,
        `\n  Render count: ${metrics.renderCount}`,
        `\n  Average: ${(metrics.totalDuration / metrics.renderCount).toFixed(2)}ms`,

      );
    }
  };

  return (
    <Profiler id={id} onRender={onRender}>
      {children}
    </Profiler>
  );
}

/**
 * Get performance metrics for a component
 */
export function getPerformanceMetrics(id: string) {
  return performanceMetrics.get(id);
}

/**
 * Get all performance metrics
 */
export function getAllPerformanceMetrics() {
  return Array.from(performanceMetrics.entries()).map(([id, metrics]) => ({
    id,
    ...metrics,
    avgDuration: metrics.totalDuration / metrics.renderCount,
  }));
}

/**
 * Clear performance metrics
 */
export function clearPerformanceMetrics() {
  performanceMetrics.clear();
}

/**
 * Export performance report
 */
export function exportPerformanceReport() {
  const metrics = getAllPerformanceMetrics();
  const report = {
    timestamp: new Date().toISOString(),
    metrics: metrics.map(m => ({
      component: m.id,
      renders: m.renderCount,
      avgMs: parseFloat(m.avgDuration.toFixed(2)),
      maxMs: parseFloat(m.maxDuration.toFixed(2)),
      minMs: parseFloat(m.minDuration.toFixed(2)),
      totalMs: parseFloat(m.totalDuration.toFixed(2)),
    })),
  };

  // Download as JSON
  const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `performance-report-${Date.now()}.json`;
  a.click();
  URL.revokeObjectURL(url);
}
