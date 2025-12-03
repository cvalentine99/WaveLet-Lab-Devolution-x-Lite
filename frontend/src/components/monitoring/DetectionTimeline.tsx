import { useEffect, useRef, useState, useMemo, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { useDetectionStore } from '@/stores/detectionStore';
import { useClusterStore } from '@/stores/clusterStore';
import { Clock } from 'lucide-react';
import { useShallow } from 'zustand/react/shallow';
import type { Detection } from '@/types';

interface TimelineEvent {
  timestamp: number;
  detectionId: string;
  clusterId: number | undefined;
  centerFreqHz: number;
}

export interface DetectionTimelineProps {
  className?: string;
  timeWindowSeconds?: number; // How many seconds of history to show
}

// Stable selector for detections array
const selectDetections = (state: { detections: Map<string, Detection> }): Detection[] =>
  Array.from(state.detections.values());

/**
 * Detection History Timeline
 * Shows detection events over time with color-coded cluster markers
 */
export function DetectionTimeline({ 
  className = '', 
  timeWindowSeconds = 60 
}: DetectionTimelineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [events, setEvents] = useState<TimelineEvent[]>([]);
  
  // Use useShallow for array results to prevent infinite loops
  const detections = useDetectionStore(useShallow(selectDetections));
  
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

  // Track new detections
  useEffect(() => {
    const now = Date.now();
    const newEvents = detections.map(d => ({
      timestamp: now,
      detectionId: d.id,
      clusterId: d.clusterId,
      centerFreqHz: d.centerFreqHz
    }));

    setEvents(prev => {
      // Merge new events, remove duplicates, and filter old events
      const cutoff = now - (timeWindowSeconds * 1000);
      const merged = [...prev, ...newEvents];
      const unique = merged.filter((event, index, self) =>
        index === self.findIndex(e => e.detectionId === event.detectionId) &&
        event.timestamp >= cutoff
      );
      return unique.sort((a, b) => a.timestamp - b.timestamp);
    });
  }, [detections, timeWindowSeconds]);

  // Draw timeline
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);

    const width = rect.width;
    const height = rect.height;

    // Clear canvas
    ctx.fillStyle = '#09090b'; // bg-background
    ctx.fillRect(0, 0, width, height);

    // Draw grid lines
    ctx.strokeStyle = '#27272a'; // border color
    ctx.lineWidth = 1;
    
    // Vertical time grid (every 10 seconds)
    const gridInterval = 10; // seconds
    for (let i = 0; i <= timeWindowSeconds; i += gridInterval) {
      const x = (i / timeWindowSeconds) * width;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
    }

    // Horizontal center line
    ctx.beginPath();
    ctx.moveTo(0, height / 2);
    ctx.lineTo(width, height / 2);
    ctx.stroke();

    // Draw events
    const now = Date.now();
    const timeWindow = timeWindowSeconds * 1000;

    events.forEach(event => {
      const age = now - event.timestamp;
      if (age > timeWindow) return;

      const x = ((timeWindow - age) / timeWindow) * width;
      const color = event.clusterId !== undefined ? getClusterColor(event.clusterId) : '#71717a';

      // Draw event marker
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, height / 2, 3, 0, Math.PI * 2);
      ctx.fill();

      // Draw vertical line
      ctx.strokeStyle = color;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.globalAlpha = 1.0;
    });

    // Draw time labels
    ctx.fillStyle = '#71717a'; // text-muted-foreground
    ctx.font = '10px monospace';
    ctx.textAlign = 'center';
    
    for (let i = 0; i <= timeWindowSeconds; i += gridInterval) {
      const x = (i / timeWindowSeconds) * width;
      const label = `-${timeWindowSeconds - i}s`;
      ctx.fillText(label, x, height - 4);
    }

  }, [events, timeWindowSeconds, getClusterColor]);

  return (
    <Card className={`p-3 ${className}`}>
      <div className="space-y-2">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Clock className="w-3 h-3 text-primary" />
            <span className="text-xs font-semibold">Detection History</span>
          </div>
          <span className="text-xs text-muted-foreground">
            Last {timeWindowSeconds}s ({events.length} events)
          </span>
        </div>

        {/* Timeline Canvas */}
        <canvas
          ref={canvasRef}
          className="w-full h-16 rounded border border-border"
          style={{ imageRendering: 'pixelated' }}
        />

        {/* Legend */}
        <div className="flex items-center gap-3 text-[10px] text-muted-foreground">
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-primary" />
            <span>Clustered</span>
          </div>
          <div className="flex items-center gap-1">
            <div className="w-2 h-2 rounded-full bg-muted" />
            <span>Unclustered</span>
          </div>
        </div>
      </div>
    </Card>
  );
}
