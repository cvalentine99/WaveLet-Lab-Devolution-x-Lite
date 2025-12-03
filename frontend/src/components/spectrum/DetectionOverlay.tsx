import { useMemo, useCallback } from 'react';
import { useDetectionStore } from '@/stores/detectionStore';
import { useClusterStore } from '@/stores/clusterStore';
import type { Detection } from '@/types';

export interface DetectionOverlayProps {
  width: number;
  height: number;
  centerFreqHz: number;
  spanHz: number;
  minPowerDbm: number;
  maxPowerDbm: number;
}

/**
 * Detection Overlay Component
 * Draws translucent bounding boxes over detected signals on spectrum display
 */
export function DetectionOverlay({
  width,
  height,
  centerFreqHz,
  spanHz,
  minPowerDbm,
  maxPowerDbm,
}: DetectionOverlayProps) {
  // Subscribe to size for re-render triggers
  const detectionCount = useDetectionStore((state) => state.detections.size);
  const selectedDetectionId = useDetectionStore((state) => state.selectedDetectionId);
  const clusterColorCount = useClusterStore((state) => state.clusterColors.size);

  // Get detections via getState() - avoids unstable selector snapshot
  const detections = useMemo(() => {
    return useDetectionStore.getState().getActiveDetections();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectionCount]);

  // Create a stable color getter
  const getClusterColor = useCallback(
    (id: number): string => {
      const CLUSTER_COLORS = [
        '#10b981', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6',
        '#ec4899', '#06b6d4', '#f97316', '#84cc16', '#6366f1',
      ];
      const colors = useClusterStore.getState().clusterColors;
      return colors.get(id) || CLUSTER_COLORS[id % CLUSTER_COLORS.length];
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [clusterColorCount]
  );

  // Convert frequency to X coordinate
  const freqToX = useCallback(
    (freqHz: number) => {
      const startFreq = centerFreqHz - spanHz / 2;
      const normalized = (freqHz - startFreq) / spanHz;
      return normalized * width;
    },
    [centerFreqHz, spanHz, width]
  );

  // Convert power to Y coordinate (inverted - higher power = lower Y)
  const powerToY = useCallback(
    (powerDbm: number) => {
      const normalized = (powerDbm - minPowerDbm) / (maxPowerDbm - minPowerDbm);
      return height - normalized * height;
    },
    [minPowerDbm, maxPowerDbm, height]
  );

  // Filter detections that are within the current frequency span
  const visibleDetections = useMemo(() => {
    const startFreq = centerFreqHz - spanHz / 2;
    const endFreq = centerFreqHz + spanHz / 2;
    
    return detections.filter((det) => {
      const bandwidth = det.bandwidth3dbHz || det.bandwidthHz;
      const detStartFreq = det.centerFreqHz - bandwidth / 2;
      const detEndFreq = det.centerFreqHz + bandwidth / 2;
      
      // Check if detection overlaps with visible frequency range
      return detEndFreq >= startFreq && detStartFreq <= endFreq;
    });
  }, [detections, centerFreqHz, spanHz]);

  return (
    <svg
      className="absolute inset-0 pointer-events-none"
      width={width}
      height={height}
      style={{ zIndex: 10 }}
    >
      {visibleDetections.map((detection) => {
        const isSelected = detection.id === selectedDetectionId;
        const clusterColor = detection.clusterId !== undefined && detection.clusterId !== null
          ? getClusterColor(detection.clusterId)
          : '#888888';

        // Calculate bounding box coordinates
        const bandwidth = detection.bandwidth3dbHz || detection.bandwidthHz;
        const startFreq = detection.centerFreqHz - bandwidth / 2;
        const endFreq = detection.centerFreqHz + bandwidth / 2;
        
        const x1 = freqToX(startFreq);
        const x2 = freqToX(endFreq);
        const boxWidth = Math.max(x2 - x1, 2); // Minimum 2px width
        
        // Use peak power for top, and estimate bottom based on SNR
        const topY = powerToY(detection.peakPowerDb);
        const bottomY = powerToY(detection.peakPowerDb - detection.snrDb);
        const boxHeight = Math.max(bottomY - topY, 4); // Minimum 4px height

        return (
          <g key={detection.id}>
            {/* Bounding box */}
            <rect
              x={x1}
              y={topY}
              width={boxWidth}
              height={boxHeight}
              fill={clusterColor}
              fillOpacity={isSelected ? 0.3 : 0.15}
              stroke={clusterColor}
              strokeWidth={isSelected ? 2 : 1}
              strokeOpacity={isSelected ? 0.9 : 0.6}
              rx={2}
            />
            
            {/* Center frequency marker */}
            <line
              x1={freqToX(detection.centerFreqHz)}
              y1={topY}
              x2={freqToX(detection.centerFreqHz)}
              y2={topY + boxHeight}
              stroke={clusterColor}
              strokeWidth={isSelected ? 2 : 1}
              strokeOpacity={0.8}
              strokeDasharray={isSelected ? "none" : "2,2"}
            />

            {/* Frequency label (only for selected or prominent detections) */}
            {(isSelected || detection.snrDb > 20) && (
              <g>
                {/* Label background */}
                <rect
                  x={x1}
                  y={topY - 20}
                  width={Math.max(boxWidth, 80)}
                  height={18}
                  fill="rgba(0, 0, 0, 0.8)"
                  rx={2}
                />
                {/* Frequency text */}
                <text
                  x={x1 + 4}
                  y={topY - 7}
                  fill={clusterColor}
                  fontSize="11"
                  fontFamily="monospace"
                  fontWeight={isSelected ? "bold" : "normal"}
                >
                  {(detection.centerFreqHz / 1e6).toFixed(3)} MHz
                </text>
              </g>
            )}

            {/* Selection indicator */}
            {isSelected && (
              <circle
                cx={freqToX(detection.centerFreqHz)}
                cy={topY}
                r={4}
                fill={clusterColor}
                stroke="white"
                strokeWidth={2}
              />
            )}
          </g>
        );
      })}
    </svg>
  );
}
