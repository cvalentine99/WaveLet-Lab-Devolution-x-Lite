import { useRef, useEffect, useCallback } from 'react';
import { useSpectrumStore } from '@/stores/spectrumStore';

export interface SpectrumOverlayProps {
  width: number;
  height: number;
  centerFreqHz: number;
  spanHz: number;
  minPowerDb: number;
  maxPowerDb: number;
}

/**
 * Spectrum Overlay Component
 * Renders cursor crosshair, peak hold trace, average trace, and markers
 */
export function SpectrumOverlay({
  width,
  height,
  centerFreqHz,
  spanHz,
  minPowerDb,
  maxPowerDb,
}: SpectrumOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const currentPsd = useSpectrumStore((state) => state.currentPsd);
  const peakHoldBuffer = useSpectrumStore((state) => state.peakHoldBuffer);
  const peakHoldEnabled = useSpectrumStore((state) => state.peakHoldEnabled);
  const avgBuffer = useSpectrumStore((state) => state.avgBuffer);
  const avgEnabled = useSpectrumStore((state) => state.avgEnabled);
  const cursorFreqHz = useSpectrumStore((state) => state.cursorFreqHz);
  const cursorPowerDb = useSpectrumStore((state) => state.cursorPowerDb);
  const markers = useSpectrumStore((state) => state.markers);
  const setCursor = useSpectrumStore((state) => state.setCursor);
  const addMarker = useSpectrumStore((state) => state.addMarker);

  const minFreqHz = centerFreqHz - spanHz / 2;
  const maxFreqHz = centerFreqHz + spanHz / 2;
  const powerRange = maxPowerDb - minPowerDb;

  // Convert frequency to X position
  const freqToX = useCallback((freqHz: number) => {
    return ((freqHz - minFreqHz) / spanHz) * width;
  }, [minFreqHz, spanHz, width]);

  // Convert X position to frequency
  const xToFreq = useCallback((x: number) => {
    return minFreqHz + (x / width) * spanHz;
  }, [minFreqHz, spanHz, width]);

  // Convert power to Y position
  const powerToY = useCallback((powerDb: number) => {
    const normalized = Math.max(0, Math.min(1, (powerDb - minPowerDb) / powerRange));
    return height - normalized * height;
  }, [minPowerDb, powerRange, height]);

  // Get power at frequency from PSD
  const getPowerAtFreq = useCallback((freqHz: number, psd: Float32Array | null) => {
    if (!psd || psd.length === 0) return null;
    const binIndex = Math.floor(((freqHz - minFreqHz) / spanHz) * psd.length);
    if (binIndex < 0 || binIndex >= psd.length) return null;
    return psd[binIndex];
  }, [minFreqHz, spanHz]);

  // Handle mouse move
  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;

    const freqHz = xToFreq(x);
    const powerDb = getPowerAtFreq(freqHz, currentPsd);

    setCursor(freqHz, powerDb);
  }, [xToFreq, getPowerAtFreq, currentPsd, setCursor]);

  // Handle mouse leave
  const handleMouseLeave = useCallback(() => {
    setCursor(null, null);
  }, [setCursor]);

  // Handle click to add marker
  const handleClick = useCallback((e: React.MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const freqHz = xToFreq(x);
    addMarker(freqHz);
  }, [xToFreq, addMarker]);

  // Render overlay
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    // Clear canvas
    ctx.clearRect(0, 0, width, height);

    // Draw peak hold trace (orange)
    if (peakHoldEnabled && peakHoldBuffer && peakHoldBuffer.length > 0) {
      ctx.strokeStyle = '#f97316';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 2]);
      ctx.beginPath();

      for (let i = 0; i < peakHoldBuffer.length; i++) {
        const x = (i / peakHoldBuffer.length) * width;
        const y = powerToY(peakHoldBuffer[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw average trace (purple)
    if (avgEnabled && avgBuffer && avgBuffer.length > 0) {
      ctx.strokeStyle = '#a855f7';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();

      for (let i = 0; i < avgBuffer.length; i++) {
        const x = (i / avgBuffer.length) * width;
        const y = powerToY(avgBuffer[i]);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Draw markers (cyan vertical lines)
    markers.forEach((marker) => {
      const x = freqToX(marker.freqHz);
      if (x >= 0 && x <= width) {
        ctx.strokeStyle = '#22d3ee';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
        ctx.setLineDash([]);

        // Draw marker label
        ctx.fillStyle = '#22d3ee';
        ctx.font = '10px monospace';
        ctx.textAlign = 'center';
        ctx.fillText(marker.label, x, 12);
        ctx.fillText(`${(marker.freqHz / 1e6).toFixed(3)}`, x, 24);
      }
    });

    // Draw cursor crosshair
    if (cursorFreqHz !== null) {
      const x = freqToX(cursorFreqHz);

      // Vertical line
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)';
      ctx.lineWidth = 1;
      ctx.setLineDash([2, 2]);
      ctx.beginPath();
      ctx.moveTo(x, 0);
      ctx.lineTo(x, height);
      ctx.stroke();
      ctx.setLineDash([]);

      // Horizontal line at power level
      if (cursorPowerDb !== null) {
        const y = powerToY(cursorPowerDb);
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();

        // Draw tooltip
        const tooltipText = `${(cursorFreqHz / 1e6).toFixed(3)} MHz | ${cursorPowerDb.toFixed(1)} dBm`;
        ctx.font = '11px monospace';
        const textWidth = ctx.measureText(tooltipText).width;

        // Position tooltip
        let tooltipX = x + 10;
        let tooltipY = y - 10;
        if (tooltipX + textWidth + 10 > width) {
          tooltipX = x - textWidth - 20;
        }
        if (tooltipY < 20) {
          tooltipY = y + 25;
        }

        // Draw tooltip background
        ctx.fillStyle = 'rgba(10, 15, 20, 0.9)';
        ctx.strokeStyle = 'rgba(34, 211, 238, 0.5)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.roundRect(tooltipX, tooltipY - 14, textWidth + 12, 20, 4);
        ctx.fill();
        ctx.stroke();

        // Draw tooltip text
        ctx.fillStyle = '#f1f5f9';
        ctx.fillText(tooltipText, tooltipX + 6, tooltipY);
      }
    }
  }, [
    width, height, currentPsd,
    peakHoldBuffer, peakHoldEnabled,
    avgBuffer, avgEnabled,
    cursorFreqHz, cursorPowerDb,
    markers, freqToX, powerToY
  ]);

  return (
    <canvas
      ref={canvasRef}
      width={width}
      height={height}
      className="absolute inset-0 pointer-events-auto"
      style={{ cursor: 'crosshair' }}
      onMouseMove={handleMouseMove}
      onMouseLeave={handleMouseLeave}
      onClick={handleClick}
    />
  );
}
