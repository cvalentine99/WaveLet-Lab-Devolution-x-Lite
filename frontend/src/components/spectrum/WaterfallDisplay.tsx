import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import { COLORMAPS, type SpectrumFrame } from '@/lib/spectrumParser';

export interface WaterfallDisplayProps {
  width?: number;
  height?: number;
  colormap?: keyof typeof COLORMAPS;
  dynamicRangeMin?: number;
  dynamicRangeMax?: number;
  showGrid?: boolean;
  showFrequencyAxis?: boolean;
  showTimeAxis?: boolean;
}

/**
 * High-performance waterfall display using Canvas 2D
 * Renders scrolling spectrogram from binary spectrum data
 */
export const WaterfallDisplay = forwardRef<WaterfallDisplayRef, WaterfallDisplayProps>(function WaterfallDisplay({
  width = 1024,
  height = 512,
  colormap = 'viridis',
  dynamicRangeMin = -120,
  dynamicRangeMax = -20,
  showGrid = true,
  showFrequencyAxis = true,
  showTimeAxis = true,
}: WaterfallDisplayProps, ref) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const [fps, setFps] = useState(0);
  const frameCountRef = useRef(0);
  const lastFpsUpdateRef = useRef(Date.now());

  // Initialize offscreen canvas for ring buffer
  useEffect(() => {
    if (!offscreenCanvasRef.current) {
      offscreenCanvasRef.current = document.createElement('canvas');
      offscreenCanvasRef.current.width = width;
      offscreenCanvasRef.current.height = height * 2; // Double height for scrolling
    }
  }, [width, height]);

  /**
   * Render a new spectrum line to the waterfall
   */
  const renderSpectrumLine = (frame: SpectrumFrame) => {
    const canvas = canvasRef.current;
    const offscreenCanvas = offscreenCanvasRef.current;
    if (!canvas || !offscreenCanvas) return;

    const ctx = canvas.getContext('2d');
    const offscreenCtx = offscreenCanvas.getContext('2d');
    if (!ctx || !offscreenCtx) return;

    // Get colormap
    const colormapData = COLORMAPS[colormap];

    // Create ImageData for the new line
    const lineData = new ImageData(frame.width, 1);
    const pixels = lineData.data;

    // Map spectrum values to colors
    for (let i = 0; i < frame.width; i++) {
      const value = frame.data[i]; // 0-255
      const colorIndex = value * 3;
      pixels[i * 4 + 0] = colormapData[colorIndex + 0]; // R
      pixels[i * 4 + 1] = colormapData[colorIndex + 1]; // G
      pixels[i * 4 + 2] = colormapData[colorIndex + 2]; // B
      pixels[i * 4 + 3] = 255; // A
    }

    // Scroll existing content down by 1 pixel
    const existingData = offscreenCtx.getImageData(0, 0, width, height);
    offscreenCtx.putImageData(existingData, 0, 1);

    // Draw new line at top
    offscreenCtx.putImageData(lineData, 0, 0);

    // Copy to main canvas
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(offscreenCanvas, 0, 0, width, height, 0, 0, width, height);

    // Draw overlays
    if (showGrid) {
      drawGrid(ctx, width, height);
    }
    if (showFrequencyAxis && frame.frequencyStart && frame.frequencyEnd) {
      drawFrequencyAxis(ctx, width, height, frame.frequencyStart, frame.frequencyEnd);
    }
    if (showTimeAxis) {
      drawTimeAxis(ctx, width, height);
    }

    // Update FPS
    frameCountRef.current++;
    const now = Date.now();
    if (now - lastFpsUpdateRef.current >= 1000) {
      setFps(frameCountRef.current);
      frameCountRef.current = 0;
      lastFpsUpdateRef.current = now;
    }
  };

  // Expose render method via imperative handle
  useImperativeHandle(ref, () => ({
    renderSpectrumLine,
  }));

  return (
    <div className="relative">
      <canvas
        ref={canvasRef}
        width={width}
        height={height}
        className="border border-border rounded-lg bg-background"
        style={{ imageRendering: 'pixelated' }}
      />
      {fps > 0 && (
        <div className="absolute top-2 right-2 bg-background/80 px-2 py-1 rounded text-xs font-mono text-muted-foreground">
          {fps} FPS
        </div>
      )}
    </div>
  );
});

function drawGrid(ctx: CanvasRenderingContext2D, width: number, height: number) {
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
  ctx.lineWidth = 1;

  // Vertical lines (frequency)
  const vSpacing = width / 10;
  for (let i = 0; i <= 10; i++) {
    const x = i * vSpacing;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, height);
    ctx.stroke();
  }

  // Horizontal lines (time)
  const hSpacing = height / 10;
  for (let i = 0; i <= 10; i++) {
    const y = i * hSpacing;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }
}

function drawFrequencyAxis(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  freqStart: number,
  freqEnd: number
) {
  ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
  ctx.font = '10px monospace';
  ctx.textAlign = 'center';

  const numLabels = 5;
  for (let i = 0; i <= numLabels; i++) {
    const x = (i / numLabels) * width;
    const freq = freqStart + (i / numLabels) * (freqEnd - freqStart);
    const label = formatFrequency(freq);
    ctx.fillText(label, x, height - 5);
  }
}

function drawTimeAxis(ctx: CanvasRenderingContext2D, width: number, height: number) {
  ctx.fillStyle = 'rgba(255, 255, 255, 0.9)';
  ctx.font = '10px monospace';
  ctx.textAlign = 'left';

  // Show time labels on left side
  const numLabels = 5;
  for (let i = 0; i <= numLabels; i++) {
    const y = (i / numLabels) * height;
    const timeAgo = i * 2; // seconds ago
    ctx.fillText(`-${timeAgo}s`, 5, y + 10);
  }
}

function formatFrequency(freq: number): string {
  if (freq >= 1e9) {
    return `${(freq / 1e9).toFixed(2)} GHz`;
  } else if (freq >= 1e6) {
    return `${(freq / 1e6).toFixed(2)} MHz`;
  } else if (freq >= 1e3) {
    return `${(freq / 1e3).toFixed(2)} kHz`;
  } else {
    return `${freq.toFixed(0)} Hz`;
  }
}

// Export ref type for parent components
export interface WaterfallDisplayRef {
  renderSpectrumLine: (frame: SpectrumFrame) => void;
}
