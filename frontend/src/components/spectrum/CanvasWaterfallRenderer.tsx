import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import type { ColorMapType } from '@/utils/colormaps';
import { COLORMAPS } from '@/utils/colormaps';

export interface CanvasWaterfallRendererProps {
  width: number;
  height: number;
  historyDepth?: number;
  colormap?: ColorMapType;
  dynamicRangeMin?: number;
  dynamicRangeMax?: number;
  onError?: (error: Error) => void;
}

export interface CanvasWaterfallRendererRef {
  addLine: (psd: Float32Array) => void;
  setColormap: (colormap: ColorMapType) => void;
  setDynamicRange: (min: number, max: number) => void;
  clear: () => void;
}

/**
 * Canvas 2D Waterfall Renderer (Fallback)
 * CPU-based scrolling spectrogram when WebGPU is not available
 */
export const CanvasWaterfallRenderer = forwardRef<CanvasWaterfallRendererRef, CanvasWaterfallRendererProps>(
  function CanvasWaterfallRenderer(
    {
      width,
      height,
      historyDepth = 512,
      colormap = 'viridis',
      dynamicRangeMin = -120,
      dynamicRangeMax = -20,
      onError,
    },
    ref
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [fps, setFps] = useState(0);
    
    const waterfallDataRef = useRef<Float32Array[]>([]);
    const colormapRef = useRef<ColorMapType>(colormap);
    const dynamicRangeRef = useRef({ min: dynamicRangeMin, max: dynamicRangeMax });
    
    const frameCountRef = useRef(0);
    const lastFpsUpdateRef = useRef(Date.now());

    /**
     * Add new PSD line to waterfall
     */
    const addLine = (psd: Float32Array) => {
      // Add to history
      waterfallDataRef.current.push(new Float32Array(psd));
      
      // Keep only historyDepth lines
      if (waterfallDataRef.current.length > historyDepth) {
        waterfallDataRef.current.shift();
      }

      render();
    };

    /**
     * Set colormap
     */
    const setColormap = (newColormap: ColorMapType) => {
      colormapRef.current = newColormap;
      render();
    };

    /**
     * Set dynamic range
     */
    const setDynamicRange = (min: number, max: number) => {
      dynamicRangeRef.current = { min, max };
      render();
    };

    /**
     * Clear waterfall
     */
    const clear = () => {
      waterfallDataRef.current = [];
      render();
    };

    /**
     * Get color from colormap
     */
    const getColor = (value: number): string => {
      const colormapData = COLORMAPS[colormapRef.current].data;
      const index = Math.floor(value * 255);
      const offset = index * 3;
      const r = colormapData[offset];
      const g = colormapData[offset + 1];
      const b = colormapData[offset + 2];
      return `rgb(${r}, ${g}, ${b})`;
    };

    /**
     * Render waterfall
     */
    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const data = waterfallDataRef.current;
      if (data.length === 0) {
        // Clear canvas
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, width, height);
        return;
      }

      // Clear canvas
      ctx.fillStyle = '#000000';
      ctx.fillRect(0, 0, width, height);

      const { min, max } = dynamicRangeRef.current;
      const range = max - min || 1;  // Avoid division by zero

      const lineHeight = height / (data.length || 1);  // Avoid division by zero

      // Draw each line
      for (let y = 0; y < data.length; y++) {
        const line = data[y];
        if (!line || line.length === 0) continue;
        const pixelWidth = width / line.length;

        for (let x = 0; x < line.length; x++) {
          const normalized = Math.max(0, Math.min(1, (line[x] - min) / range));
          ctx.fillStyle = getColor(normalized);
          ctx.fillRect(x * pixelWidth, y * lineHeight, pixelWidth + 1, lineHeight + 1);
        }
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

    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      addLine,
      setColormap,
      setDynamicRange,
      clear,
    }));

    return (
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="border border-border rounded-lg bg-background"
        />
        {fps > 0 && (
          <div className="absolute top-2 right-2 bg-background/80 px-2 py-1 rounded text-xs font-mono text-muted-foreground">
            {fps} FPS (Canvas 2D)
          </div>
        )}
      </div>
    );
  }
);
