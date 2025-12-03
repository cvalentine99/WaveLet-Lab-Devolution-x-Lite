import { useEffect, useRef, useState, forwardRef, useImperativeHandle, useCallback } from 'react';
import type { ColorMapType } from '@/utils/colormaps';
import { COLORMAPS } from '@/utils/colormaps';

export interface CanvasSpectrumRendererProps {
  width: number;
  height: number;
  colormap?: ColorMapType;
  dynamicRangeMin?: number;
  dynamicRangeMax?: number;
  onError?: (error: Error) => void;
}

export interface CanvasSpectrumRendererRef {
  updatePSD: (psd: Float32Array) => void;
  setColormap: (colormap: ColorMapType) => void;
  setDynamicRange: (min: number, max: number) => void;
}

/**
 * Canvas 2D Spectrum Renderer (Optimized Fallback)
 * High-performance CPU-based rendering when WebGPU is not available
 */
export const CanvasSpectrumRenderer = forwardRef<CanvasSpectrumRendererRef, CanvasSpectrumRendererProps>(
  function CanvasSpectrumRenderer(
    {
      width,
      height,
      colormap = 'viridis',
      dynamicRangeMin = -120,
      dynamicRangeMax = -20,
      onError,
    },
    ref
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const ctxRef = useRef<CanvasRenderingContext2D | null>(null);
    const [fps, setFps] = useState(0);

    const currentPsdRef = useRef<Float32Array | null>(null);
    const colormapRef = useRef<ColorMapType>(colormap);
    const dynamicRangeRef = useRef({ min: dynamicRangeMin, max: dynamicRangeMax });

    const frameCountRef = useRef(0);
    const lastFpsUpdateRef = useRef(Date.now());
    const rafIdRef = useRef<number | null>(null);
    const needsRenderRef = useRef(false);

    // Initialize canvas context once
    useEffect(() => {
      const canvas = canvasRef.current;
      if (canvas) {
        ctxRef.current = canvas.getContext('2d', {
          alpha: false, // Optimization: no transparency needed
          desynchronized: true // Optimization: allow async rendering
        });
      }
      return () => {
        if (rafIdRef.current) {
          cancelAnimationFrame(rafIdRef.current);
        }
      };
    }, []);

    /**
     * Render loop using requestAnimationFrame for smooth performance
     */
    const renderLoop = useCallback(() => {
      if (!needsRenderRef.current) {
        rafIdRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      const ctx = ctxRef.current;
      const psd = currentPsdRef.current;

      if (!ctx || !psd || psd.length === 0) {
        rafIdRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      needsRenderRef.current = false;

      // Clear canvas
      ctx.fillStyle = '#0a0a0a';
      ctx.fillRect(0, 0, width, height);

      // Draw spectrum line with optimized path batching
      const { min, max } = dynamicRangeRef.current;
      const range = max - min;
      const psdLen = psd.length;

      // Decimation for large datasets (skip points when width < psdLen)
      const step = Math.max(1, Math.floor(psdLen / width));

      ctx.strokeStyle = '#22c55e'; // green-500
      ctx.lineWidth = 1.5;
      ctx.lineJoin = 'round';
      ctx.beginPath();

      let firstPoint = true;
      for (let i = 0; i < psdLen; i += step) {
        const x = (i / psdLen) * width;
        const normalized = Math.max(0, Math.min(1, (psd[i] - min) / range));
        const y = height - normalized * height;

        if (firstPoint) {
          ctx.moveTo(x, y);
          firstPoint = false;
        } else {
          ctx.lineTo(x, y);
        }
      }

      ctx.stroke();

      // Draw filled gradient below the line for better visibility
      ctx.globalAlpha = 0.15;
      ctx.fillStyle = '#22c55e';
      ctx.lineTo(width, height);
      ctx.lineTo(0, height);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;

      // Update FPS counter
      frameCountRef.current++;
      const now = Date.now();
      if (now - lastFpsUpdateRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsUpdateRef.current = now;
      }

      rafIdRef.current = requestAnimationFrame(renderLoop);
    }, [width, height]);

    // Start render loop
    useEffect(() => {
      rafIdRef.current = requestAnimationFrame(renderLoop);
      return () => {
        if (rafIdRef.current) {
          cancelAnimationFrame(rafIdRef.current);
        }
      };
    }, [renderLoop]);

    /**
     * Update PSD data (mark for rendering)
     */
    const updatePSD = useCallback((psd: Float32Array) => {
      currentPsdRef.current = psd;
      needsRenderRef.current = true;
    }, []);

    /**
     * Set colormap
     */
    const setColormap = useCallback((newColormap: ColorMapType) => {
      colormapRef.current = newColormap;
      needsRenderRef.current = true;
    }, []);

    /**
     * Set dynamic range
     */
    const setDynamicRange = useCallback((min: number, max: number) => {
      dynamicRangeRef.current = { min, max };
      needsRenderRef.current = true;
    }, []);

    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      updatePSD,
      setColormap,
      setDynamicRange,
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
            {fps} FPS
          </div>
        )}
      </div>
    );
  }
);
