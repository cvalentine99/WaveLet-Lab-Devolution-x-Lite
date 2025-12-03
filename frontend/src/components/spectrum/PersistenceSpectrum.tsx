import { useRef, useEffect, forwardRef, useImperativeHandle, useState } from 'react';
import type { ColorMapType } from '@/utils/colormaps';
import { COLORMAPS } from '@/utils/colormaps';

export interface PersistenceSpectrumProps {
  width: number;
  height: number;
  decayRate?: number; // 0-1, how fast old data fades (0.02 = slow, 0.1 = fast)
  colormap?: ColorMapType;
  dynamicRangeMin?: number;
  dynamicRangeMax?: number;
}

export interface PersistenceSpectrumRef {
  addPSD: (psd: Float32Array) => void;
  setColormap: (colormap: ColorMapType) => void;
  setDynamicRange: (min: number, max: number) => void;
  setDecayRate: (rate: number) => void;
  clear: () => void;
}

/**
 * Persistence Spectrum Display
 *
 * Shows signal density/occupancy over time by accumulating and decaying
 * power measurements. Brighter areas = more frequent signals.
 * Similar to "Max Hold" but with decay for temporal analysis.
 */
export const PersistenceSpectrum = forwardRef<PersistenceSpectrumRef, PersistenceSpectrumProps>(
  function PersistenceSpectrum(
    {
      width,
      height,
      decayRate = 0.02,
      colormap = 'viridis',
      dynamicRangeMin = -120,
      dynamicRangeMax = -20,
    },
    ref
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [fps, setFps] = useState(0);

    // Persistence accumulator - 2D array [frequency_bin][power_level]
    // Each cell counts how often that power level was seen at that frequency
    const persistenceRef = useRef<Float32Array | null>(null);
    const numBinsRef = useRef<number>(0);
    const numPowerLevels = 256; // Quantize power to 256 levels

    const colormapRef = useRef<ColorMapType>(colormap);
    const dynamicRangeRef = useRef({ min: dynamicRangeMin, max: dynamicRangeMax });
    const decayRateRef = useRef(decayRate);

    const frameCountRef = useRef(0);
    const lastFpsUpdateRef = useRef(Date.now());
    const animationFrameRef = useRef<number>(0);

    /**
     * Initialize persistence buffer for given number of bins
     */
    const initPersistence = (numBins: number) => {
      if (numBinsRef.current !== numBins) {
        numBinsRef.current = numBins;
        // Create 2D persistence buffer: [numBins * numPowerLevels]
        persistenceRef.current = new Float32Array(numBins * numPowerLevels);
      }
    };

    /**
     * Add new PSD data and update persistence
     */
    const addPSD = (psd: Float32Array) => {
      if (psd.length === 0) return;

      initPersistence(psd.length);
      const persistence = persistenceRef.current;
      if (!persistence) return;

      const { min, max } = dynamicRangeRef.current;
      const range = max - min || 1;
      const decay = decayRateRef.current;

      // Apply decay to all cells
      for (let i = 0; i < persistence.length; i++) {
        persistence[i] = Math.max(0, persistence[i] - decay);
      }

      // Add new PSD data
      for (let bin = 0; bin < psd.length; bin++) {
        // Quantize power value to 0-255
        const normalized = Math.max(0, Math.min(1, (psd[bin] - min) / range));
        const powerLevel = Math.floor(normalized * (numPowerLevels - 1));

        // Increment persistence at this frequency/power
        const idx = bin * numPowerLevels + powerLevel;
        persistence[idx] = Math.min(1, persistence[idx] + 0.3);
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
    };

    /**
     * Set decay rate
     */
    const setDecayRate = (rate: number) => {
      decayRateRef.current = Math.max(0.001, Math.min(0.5, rate));
    };

    /**
     * Clear persistence buffer
     */
    const clear = () => {
      if (persistenceRef.current) {
        persistenceRef.current.fill(0);
      }
      render();
    };

    /**
     * Get color from colormap
     */
    const getColor = (value: number): [number, number, number] => {
      const colormapData = COLORMAPS[colormapRef.current].data;
      const index = Math.floor(Math.max(0, Math.min(1, value)) * 255);
      const offset = index * 3;
      return [colormapData[offset], colormapData[offset + 1], colormapData[offset + 2]];
    };

    /**
     * Render persistence display
     */
    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const persistence = persistenceRef.current;
      const numBins = numBinsRef.current;

      if (!persistence || numBins === 0) {
        ctx.fillStyle = '#0a0e14';
        ctx.fillRect(0, 0, width, height);

        // Draw "waiting for data" message
        ctx.fillStyle = '#666';
        ctx.font = '14px "Space Grotesk", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for spectrum data...', width / 2, height / 2);
        return;
      }

      // Create ImageData for efficient pixel manipulation
      const imageData = ctx.createImageData(width, height);
      const pixels = imageData.data;

      const binWidth = width / numBins;
      const powerHeight = height / numPowerLevels;

      // Fill with background
      for (let i = 0; i < pixels.length; i += 4) {
        pixels[i] = 10;     // R
        pixels[i + 1] = 14; // G
        pixels[i + 2] = 20; // B
        pixels[i + 3] = 255; // A
      }

      // Draw persistence data
      for (let bin = 0; bin < numBins; bin++) {
        const x1 = Math.floor(bin * binWidth);
        const x2 = Math.floor((bin + 1) * binWidth);

        for (let power = 0; power < numPowerLevels; power++) {
          const idx = bin * numPowerLevels + power;
          const intensity = persistence[idx];

          if (intensity > 0.01) {
            // Y is inverted (0 power at bottom, high power at top)
            const y1 = Math.floor((numPowerLevels - power - 1) * powerHeight);
            const y2 = Math.floor((numPowerLevels - power) * powerHeight);

            const [r, g, b] = getColor(intensity);

            // Fill pixels
            for (let y = y1; y < y2 && y < height; y++) {
              for (let x = x1; x < x2 && x < width; x++) {
                const pixelIdx = (y * width + x) * 4;
                // Blend with existing (additive)
                pixels[pixelIdx] = Math.min(255, pixels[pixelIdx] + r * intensity);
                pixels[pixelIdx + 1] = Math.min(255, pixels[pixelIdx + 1] + g * intensity);
                pixels[pixelIdx + 2] = Math.min(255, pixels[pixelIdx + 2] + b * intensity);
              }
            }
          }
        }
      }

      ctx.putImageData(imageData, 0, 0);

      // Draw power scale on right side
      const scaleWidth = 3;
      const { min, max } = dynamicRangeRef.current;
      for (let y = 0; y < height; y++) {
        const normalized = 1 - y / height;
        const [r, g, b] = getColor(normalized);
        ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
        ctx.fillRect(width - scaleWidth - 2, y, scaleWidth, 1);
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
      addPSD,
      setColormap,
      setDynamicRange,
      setDecayRate,
      clear,
    }));

    // Initial render
    useEffect(() => {
      render();
    }, [width, height]);

    return (
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="border border-border rounded-lg"
        />
        <div className="absolute top-2 left-2 bg-background/80 px-2 py-1 rounded text-xs font-mono text-muted-foreground">
          Persistence
        </div>
        {fps > 0 && (
          <div className="absolute top-2 right-2 bg-background/80 px-2 py-1 rounded text-xs font-mono text-muted-foreground">
            {fps} FPS
          </div>
        )}
      </div>
    );
  }
);
