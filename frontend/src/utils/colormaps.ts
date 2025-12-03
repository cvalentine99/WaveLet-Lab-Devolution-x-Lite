/**
 * Colormap Utilities
 * Predefined color palettes for spectrum visualization
 */

export type ColorMapType = 'viridis' | 'plasma' | 'turbo' | 'jet' | 'grayscale' | 'custom';

/**
 * Colormap data structure (RGB values 0-255)
 */
export interface ColorMap {
  name: string;
  data: Uint8Array; // RGB triplets (length = 256 * 3)
}

/**
 * Viridis colormap (perceptually uniform)
 * Good for general-purpose spectrum visualization
 */
const VIRIDIS_DATA = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  VIRIDIS_DATA[i * 3 + 0] = Math.floor(255 * (0.267 + 0.005 * t + 0.323 * t * t - 0.094 * t * t * t));
  VIRIDIS_DATA[i * 3 + 1] = Math.floor(255 * (0.005 + 0.503 * t + 0.489 * t * t - 0.257 * t * t * t));
  VIRIDIS_DATA[i * 3 + 2] = Math.floor(255 * (0.330 + 1.108 * t - 1.315 * t * t + 0.377 * t * t * t));
}

/**
 * Plasma colormap (high contrast)
 */
const PLASMA_DATA = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  PLASMA_DATA[i * 3 + 0] = Math.floor(255 * (0.050 + 1.065 * t - 0.334 * t * t + 0.220 * t * t * t));
  PLASMA_DATA[i * 3 + 1] = Math.floor(255 * (0.030 + 0.719 * t + 1.055 * t * t - 1.803 * t * t * t));
  PLASMA_DATA[i * 3 + 2] = Math.floor(255 * (0.528 + 1.388 * t - 2.534 * t * t + 1.618 * t * t * t));
}

/**
 * Turbo colormap (rainbow, wide dynamic range)
 */
const TURBO_DATA = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  const r = Math.max(0, Math.min(1, 0.13 + 1.48 * t - 2.24 * t * t + 1.63 * t * t * t));
  const g = Math.max(0, Math.min(1, 0.09 + 2.00 * t - 2.00 * t * t + 0.91 * t * t * t));
  const b = Math.max(0, Math.min(1, 0.40 + 1.60 * t - 2.00 * t * t + 1.00 * t * t * t));
  TURBO_DATA[i * 3 + 0] = Math.floor(255 * r);
  TURBO_DATA[i * 3 + 1] = Math.floor(255 * g);
  TURBO_DATA[i * 3 + 2] = Math.floor(255 * b);
}

/**
 * Jet colormap (classic rainbow)
 */
const JET_DATA = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  const t = i / 255;
  const r = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 3)));
  const g = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 2)));
  const b = Math.max(0, Math.min(1, 1.5 - Math.abs(4 * t - 1)));
  JET_DATA[i * 3 + 0] = Math.floor(255 * r);
  JET_DATA[i * 3 + 1] = Math.floor(255 * g);
  JET_DATA[i * 3 + 2] = Math.floor(255 * b);
}

/**
 * Grayscale colormap
 */
const GRAYSCALE_DATA = new Uint8Array(256 * 3);
for (let i = 0; i < 256; i++) {
  GRAYSCALE_DATA[i * 3 + 0] = i;
  GRAYSCALE_DATA[i * 3 + 1] = i;
  GRAYSCALE_DATA[i * 3 + 2] = i;
}

/**
 * Colormap registry
 */
export const COLORMAPS: Record<ColorMapType, ColorMap> = {
  viridis: { name: 'Viridis', data: VIRIDIS_DATA },
  plasma: { name: 'Plasma', data: PLASMA_DATA },
  turbo: { name: 'Turbo', data: TURBO_DATA },
  jet: { name: 'Jet', data: JET_DATA },
  grayscale: { name: 'Grayscale', data: GRAYSCALE_DATA },
  custom: { name: 'Custom', data: VIRIDIS_DATA }, // Default to viridis for custom
};

/**
 * Get colormap by name
 */
export function getColorMap(name: ColorMapType): ColorMap {
  return COLORMAPS[name];
}

/**
 * Create RGBA texture data from colormap (for WebGPU)
 */
export function createColormapTexture(name: ColorMapType): Uint8Array {
  const colormap = getColorMap(name);
  const rgba = new Uint8Array(256 * 4);
  
  for (let i = 0; i < 256; i++) {
    rgba[i * 4 + 0] = colormap.data[i * 3 + 0]; // R
    rgba[i * 4 + 1] = colormap.data[i * 3 + 1]; // G
    rgba[i * 4 + 2] = colormap.data[i * 3 + 2]; // B
    rgba[i * 4 + 3] = 255; // A
  }
  
  return rgba;
}

/**
 * Get colormap CSS gradient for legend
 */
export function getColormapGradient(name: ColorMapType): string {
  const colormap = getColorMap(name);
  const stops: string[] = [];
  
  // Sample 10 points for gradient
  for (let i = 0; i <= 10; i++) {
    const idx = Math.floor((i / 10) * 255) * 3;
    const r = colormap.data[idx + 0];
    const g = colormap.data[idx + 1];
    const b = colormap.data[idx + 2];
    stops.push(`rgb(${r},${g},${b}) ${i * 10}%`);
  }
  
  return `linear-gradient(to right, ${stops.join(', ')})`;
}
