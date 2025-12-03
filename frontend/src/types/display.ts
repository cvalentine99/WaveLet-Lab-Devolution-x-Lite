/**
 * Display Settings Types - Spectrum visualization configuration
 */

import type { ColorMapType } from './visualization';

/**
 * Display settings for spectrum visualization
 */
export interface DisplaySettings {
  /** Selected colormap */
  colorMap: ColorMapType;

  /** Dynamic range minimum (dBm) */
  dynamicRangeMin: number;

  /** Dynamic range maximum (dBm) */
  dynamicRangeMax: number;

  /** Show grid lines */
  showGrid: boolean;

  /** Show frequency axis */
  showFrequencyAxis: boolean;

  /** Show power axis */
  showPowerAxis: boolean;

  /** Show time axis */
  showTimeAxis: boolean;

  /** Show detection overlays */
  showDetections: boolean;

  /** Show band markers */
  showBandMarkers: boolean;

  /** Show CFAR threshold */
  showCfarThreshold: boolean;

  /** Waterfall history depth (lines) */
  waterfallHistoryDepth: number;

  /** Persistence mode */
  persistenceMode: 'none' | 'max-hold' | 'average';

  /** FFT size */
  fftSize: number;

  /** Window function */
  windowFunction: 'hann' | 'kaiser' | 'blackman-harris' | 'hamming' | 'flat-top';
}
