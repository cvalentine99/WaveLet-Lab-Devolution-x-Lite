/**
 * Visualization Types - Spectrum display configuration
 */

/**
 * Colormap type for spectrum visualization
 */
export type ColorMapType = 'viridis' | 'plasma' | 'turbo' | 'jet' | 'grayscale' | 'custom';

/**
 * Frequency scale configuration
 */
export interface FrequencyScale {
  /** Minimum frequency in Hz */
  minHz: number;

  /** Maximum frequency in Hz */
  maxHz: number;

  /** Frequency resolution (Hz per bin) */
  resolutionHz: number;
}

/**
 * Power scale configuration
 */
export interface PowerScale {
  /** Minimum power in dBm */
  minDbm: number;

  /** Maximum power in dBm */
  maxDbm: number;

  /** Reference level in dBm */
  referenceLevelDbm: number;
}

/**
 * Time scale configuration
 */
export interface TimeScale {
  /** Start time (Unix milliseconds) */
  startMs: number;

  /** End time (Unix milliseconds) */
  endMs: number;

  /** Time resolution in milliseconds */
  resolutionMs: number;
}

/**
 * Viewport bounds for zooming/panning
 */
export interface ViewportBounds {
  /** Frequency range */
  frequency: FrequencyScale;

  /** Time range */
  time: TimeScale;

  /** Power range */
  power: PowerScale;
}

/**
 * Marker type for annotations
 */
export type MarkerType = 'cursor' | 'peak' | 'delta' | 'annotation' | 'band';

/**
 * Spectrum marker/annotation
 */
export interface Marker {
  /** Marker ID */
  id: string;

  /** Marker type */
  type: MarkerType;

  /** Frequency in Hz */
  frequencyHz: number;

  /** Optional end frequency for bands */
  endFrequencyHz?: number;

  /** Label text */
  label?: string;

  /** Color (hex) */
  color: string;

  /** User-created flag */
  userCreated: boolean;
}
