/**
 * Preset Types - Configuration presets
 */

import type { SDRConfig } from './sdr';
import type { DisplaySettings } from './display';

/**
 * Configuration preset
 */
export interface Preset {
  /** Preset name */
  name: string;

  /** SDR configuration */
  sdrConfig: SDRConfig;

  /** Display settings */
  displaySettings: DisplaySettings;

  /** Description */
  description?: string;

  /** Creation timestamp */
  createdAt: number;
}
