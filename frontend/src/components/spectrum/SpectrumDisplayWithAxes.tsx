import { useRef, useEffect } from 'react';
import { WebGPUSpectrumRenderer, type WebGPUSpectrumRendererRef } from './WebGPUSpectrumRenderer';
import { WebGPUWaterfallRenderer, type WebGPUWaterfallRendererRef } from './WebGPUWaterfallRenderer';
import { FrequencyAxis } from './FrequencyAxis';
import { PowerAxis } from './PowerAxis';
import { useSpectrumStore } from '@/stores/spectrumStore';
import { useConfigStore } from '@/stores/configStore';
import { useUIStore } from '@/stores/uiStore';
import { Card } from '@/components/ui/card';

export interface SpectrumDisplayWithAxesProps {
  width?: number;
  height?: number;
  showAxes?: boolean;
}

/**
 * Spectrum Display with D3 Axes
 * Complete spectrum analyzer with frequency and power axes
 */
export function SpectrumDisplayWithAxes({
  width = 1200,
  height = 600,
  showAxes = true,
}: SpectrumDisplayWithAxesProps) {
  const spectrumRef = useRef<WebGPUSpectrumRendererRef>(null);
  const waterfallRef = useRef<WebGPUWaterfallRendererRef>(null);
  
  const currentPsd = useSpectrumStore((state) => state.currentPsd);
  const centerFreqHz = useSpectrumStore((state) => state.centerFreqHz);
  const spanHz = useSpectrumStore((state) => state.spanHz);
  const displaySettings = useConfigStore((state) => state.displaySettings);
  const colorMap = useUIStore((state) => state.colorMap);

  const minFreqHz = centerFreqHz - spanHz / 2;
  const maxFreqHz = centerFreqHz + spanHz / 2;

  const axisMargin = showAxes ? { left: 60, bottom: 40, right: 10, top: 10 } : { left: 0, bottom: 0, right: 0, top: 0 };
  const spectrumWidth = width - axisMargin.left - axisMargin.right;
  const spectrumHeight = Math.floor(height * 0.4);
  const waterfallHeight = Math.floor(height * 0.6) - axisMargin.bottom;

  /**
   * Update renderers when PSD data changes
   */
  useEffect(() => {
    if (!currentPsd) return;

    spectrumRef.current?.updatePSD(currentPsd);
    waterfallRef.current?.addLine(currentPsd);
  }, [currentPsd]);

  /**
   * Update colormap when changed
   */
  useEffect(() => {
    spectrumRef.current?.setColormap(colorMap);
    waterfallRef.current?.setColormap(colorMap);
  }, [colorMap]);

  /**
   * Update dynamic range when changed
   */
  useEffect(() => {
    const { dynamicRangeMin, dynamicRangeMax } = displaySettings;
    spectrumRef.current?.setDynamicRange(dynamicRangeMin, dynamicRangeMax);
    waterfallRef.current?.setDynamicRange(dynamicRangeMin, dynamicRangeMax);
  }, [displaySettings.dynamicRangeMin, displaySettings.dynamicRangeMax]);

  return (
    <Card className="p-4 bg-background/50 backdrop-blur">
      <div className="space-y-2">
        {/* Spectrum (PSD) */}
        <div className="flex">
          {showAxes && (
            <PowerAxis
              height={spectrumHeight}
              minPowerDb={displaySettings.dynamicRangeMin}
              maxPowerDb={displaySettings.dynamicRangeMax}
            />
          )}
          <div className="flex-1">
            <div className="text-xs font-medium mb-1 text-muted-foreground">
              Power Spectral Density
            </div>
            <WebGPUSpectrumRenderer
              ref={spectrumRef}
              width={spectrumWidth}
              height={spectrumHeight}
              colormap={colorMap}
              dynamicRangeMin={displaySettings.dynamicRangeMin}
              dynamicRangeMax={displaySettings.dynamicRangeMax}
            />
          </div>
        </div>

        {/* Waterfall (Spectrogram) */}
        <div className="flex">
          {showAxes && (
            <PowerAxis
              height={waterfallHeight}
              minPowerDb={displaySettings.dynamicRangeMin}
              maxPowerDb={displaySettings.dynamicRangeMax}
            />
          )}
          <div className="flex-1">
            <div className="text-xs font-medium mb-1 text-muted-foreground">
              Waterfall (Spectrogram)
            </div>
            <WebGPUWaterfallRenderer
              ref={waterfallRef}
              width={spectrumWidth}
              height={waterfallHeight}
              historyDepth={displaySettings.waterfallHistoryDepth}
              colormap={colorMap}
              dynamicRangeMin={displaySettings.dynamicRangeMin}
              dynamicRangeMax={displaySettings.dynamicRangeMax}
            />
            {showAxes && (
              <div className="mt-1">
                <FrequencyAxis
                  width={spectrumWidth}
                  minFreqHz={minFreqHz}
                  maxFreqHz={maxFreqHz}
                />
              </div>
            )}
          </div>
        </div>

        {/* Center frequency indicator */}
        <div className="flex items-center justify-center gap-4 text-xs text-muted-foreground font-mono">
          <span>
            Center: <span className="text-primary font-medium">{(centerFreqHz / 1e6).toFixed(3)} MHz</span>
          </span>
          <span>•</span>
          <span>
            Span: <span className="text-primary font-medium">{(spanHz / 1e6).toFixed(3)} MHz</span>
          </span>
          <span>•</span>
          <span>
            Range: <span className="text-primary font-medium">{displaySettings.dynamicRangeMin} to {displaySettings.dynamicRangeMax} dB</span>
          </span>
        </div>
      </div>
    </Card>
  );
}
