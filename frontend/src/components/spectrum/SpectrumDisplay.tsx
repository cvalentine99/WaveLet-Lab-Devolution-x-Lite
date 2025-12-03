import { useRef, useEffect } from 'react';
import { WebGPUSpectrumRenderer, type WebGPUSpectrumRendererRef } from './WebGPUSpectrumRenderer';
import { WebGPUWaterfallRenderer, type WebGPUWaterfallRendererRef } from './WebGPUWaterfallRenderer';
import { useSpectrumStore } from '@/stores/spectrumStore';
import { useConfigStore } from '@/stores/configStore';
import { useUIStore } from '@/stores/uiStore';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

export interface SpectrumDisplayProps {
  width?: number;
  height?: number;
}

/**
 * Spectrum Display
 * Combined PSD and waterfall visualization with WebGPU rendering
 */
export function SpectrumDisplay({ width = 1200, height = 400 }: SpectrumDisplayProps) {
  const spectrumRef = useRef<WebGPUSpectrumRendererRef>(null);
  const waterfallRef = useRef<WebGPUWaterfallRendererRef>(null);
  
  const currentPsd = useSpectrumStore((state) => state.currentPsd);
  const displaySettings = useConfigStore((state) => state.displaySettings);
  const colorMap = useUIStore((state) => state.colorMap);

  /**
   * Update renderers when PSD data changes
   */
  useEffect(() => {
    if (!currentPsd) return;

    // Update spectrum renderer
    spectrumRef.current?.updatePSD(currentPsd);

    // Update waterfall renderer
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
    <Card className="p-4">
      <Tabs defaultValue="combined" className="w-full">
        <TabsList className="mb-4">
          <TabsTrigger value="combined">Combined</TabsTrigger>
          <TabsTrigger value="spectrum">Spectrum Only</TabsTrigger>
          <TabsTrigger value="waterfall">Waterfall Only</TabsTrigger>
        </TabsList>

        <TabsContent value="combined" className="space-y-4">
          <div>
            <h3 className="text-sm font-medium mb-2">Power Spectral Density</h3>
            <WebGPUSpectrumRenderer
              ref={spectrumRef}
              width={width}
              height={height / 2}
              colormap={colorMap}
              dynamicRangeMin={displaySettings.dynamicRangeMin}
              dynamicRangeMax={displaySettings.dynamicRangeMax}
            />
          </div>
          <div>
            <h3 className="text-sm font-medium mb-2">Waterfall (Spectrogram)</h3>
            <WebGPUWaterfallRenderer
              ref={waterfallRef}
              width={width}
              height={height / 2}
              historyDepth={displaySettings.waterfallHistoryDepth}
              colormap={colorMap}
              dynamicRangeMin={displaySettings.dynamicRangeMin}
              dynamicRangeMax={displaySettings.dynamicRangeMax}
            />
          </div>
        </TabsContent>

        <TabsContent value="spectrum">
          <WebGPUSpectrumRenderer
            ref={spectrumRef}
            width={width}
            height={height}
            colormap={colorMap}
            dynamicRangeMin={displaySettings.dynamicRangeMin}
            dynamicRangeMax={displaySettings.dynamicRangeMax}
          />
        </TabsContent>

        <TabsContent value="waterfall">
          <WebGPUWaterfallRenderer
            ref={waterfallRef}
            width={width}
            height={height}
            historyDepth={displaySettings.waterfallHistoryDepth}
            colormap={colorMap}
            dynamicRangeMin={displaySettings.dynamicRangeMin}
            dynamicRangeMax={displaySettings.dynamicRangeMax}
          />
        </TabsContent>
      </Tabs>
    </Card>
  );
}
