import { useRef, useEffect, useState, useCallback } from 'react';
import { WebGL2SpectrumRenderer, type WebGL2SpectrumRendererRef } from './WebGL2SpectrumRenderer';
import { WebGL2WaterfallRenderer, type WebGL2WaterfallRendererRef } from './WebGL2WaterfallRenderer';
import { CanvasSpectrumRenderer, type CanvasSpectrumRendererRef } from './CanvasSpectrumRenderer';
import { CanvasWaterfallRenderer, type CanvasWaterfallRendererRef } from './CanvasWaterfallRenderer';
import { FrequencyAxis } from './FrequencyAxis';
import { PowerAxis } from './PowerAxis';
import { SpectrumOverlay } from './SpectrumOverlay';
import { useSpectrumStore } from '@/stores/spectrumStore';
import { useConfigStore } from '@/stores/configStore';
import { useUIStore } from '@/stores/uiStore';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { PerformanceProfiler } from '@/components/performance/PerformanceProfiler';
import { DetectionOverlay } from './DetectionOverlay';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { AlertCircle, TrendingUp, Activity, Trash2, X } from 'lucide-react';

/**
 * Check if WebGL2 is available
 */
function checkWebGL2Available(): boolean {
  try {
    const canvas = document.createElement('canvas');
    const gl = canvas.getContext('webgl2');
    if (!gl) return false;

    // Check for required extensions
    const floatExt = gl.getExtension('EXT_color_buffer_float');
    const floatLinear = gl.getExtension('OES_texture_float_linear');

    // Cleanup
    const loseContext = gl.getExtension('WEBGL_lose_context');
    loseContext?.loseContext();

    return true;
  } catch {
    return false;
  }
}

export interface AdaptiveSpectrumDisplayProps {
  width?: number;
  height?: number;
  showAxes?: boolean;
}

/**
 * Adaptive Spectrum Display
 * Automatically uses WebGPU or falls back to Canvas 2D
 */
export function AdaptiveSpectrumDisplay({
  width = 1200,
  height = 600,
  showAxes = true,
}: AdaptiveSpectrumDisplayProps) {
  return (
    <PerformanceProfiler id="AdaptiveSpectrumDisplay" slowThresholdMs={16}>
      <AdaptiveSpectrumDisplayInner width={width} height={height} showAxes={showAxes} />
    </PerformanceProfiler>
  );
}

// Session storage key for dismissed warning
const WEBGL2_WARNING_DISMISSED_KEY = 'webgl2-warning-dismissed';

function AdaptiveSpectrumDisplayInner({
  width = 1200,
  height = 600,
  showAxes = true,
}: AdaptiveSpectrumDisplayProps) {
  const [useWebGL2, setUseWebGL2] = useState(false);
  const [checkingWebGL2, setCheckingWebGL2] = useState(true);
  const [warningDismissed, setWarningDismissed] = useState(() => {
    // Check if warning was already dismissed this session
    if (typeof window !== 'undefined') {
      return sessionStorage.getItem(WEBGL2_WARNING_DISMISSED_KEY) === 'true';
    }
    return false;
  });

  const webgl2SpectrumRef = useRef<WebGL2SpectrumRendererRef>(null);
  const webgl2WaterfallRef = useRef<WebGL2WaterfallRendererRef>(null);
  const canvasSpectrumRef = useRef<CanvasSpectrumRendererRef>(null);
  const canvasWaterfallRef = useRef<CanvasWaterfallRendererRef>(null);

  const currentPsd = useSpectrumStore((state) => state.currentPsd);
  const centerFreqHz = useSpectrumStore((state) => state.centerFreqHz);
  const spanHz = useSpectrumStore((state) => state.spanHz);
  const displaySettings = useConfigStore((state) => state.displaySettings);
  const colorMap = useUIStore((state) => state.colorMap);

  // Peak hold and average state
  const peakHoldEnabled = useSpectrumStore((state) => state.peakHoldEnabled);
  const avgEnabled = useSpectrumStore((state) => state.avgEnabled);
  const markers = useSpectrumStore((state) => state.markers);
  const togglePeakHold = useSpectrumStore((state) => state.togglePeakHold);
  const toggleAverage = useSpectrumStore((state) => state.toggleAverage);
  const clearPeakHold = useSpectrumStore((state) => state.clearPeakHold);
  const clearMarkers = useSpectrumStore((state) => state.clearMarkers);

  const minFreqHz = centerFreqHz - spanHz / 2;
  const maxFreqHz = centerFreqHz + spanHz / 2;

  const axisMargin = showAxes ? { left: 60, bottom: 40, right: 10, top: 10 } : { left: 0, bottom: 0, right: 0, top: 0 };
  const spectrumWidth = width - axisMargin.left - axisMargin.right;
  const spectrumHeight = Math.floor(height * 0.4);
  const waterfallHeight = Math.floor(height * 0.6) - axisMargin.bottom;

  const dismissWarning = useCallback(() => {
    setWarningDismissed(true);
    if (typeof window !== 'undefined') {
      sessionStorage.setItem(WEBGL2_WARNING_DISMISSED_KEY, 'true');
    }
  }, []);

  /**
   * Check WebGL2 availability (synchronous check)
   */
  useEffect(() => {
    const supported = checkWebGL2Available();
    setUseWebGL2(supported);

    if (!supported) {
      console.log('[AdaptiveSpectrumDisplay] Using Canvas 2D fallback (WebGL2 not available)');
    } else {
      console.log('[AdaptiveSpectrumDisplay] WebGL2 rendering enabled');
    }

    setCheckingWebGL2(false);
  }, []);

  /**
   * Update renderers when PSD data changes
   */
  useEffect(() => {
    if (!currentPsd) return;

    if (useWebGL2) {
      webgl2SpectrumRef.current?.updatePSD(currentPsd);
      webgl2WaterfallRef.current?.addLine(currentPsd);
    } else {
      canvasSpectrumRef.current?.updatePSD(currentPsd);
      canvasWaterfallRef.current?.addLine(currentPsd);
    }
  }, [currentPsd, useWebGL2]);

  /**
   * Update colormap when changed
   */
  useEffect(() => {
    if (useWebGL2) {
      webgl2SpectrumRef.current?.setColormap(colorMap);
      webgl2WaterfallRef.current?.setColormap(colorMap);
    } else {
      canvasSpectrumRef.current?.setColormap(colorMap);
      canvasWaterfallRef.current?.setColormap(colorMap);
    }
  }, [colorMap, useWebGL2]);

  /**
   * Update dynamic range when changed
   */
  useEffect(() => {
    const { dynamicRangeMin, dynamicRangeMax } = displaySettings;

    if (useWebGL2) {
      webgl2SpectrumRef.current?.setDynamicRange(dynamicRangeMin, dynamicRangeMax);
      webgl2WaterfallRef.current?.setDynamicRange(dynamicRangeMin, dynamicRangeMax);
    } else {
      canvasSpectrumRef.current?.setDynamicRange(dynamicRangeMin, dynamicRangeMax);
      canvasWaterfallRef.current?.setDynamicRange(dynamicRangeMin, dynamicRangeMax);
    }
  }, [displaySettings.dynamicRangeMin, displaySettings.dynamicRangeMax, useWebGL2]);

  if (checkingWebGL2) {
    return (
      <Card className="p-4 bg-background/50 backdrop-blur">
        <div className="flex items-center justify-center h-96">
          <div className="text-muted-foreground">Initializing renderer...</div>
        </div>
      </Card>
    );
  }

  return (
    <Card className="p-4 bg-background/50 backdrop-blur">
      {!useWebGL2 && !warningDismissed && (
        <Alert className="mb-4 relative">
          <AlertCircle className="h-4 w-4" />
          <AlertDescription className="pr-8">
            Using Canvas 2D renderer. For GPU-accelerated performance, use a modern browser with WebGL2 support.
          </AlertDescription>
          <Button
            variant="ghost"
            size="sm"
            className="absolute top-2 right-2 h-6 w-6 p-0"
            onClick={dismissWarning}
            title="Dismiss"
          >
            <X className="h-4 w-4" />
          </Button>
        </Alert>
      )}

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
            <div className="relative">
              {useWebGL2 ? (
                <WebGL2SpectrumRenderer
                  ref={webgl2SpectrumRef}
                  width={spectrumWidth}
                  height={spectrumHeight}
                  colormap={colorMap}
                  dynamicRangeMin={displaySettings.dynamicRangeMin}
                  dynamicRangeMax={displaySettings.dynamicRangeMax}
                />
              ) : (
                <CanvasSpectrumRenderer
                  ref={canvasSpectrumRef}
                  width={spectrumWidth}
                  height={spectrumHeight}
                  colormap={colorMap}
                  dynamicRangeMin={displaySettings.dynamicRangeMin}
                  dynamicRangeMax={displaySettings.dynamicRangeMax}
                />
              )}
              <DetectionOverlay
                width={spectrumWidth}
                height={spectrumHeight}
                centerFreqHz={centerFreqHz}
                spanHz={spanHz}
                minPowerDbm={displaySettings.dynamicRangeMin}
                maxPowerDbm={displaySettings.dynamicRangeMax}
              />
              <SpectrumOverlay
                width={spectrumWidth}
                height={spectrumHeight}
                centerFreqHz={centerFreqHz}
                spanHz={spanHz}
                minPowerDb={displaySettings.dynamicRangeMin}
                maxPowerDb={displaySettings.dynamicRangeMax}
              />
            </div>
            {/* Spectrum Control Buttons */}
            <div className="flex items-center gap-1 mt-1">
              <Button
                size="sm"
                variant={peakHoldEnabled ? "default" : "outline"}
                onClick={togglePeakHold}
                className="h-6 text-[10px] gap-1 px-2"
                title="Peak Hold (P)"
              >
                <TrendingUp className="w-3 h-3" />
                Peak
              </Button>
              <Button
                size="sm"
                variant={avgEnabled ? "default" : "outline"}
                onClick={toggleAverage}
                className="h-6 text-[10px] gap-1 px-2"
                title="Average Trace"
              >
                <Activity className="w-3 h-3" />
                Avg
              </Button>
              {markers.length > 0 && (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={clearMarkers}
                  className="h-6 text-[10px] gap-1 px-2"
                  title="Clear Markers (C)"
                >
                  <Trash2 className="w-3 h-3" />
                  {markers.length} Markers
                </Button>
              )}
              <span className="text-[10px] text-muted-foreground ml-auto">
                Click spectrum to add marker
              </span>
            </div>
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
            {useWebGL2 ? (
              <WebGL2WaterfallRenderer
                ref={webgl2WaterfallRef}
                width={spectrumWidth}
                height={waterfallHeight}
                historyDepth={displaySettings.waterfallHistoryDepth}
                colormap={colorMap}
                dynamicRangeMin={displaySettings.dynamicRangeMin}
                dynamicRangeMax={displaySettings.dynamicRangeMax}
              />
            ) : (
              <CanvasWaterfallRenderer
                ref={canvasWaterfallRef}
                width={spectrumWidth}
                height={waterfallHeight}
                historyDepth={displaySettings.waterfallHistoryDepth}
                colormap={colorMap}
                dynamicRangeMin={displaySettings.dynamicRangeMin}
                dynamicRangeMax={displaySettings.dynamicRangeMax}
              />
            )}
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
