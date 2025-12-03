import { useRef, useEffect, useState, useCallback } from 'react';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Label } from '@/components/ui/label';
import { Badge } from '@/components/ui/badge';
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from '@/components/ui/popover';
import { Settings2, Activity, Waves, Radio, Target } from 'lucide-react';
import { AdaptiveSpectrumDisplay } from './AdaptiveSpectrumDisplay';
import { PersistenceSpectrum, type PersistenceSpectrumRef } from './PersistenceSpectrum';
import { IQConstellation, type IQConstellationRef } from './IQConstellation';
import { useSpectrumStore, selectCurrentIQ } from '@/stores/spectrumStore';
import { useConfigStore } from '@/stores/configStore';
import { startIQStream, stopIQStream, areStreamsConnected } from '@/services/websocket';

export interface EnhancedSpectrumDisplayProps {
  width?: number;
  height?: number;
  className?: string;
}

/**
 * Enhanced Spectrum Display
 *
 * Combines multiple visualization modes for comprehensive RF analysis:
 * - Spectrum + Waterfall (default)
 * - Persistence Spectrum (signal density over time)
 * - IQ Constellation (modulation analysis)
 */
export function EnhancedSpectrumDisplay({
  width: propWidth,
  height: propHeight = 500,
  className,
}: EnhancedSpectrumDisplayProps) {
  const [activeView, setActiveView] = useState('spectrum');
  const [decayRate, setDecayRate] = useState(0.02);
  const [iqConnected, setIqConnected] = useState(false);
  const [useSyntheticIQ, setUseSyntheticIQ] = useState(true);
  const [containerWidth, setContainerWidth] = useState(800);

  const containerRef = useRef<HTMLDivElement>(null);
  const persistenceRef = useRef<PersistenceSpectrumRef>(null);
  const constellationRef = useRef<IQConstellationRef>(null);

  // Auto-size to container width
  useEffect(() => {
    if (!containerRef.current) return;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        setContainerWidth(entry.contentRect.width - 32); // Account for padding
      }
    });
    observer.observe(containerRef.current);
    // Initial measurement
    setContainerWidth(containerRef.current.clientWidth - 32);
    return () => observer.disconnect();
  }, []);

  const width = propWidth ?? containerWidth;
  const height = propHeight;

  const currentPsd = useSpectrumStore((state) => state.currentPsd);
  const currentIQ = useSpectrumStore(selectCurrentIQ);
  const displaySettings = useConfigStore((state) => state.displaySettings);

  // Start/stop IQ stream when constellation view is active
  useEffect(() => {
    if (activeView === 'constellation') {
      startIQStream();
      // Check connection status periodically
      const checkInterval = setInterval(() => {
        const connected = areStreamsConnected().iq;
        setIqConnected(connected);
        // Switch to real IQ data if connected
        if (connected) {
          setUseSyntheticIQ(false);
        }
      }, 1000);
      return () => {
        clearInterval(checkInterval);
        stopIQStream();
      };
    }
  }, [activeView]);

  // Feed PSD to persistence display
  useEffect(() => {
    if (currentPsd && activeView === 'persistence') {
      persistenceRef.current?.addPSD(currentPsd);
    }
  }, [currentPsd, activeView]);

  // Feed real IQ data to constellation when available
  useEffect(() => {
    if (currentIQ && activeView === 'constellation' && !useSyntheticIQ) {
      constellationRef.current?.addSamples(currentIQ.i, currentIQ.q);
    }
  }, [currentIQ, activeView, useSyntheticIQ]);

  // Fallback: Generate synthetic IQ from PSD when real IQ not available
  useEffect(() => {
    if (currentPsd && activeView === 'constellation' && useSyntheticIQ) {
      const len = Math.min(256, currentPsd.length);
      const iData = new Float32Array(len);
      const qData = new Float32Array(len);

      for (let i = 0; i < len; i++) {
        const power = Math.pow(10, currentPsd[i] / 20);
        const phase = Math.random() * Math.PI * 2;
        const noise = 0.1;

        iData[i] = power * Math.cos(phase) + (Math.random() - 0.5) * noise;
        qData[i] = power * Math.sin(phase) + (Math.random() - 0.5) * noise;
      }

      constellationRef.current?.addSamples(iData, qData);
    }
  }, [currentPsd, activeView, useSyntheticIQ]);

  // Update decay rate
  useEffect(() => {
    persistenceRef.current?.setDecayRate(decayRate);
    constellationRef.current?.setDecayRate(decayRate);
  }, [decayRate]);

  const handleClearPersistence = () => {
    persistenceRef.current?.clear();
  };

  const handleClearConstellation = () => {
    constellationRef.current?.clear();
  };

  return (
    <Card ref={containerRef} className={`p-4 bg-background/50 backdrop-blur w-full ${className || ''}`}>
      <div className="space-y-4">
        {/* Header with tabs and settings */}
        <div className="flex items-center justify-between">
          <Tabs value={activeView} onValueChange={setActiveView} className="w-auto">
            <TabsList className="grid grid-cols-3 w-auto">
              <TabsTrigger value="spectrum" className="gap-2 px-4">
                <Waves className="w-4 h-4" />
                Spectrum
              </TabsTrigger>
              <TabsTrigger value="persistence" className="gap-2 px-4">
                <Activity className="w-4 h-4" />
                Persistence
              </TabsTrigger>
              <TabsTrigger value="constellation" className="gap-2 px-4">
                <Target className="w-4 h-4" />
                IQ
              </TabsTrigger>
            </TabsList>
          </Tabs>

          <div className="flex items-center gap-2">
            {activeView === 'persistence' && (
              <Button variant="outline" size="sm" onClick={handleClearPersistence}>
                Clear
              </Button>
            )}
            {activeView === 'constellation' && (
              <>
                <Badge
                  variant={iqConnected ? "default" : "outline"}
                  className="text-xs"
                >
                  {iqConnected ? "Live IQ" : useSyntheticIQ ? "Synthetic" : "Connecting..."}
                </Badge>
                {useSyntheticIQ && !iqConnected && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => setUseSyntheticIQ(false)}
                    className="text-xs"
                  >
                    Try Live
                  </Button>
                )}
                <Button variant="outline" size="sm" onClick={handleClearConstellation}>
                  Clear
                </Button>
              </>
            )}

            <Popover>
              <PopoverTrigger asChild>
                <Button variant="outline" size="sm" className="gap-2">
                  <Settings2 className="w-4 h-4" />
                  Settings
                </Button>
              </PopoverTrigger>
              <PopoverContent className="w-80">
                <div className="space-y-4">
                  <h4 className="font-medium">Display Settings</h4>

                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <Label className="text-sm">Decay Rate</Label>
                      <span className="text-sm text-muted-foreground">
                        {decayRate.toFixed(3)}
                      </span>
                    </div>
                    <Slider
                      min={0.005}
                      max={0.2}
                      step={0.005}
                      value={[decayRate]}
                      onValueChange={([v]) => setDecayRate(v)}
                    />
                    <p className="text-xs text-muted-foreground">
                      Lower = longer trails, Higher = faster decay
                    </p>
                  </div>

                  <div className="pt-2 border-t">
                    <p className="text-xs text-muted-foreground">
                      <strong>Spectrum:</strong> Real-time PSD + waterfall<br />
                      <strong>Persistence:</strong> Signal density heatmap<br />
                      <strong>IQ:</strong> Constellation diagram (live or synthetic)
                    </p>
                  </div>
                </div>
              </PopoverContent>
            </Popover>
          </div>
        </div>

        {/* View content */}
        <div className="min-h-[500px]">
          {activeView === 'spectrum' && (
            <AdaptiveSpectrumDisplay width={width} height={height} showAxes={true} />
          )}

          {activeView === 'persistence' && (
            <div className="space-y-2">
              <PersistenceSpectrum
                ref={persistenceRef}
                width={width}
                height={height - 50}
                decayRate={decayRate}
                dynamicRangeMin={displaySettings.dynamicRangeMin}
                dynamicRangeMax={displaySettings.dynamicRangeMax}
              />
              <div className="text-xs text-muted-foreground text-center">
                Brighter areas indicate more frequent signal activity at that frequency/power level
              </div>
            </div>
          )}

          {activeView === 'constellation' && (
            <div className="flex gap-4">
              <div className="flex-1">
                <IQConstellation
                  ref={constellationRef}
                  width={Math.min(height - 50, width / 2)}
                  height={height - 50}
                  decayRate={decayRate}
                  maxPoints={2048}
                  gridLines={true}
                  showStats={true}
                />
              </div>
              <div className="flex-1 p-4 bg-muted/30 rounded-lg">
                <h4 className="font-medium mb-3">IQ Constellation Analysis</h4>
                <div className="space-y-3 text-sm">
                  <div>
                    <div className="text-muted-foreground">What it shows:</div>
                    <p>Each point represents one IQ sample. The pattern reveals modulation type.</p>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Common patterns:</div>
                    <ul className="list-disc list-inside space-y-1 text-xs">
                      <li><strong>Single blob:</strong> Noise / no signal</li>
                      <li><strong>Circle:</strong> FM / constant envelope</li>
                      <li><strong>4 points:</strong> QPSK</li>
                      <li><strong>16 points:</strong> 16-QAM</li>
                      <li><strong>Ring of points:</strong> PSK</li>
                    </ul>
                  </div>
                  <div className="pt-2 border-t">
                    <Badge
                      variant={iqConnected ? "default" : "secondary"}
                      className="text-xs"
                    >
                      {iqConnected
                        ? "Receiving live IQ samples from backend"
                        : useSyntheticIQ
                        ? "Showing synthetic IQ data. Click 'Try Live' for real samples."
                        : "Connecting to IQ stream..."
                      }
                    </Badge>
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </Card>
  );
}
