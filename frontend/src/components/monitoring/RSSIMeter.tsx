import { useEffect, useState } from 'react';
import { Card } from '@/components/ui/card';
import { Activity, WifiOff } from 'lucide-react';

export interface RSSIMeterProps {
  currentRSSI: number | null; // Current signal strength in dBm, null = no signal
  className?: string;
}

/**
 * RSSI (Received Signal Strength Indicator) Meter
 * Shows real-time signal strength with min/max/average indicators
 */
export function RSSIMeter({ currentRSSI, className = '' }: RSSIMeterProps) {
  const [minRSSI, setMinRSSI] = useState<number | null>(currentRSSI);
  const [maxRSSI, setMaxRSSI] = useState<number | null>(currentRSSI);
  const [avgRSSI, setAvgRSSI] = useState<number | null>(currentRSSI);
  const [samples, setSamples] = useState<number[]>(currentRSSI !== null ? [currentRSSI] : []);

  // Update statistics when RSSI changes
  useEffect(() => {
    if (currentRSSI === null || isNaN(currentRSSI) || !isFinite(currentRSSI)) return;

    setMinRSSI(prev => prev === null ? currentRSSI : Math.min(prev, currentRSSI));
    setMaxRSSI(prev => prev === null ? currentRSSI : Math.max(prev, currentRSSI));

    setSamples(prev => {
      const newSamples = [...prev, currentRSSI].slice(-100); // Keep last 100 samples
      const avg = newSamples.reduce((sum, val) => sum + val, 0) / newSamples.length;
      setAvgRSSI(avg);
      return newSamples;
    });
  }, [currentRSSI]);

  // No signal state
  if (currentRSSI === null) {
    return (
      <Card className={`p-3 ${className}`}>
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <WifiOff className="w-3 h-3 text-muted-foreground" />
              <span className="text-xs font-semibold text-muted-foreground">Signal Strength</span>
            </div>
            <span className="text-xs font-mono text-muted-foreground">
              -- dBm
            </span>
          </div>
          <div className="h-6 bg-secondary/20 rounded flex items-center justify-center">
            <span className="text-xs text-muted-foreground">No Signal</span>
          </div>
          <div className="text-[10px] text-center text-muted-foreground">
            Waiting for spectrum data...
          </div>
        </div>
      </Card>
    );
  }

  // Convert dBm to percentage for visualization (assuming -120 to -20 dBm range)
  const rssiToPercent = (dbm: number) => {
    const min = -120;
    const max = -20;
    return Math.max(0, Math.min(100, ((dbm - min) / (max - min)) * 100));
  };

  // Get color based on signal strength
  const getSignalColor = (dbm: number) => {
    if (dbm >= -50) return 'bg-green-500';
    if (dbm >= -70) return 'bg-yellow-500';
    if (dbm >= -90) return 'bg-orange-500';
    return 'bg-red-500';
  };

  const currentPercent = rssiToPercent(currentRSSI);
  const avgPercent = rssiToPercent(avgRSSI ?? currentRSSI);

  return (
    <Card className={`p-3 ${className}`}>
      <div className="space-y-2">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-3 h-3 text-primary" />
            <span className="text-xs font-semibold">Signal Strength</span>
          </div>
          <span className="text-xs font-mono text-muted-foreground">
            {currentRSSI.toFixed(1)} dBm
          </span>
        </div>

        {/* RSSI Bar Chart */}
        <div className="space-y-1">
          {/* Current RSSI Bar */}
          <div className="relative h-6 bg-secondary/20 rounded overflow-hidden">
            <div
              className={`h-full transition-all duration-300 ${getSignalColor(currentRSSI)}`}
              style={{ width: `${currentPercent}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-[10px] font-medium text-foreground mix-blend-difference">
                Current
              </span>
            </div>
          </div>

          {/* Average RSSI Bar */}
          <div className="relative h-4 bg-secondary/20 rounded overflow-hidden">
            <div
              className="h-full bg-blue-500/60 transition-all duration-500"
              style={{ width: `${avgPercent}%` }}
            />
            <div className="absolute inset-0 flex items-center justify-center">
              <span className="text-[9px] font-medium text-muted-foreground">
                Average
              </span>
            </div>
          </div>
        </div>

        {/* Statistics */}
        <div className="flex items-center justify-between text-[10px] text-muted-foreground font-mono">
          <div className="flex items-center gap-1">
            <span className="text-red-500">▼</span>
            <span>Min: {minRSSI?.toFixed(1) ?? '--'}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-blue-500">●</span>
            <span>Avg: {avgRSSI?.toFixed(1) ?? '--'}</span>
          </div>
          <div className="flex items-center gap-1">
            <span className="text-green-500">▲</span>
            <span>Max: {maxRSSI?.toFixed(1) ?? '--'}</span>
          </div>
        </div>

        {/* Signal Quality Indicator */}
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-muted-foreground">Quality:</span>
          <span className={`font-semibold ${
            currentRSSI >= -50 ? 'text-green-500' :
            currentRSSI >= -70 ? 'text-yellow-500' :
            currentRSSI >= -90 ? 'text-orange-500' :
            'text-red-500'
          }`}>
            {currentRSSI >= -50 ? 'Excellent' :
             currentRSSI >= -70 ? 'Good' :
             currentRSSI >= -90 ? 'Fair' :
             'Poor'}
          </span>
        </div>
      </div>
    </Card>
  );
}
