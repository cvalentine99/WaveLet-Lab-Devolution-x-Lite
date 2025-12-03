import { useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Radio, Signal } from 'lucide-react';
import type { LoRaFrame } from '@/types';

export interface LoRaDecoderProps {
  frames: LoRaFrame[];
  maxHeight?: number;
}

/**
 * LoRa Protocol Decoder
 * Displays decoded LoRa packets with spreading factor, bandwidth, and payload
 */
export function LoRaDecoder({ frames, maxHeight = 400 }: LoRaDecoderProps) {
  const recentFrames = useMemo(() => {
    return [...frames].reverse().slice(0, 100); // Show last 100 frames
  }, [frames]);

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Radio className="w-4 h-4 text-blue-500" />
          <h3 className="text-sm font-semibold">LoRa Packets</h3>
          <Badge variant="secondary" className="text-xs">
            {frames.length}
          </Badge>
        </div>
      </div>

      <ScrollArea style={{ maxHeight }}>
        <div className="space-y-2">
          {recentFrames.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-sm">
              <Radio className="w-8 h-8 mx-auto mb-2 opacity-50" />
              No LoRa packets decoded
            </div>
          ) : (
            recentFrames.map((frame, idx) => (
              <LoRaFrameCard key={`${frame.timestamp_ns}-${idx}`} frame={frame} />
            ))
          )}
        </div>
      </ScrollArea>
    </Card>
  );
}

/**
 * LoRa Frame Card
 */
interface LoRaFrameCardProps {
  frame: LoRaFrame;
}

function LoRaFrameCard({ frame }: LoRaFrameCardProps) {
  const timestamp = new Date(frame.timestamp_ns / 1_000_000);
  const timeStr = timestamp.toLocaleTimeString('en-US', { 
    hour12: false, 
    hour: '2-digit', 
    minute: '2-digit', 
    second: '2-digit',
    fractionalSecondDigits: 3 
  });

  return (
    <div className="p-3 rounded-lg border border-border bg-background/50 hover:bg-background/80 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <Signal className="w-3 h-3 text-blue-500" />
          <span className="font-mono text-xs text-muted-foreground">{timeStr}</span>
        </div>
        <Badge variant="outline" className="text-xs">
          SF{frame.spreadingFactor}
        </Badge>
      </div>

      <div className="grid grid-cols-3 gap-2 text-xs mb-2">
        <div>
          <span className="text-muted-foreground">Freq:</span>{' '}
          <span className="font-mono">{(frame.freqHz / 1e6).toFixed(3)} MHz</span>
        </div>
        <div>
          <span className="text-muted-foreground">BW:</span>{' '}
          <span className="font-mono">{(frame.bandwidthHz / 1e3).toFixed(0)} kHz</span>
        </div>
        <div>
          <span className="text-muted-foreground">SNR:</span>{' '}
          <span className="font-mono">{frame.snrDb.toFixed(1)} dB</span>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs mb-2">
        <div>
          <span className="text-muted-foreground">CR:</span>{' '}
          <span className="font-mono">{frame.codingRate}</span>
        </div>
        <div>
          <span className="text-muted-foreground">CRC:</span>{' '}
          <span className={`font-mono ${frame.crcValid ? 'text-green-500' : 'text-red-500'}`}>
            {frame.crcValid ? 'Valid' : 'Invalid'}
          </span>
        </div>
      </div>

      {frame.payload && (
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground mb-1">Payload ({frame.payload.length} bytes):</div>
          <div className="font-mono text-xs bg-muted/50 p-2 rounded break-all">
            {frame.payload}
          </div>
        </div>
      )}
    </div>
  );
}
