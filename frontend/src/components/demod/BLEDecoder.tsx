import { useMemo } from 'react';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Bluetooth, Signal } from 'lucide-react';
import type { BLEPacket } from '@/types';

export interface BLEDecoderProps {
  packets: BLEPacket[];
  maxHeight?: number;
}

/**
 * BLE Protocol Decoder
 * Displays decoded Bluetooth Low Energy packets
 */
export function BLEDecoder({ packets, maxHeight = 400 }: BLEDecoderProps) {
  const recentPackets = useMemo(() => {
    return [...packets].reverse().slice(0, 100); // Show last 100 packets
  }, [packets]);

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Bluetooth className="w-4 h-4 text-cyan-500" />
          <h3 className="text-sm font-semibold">BLE Packets</h3>
          <Badge variant="secondary" className="text-xs">
            {packets.length}
          </Badge>
        </div>
      </div>

      <ScrollArea style={{ maxHeight }}>
        <div className="space-y-2">
          {recentPackets.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-sm">
              <Bluetooth className="w-8 h-8 mx-auto mb-2 opacity-50" />
              No BLE packets decoded
            </div>
          ) : (
            recentPackets.map((packet, idx) => (
              <BLEPacketCard key={`${packet.timestamp_ns}-${idx}`} packet={packet} />
            ))
          )}
        </div>
      </ScrollArea>
    </Card>
  );
}

/**
 * BLE Packet Card
 */
interface BLEPacketCardProps {
  packet: BLEPacket;
}

function BLEPacketCard({ packet }: BLEPacketCardProps) {
  const timestamp = new Date(packet.timestamp_ns / 1_000_000);
  const timeStr = timestamp.toLocaleTimeString('en-US', { 
    hour12: false, 
    hour: '2-digit', 
    minute: '2-digit', 
    second: '2-digit',
    fractionalSecondDigits: 3 
  });

  const getPacketTypeColor = (type: string) => {
    switch (type) {
      case 'ADV_IND':
      case 'ADV_DIRECT_IND':
        return 'text-blue-500';
      case 'SCAN_REQ':
      case 'SCAN_RSP':
        return 'text-cyan-500';
      case 'CONNECT_REQ':
        return 'text-green-500';
      default:
        return 'text-muted-foreground';
    }
  };

  return (
    <div className="p-3 rounded-lg border border-border bg-background/50 hover:bg-background/80 transition-colors">
      <div className="flex items-start justify-between mb-2">
        <div className="flex items-center gap-2">
          <Signal className="w-3 h-3 text-cyan-500" />
          <span className="font-mono text-xs text-muted-foreground">{timeStr}</span>
        </div>
        <Badge variant="outline" className="text-xs">
          Ch {packet.channel}
        </Badge>
      </div>

      <div className="grid grid-cols-2 gap-2 text-xs mb-2">
        <div>
          <span className="text-muted-foreground">Type:</span>{' '}
          <span className={`font-mono font-medium ${getPacketTypeColor(packet.packetType)}`}>
            {packet.packetType}
          </span>
        </div>
        <div>
          <span className="text-muted-foreground">RSSI:</span>{' '}
          <span className="font-mono">{packet.rssi} dBm</span>
        </div>
      </div>

      <div className="text-xs mb-2">
        <span className="text-muted-foreground">Access Addr:</span>{' '}
        <span className="font-mono text-primary">
          0x{packet.accessAddress.toString(16).toUpperCase().padStart(8, '0')}
        </span>
      </div>

      {packet.advAddress && (
        <div className="text-xs mb-2">
          <span className="text-muted-foreground">Advertiser:</span>{' '}
          <span className="font-mono">{packet.advAddress}</span>
        </div>
      )}

      {packet.payload && (
        <div className="mt-2 pt-2 border-t border-border">
          <div className="text-xs text-muted-foreground mb-1">
            Payload ({packet.payload.length} bytes):
          </div>
          <div className="font-mono text-xs bg-muted/50 p-2 rounded break-all">
            {packet.payload}
          </div>
        </div>
      )}

      {packet.crcValid !== undefined && (
        <div className="mt-2 text-xs">
          <span className="text-muted-foreground">CRC:</span>{' '}
          <span className={`font-mono ${packet.crcValid ? 'text-green-500' : 'text-red-500'}`}>
            {packet.crcValid ? 'Valid' : 'Invalid'}
          </span>
        </div>
      )}
    </div>
  );
}
