import { useState } from 'react';
import { Card } from '@/components/ui/card';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { LoRaDecoder } from './LoRaDecoder';
import { BLEDecoder } from './BLEDecoder';
import type { LoRaFrame, BLEPacket } from '@/types';

export interface DecoderPanelProps {
  loraFrames: LoRaFrame[];
  blePackets: BLEPacket[];
  maxHeight?: number;
}

/**
 * Protocol Decoder Panel
 * Tabbed interface for viewing decoded LoRa and BLE packets
 */
export function DecoderPanel({ 
  loraFrames, 
  blePackets, 
  maxHeight = 500 
}: DecoderPanelProps) {
  const [activeTab, setActiveTab] = useState<string>('lora');

  return (
    <Card className="p-4">
      <Tabs value={activeTab} onValueChange={setActiveTab}>
        <TabsList className="grid w-full grid-cols-2">
          <TabsTrigger value="lora" className="text-xs">
            LoRa
            {loraFrames.length > 0 && (
              <span className="ml-2 px-1.5 py-0.5 rounded-full bg-blue-500/20 text-blue-500 text-xs">
                {loraFrames.length}
              </span>
            )}
          </TabsTrigger>
          <TabsTrigger value="ble" className="text-xs">
            BLE
            {blePackets.length > 0 && (
              <span className="ml-2 px-1.5 py-0.5 rounded-full bg-cyan-500/20 text-cyan-500 text-xs">
                {blePackets.length}
              </span>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="lora" className="mt-4">
          <LoRaDecoder frames={loraFrames} maxHeight={maxHeight} />
        </TabsContent>

        <TabsContent value="ble" className="mt-4">
          <BLEDecoder packets={blePackets} maxHeight={maxHeight} />
        </TabsContent>
      </Tabs>
    </Card>
  );
}
