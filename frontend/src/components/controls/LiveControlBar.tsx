import { useState } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Play, Square, Save, Circle } from 'lucide-react';
import { api } from '@/lib/api';
import { toast } from 'sonner';

export interface LiveControlBarProps {
  onStart: () => void;
  onStop: () => void;
  isRunning: boolean;
  selectedDeviceId?: string;
}

/**
 * Live Control Bar - Simple Start/Stop/Save controls for the header
 * Wired directly to the uSDR DevBoard pipeline
 */
export function LiveControlBar({ onStart, onStop, isRunning, selectedDeviceId }: LiveControlBarProps) {
  const [isSaving, setIsSaving] = useState(false);
  const [recordingId, setRecordingId] = useState<string | null>(null);

  const handleStart = async () => {
    if (!selectedDeviceId) {
      toast.error('Select an SDR before starting the stream');
      return;
    }
    try {
      // Start backend stream for the selected device
      await api.startDeviceStream(selectedDeviceId);
      // Start WebSocket streams
      onStart();
      toast.success('Pipeline started');
    } catch (error) {
      console.error('Failed to start:', error);
      toast.error('Failed to start pipeline');
    }
  };

  const handleStop = async () => {
    try {
      // Stop backend pipeline
      if (selectedDeviceId) {
        await api.stopDeviceStream(selectedDeviceId);
      }
      // Stop WebSocket streams
      onStop();
      toast.success('Pipeline stopped');
    } catch (error) {
      console.error('Failed to stop:', error);
      toast.error('Failed to stop pipeline');
    }
  };

  const handleSave = async () => {
    if (isSaving) return;

    try {
      setIsSaving(true);

      // Generate auto-name with timestamp
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
      const name = `capture-${timestamp}`;

      // Start recording
      const startResponse = await api.startRecording({
        name,
        description: 'Quick capture from Live page',
      });

      setRecordingId(startResponse.recording_id);
      toast.success('Recording started...');

      // Auto-stop after 5 seconds for quick capture
      setTimeout(async () => {
        try {
          const stopResponse = await api.stopRecording({
            recording_id: startResponse.recording_id
          });
          const sizeMB = (stopResponse.file_size_bytes / (1024 * 1024)).toFixed(1);
          toast.success(`Saved: ${name} (${sizeMB} MB)`);
        } catch (error) {
          console.error('Failed to stop recording:', error);
          toast.error('Failed to save recording');
        } finally {
          setIsSaving(false);
          setRecordingId(null);
        }
      }, 5000);

    } catch (error) {
      console.error('Failed to save:', error);
      toast.error('Failed to start recording');
      setIsSaving(false);
    }
  };

  return (
    <div className="flex items-center gap-2">
      {/* Recording indicator */}
      {isSaving && (
        <Badge variant="destructive" className="gap-1 animate-pulse">
          <Circle className="w-2 h-2 fill-current" />
          Saving...
        </Badge>
      )}

      {/* Start/Stop Toggle */}
      {!isRunning ? (
        <Button
          size="sm"
          variant="default"
          onClick={handleStart}
          disabled={!selectedDeviceId}
          className="gap-2"
        >
          <Play className="w-4 h-4" />
          Start
        </Button>
      ) : (
        <Button
          size="sm"
          variant="destructive"
          onClick={handleStop}
          className="gap-2"
        >
          <Square className="w-4 h-4" />
          Stop
        </Button>
      )}

      {/* Save Button */}
      <Button
        size="sm"
        variant="secondary"
        onClick={handleSave}
        disabled={!isRunning || isSaving}
        className="gap-2"
      >
        <Save className="w-4 h-4" />
        Save
      </Button>
    </div>
  );
}
