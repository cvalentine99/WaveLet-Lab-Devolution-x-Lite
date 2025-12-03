import { useState, useEffect } from 'react';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Circle, Square, Loader2, Download } from 'lucide-react';
import { api } from '@/lib/api';
import { toast } from 'sonner';

export interface RecordingControlsProps {
  className?: string;
  onRecordingComplete?: () => void;
}

/**
 * Recording Controls Component
 * Start/stop IQ sample recording to SigMF format
 */
export function RecordingControls({ className, onRecordingComplete }: RecordingControlsProps) {
  const [isRecording, setIsRecording] = useState(false);
  const [recordingId, setRecordingId] = useState<string | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [showStartDialog, setShowStartDialog] = useState(false);
  const [showExportDialog, setShowExportDialog] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  
  // Recording metadata
  const [recordingName, setRecordingName] = useState('');
  const [recordingDescription, setRecordingDescription] = useState('');
  const [autoDuration, setAutoDuration] = useState<number | undefined>(undefined);

  // Timer effect
  useEffect(() => {
    if (!isRecording) return;
    
    const interval = setInterval(() => {
      setRecordingDuration((prev) => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, [isRecording]);

  const handleStartRecording = async () => {
    if (!recordingName.trim()) {
      toast.error('Please enter a recording name');
      return;
    }

    try {
      const response = await api.startRecording({
        name: recordingName,
        description: recordingDescription,
        duration_seconds: autoDuration,
      });

      setRecordingId(response.recording_id);
      setIsRecording(true);
      setRecordingDuration(0);
      setShowStartDialog(false);
      toast.success('Recording started');

      // Auto-stop if duration is set
      if (autoDuration) {
        setTimeout(() => {
          handleStopRecording();
        }, autoDuration * 1000);
      }
    } catch (error) {
      console.error('Failed to start recording:', error);
      toast.error('Failed to start recording');
    }
  };

  const handleStopRecording = async () => {
    if (!recordingId) return;

    try {
      const response = await api.stopRecording({ recording_id: recordingId });
      setIsRecording(false);
      toast.success(`Recording stopped: ${(response.file_size_bytes / (1024 * 1024)).toFixed(1)} MB`);
      onRecordingComplete?.();
      
      // Reset state
      setRecordingId(null);
      setRecordingName('');
      setRecordingDescription('');
      setAutoDuration(undefined);
    } catch (error) {
      console.error('Failed to stop recording:', error);
      toast.error('Failed to stop recording');
    }
  };

  const handleExportDetections = async () => {
    try {
      setIsExporting(true);
      const blob = await api.exportDetectionsJSON();
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
      a.download = `detections-${timestamp}.json`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.success('Detections exported');
      setShowExportDialog(false);
    } catch (error) {
      console.error('Failed to export detections:', error);
      toast.error('Failed to export detections');
    } finally {
      setIsExporting(false);
    }
  };

  const formatDuration = (seconds: number): string => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    
    if (h > 0) return `${h}:${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className={className}>
      <div className="flex items-center gap-2">
        {isRecording ? (
          <>
            <Badge variant="destructive" className="gap-1 animate-pulse">
              <Circle className="w-2 h-2 fill-current" />
              Recording
            </Badge>
            <span className="text-xs font-mono text-muted-foreground">
              {formatDuration(recordingDuration)}
            </span>
            <Button
              size="sm"
              variant="outline"
              onClick={handleStopRecording}
              className="gap-2"
            >
              <Square className="w-3 h-3" />
              Stop
            </Button>
          </>
        ) : (
          <>
            <Button
              size="sm"
              variant="default"
              onClick={() => setShowStartDialog(true)}
              className="gap-2"
            >
              <Circle className="w-3 h-3" />
              Start Recording
            </Button>
            <Button
              size="sm"
              variant="outline"
              onClick={() => setShowExportDialog(true)}
              className="gap-2"
            >
              <Download className="w-3 h-3" />
              Export Detections
            </Button>
          </>
        )}
      </div>

      {/* Start Recording Dialog */}
      <Dialog open={showStartDialog} onOpenChange={setShowStartDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Start Recording</DialogTitle>
            <DialogDescription>
              Record IQ samples to SigMF format (.sigmf-meta + .sigmf-data)
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <Label htmlFor="name">Recording Name *</Label>
              <Input
                id="name"
                placeholder="e.g., 915 MHz LoRa Capture"
                value={recordingName}
                onChange={(e) => setRecordingName(e.target.value)}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="description">Description</Label>
              <Textarea
                id="description"
                placeholder="ISM band monitoring session..."
                value={recordingDescription}
                onChange={(e) => setRecordingDescription(e.target.value)}
                rows={3}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="duration">Auto-stop Duration (seconds, optional)</Label>
              <Input
                id="duration"
                type="number"
                placeholder="Leave empty for manual stop"
                value={autoDuration || ''}
                onChange={(e) => setAutoDuration(e.target.value ? Number(e.target.value) : undefined)}
              />
            </div>

            <div className="text-xs text-muted-foreground space-y-1 p-3 bg-muted rounded-md">
              <p className="font-semibold">Recording Format</p>
              <p>• Datatype: cf32_le (complex float32, little-endian)</p>
              <p>• 8 bytes per sample (4 I + 4 Q interleaved)</p>
              <p>• Files: .sigmf-meta (JSON) + .sigmf-data (binary)</p>
            </div>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowStartDialog(false)}>
              Cancel
            </Button>
            <Button onClick={handleStartRecording} className="gap-2">
              <Circle className="w-3 h-3" />
              Start Recording
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Export Detections Dialog */}
      <Dialog open={showExportDialog} onOpenChange={setShowExportDialog}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Export Detections</DialogTitle>
            <DialogDescription>
              Download all detections as JSON file
            </DialogDescription>
          </DialogHeader>

          <div className="py-4">
            <p className="text-sm text-muted-foreground">
              This will export all current detections with their metadata including frequency, 
              bandwidth, SNR, timestamps, and other parameters.
            </p>
          </div>

          <DialogFooter>
            <Button variant="outline" onClick={() => setShowExportDialog(false)}>
              Cancel
            </Button>
            <Button 
              onClick={handleExportDetections} 
              disabled={isExporting}
              className="gap-2"
            >
              {isExporting ? (
                <>
                  <Loader2 className="w-3 h-3 animate-spin" />
                  Exporting...
                </>
              ) : (
                <>
                  <Download className="w-3 h-3" />
                  Export JSON
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
