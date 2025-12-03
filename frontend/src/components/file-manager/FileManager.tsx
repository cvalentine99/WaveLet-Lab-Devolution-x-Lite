import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
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
import { 
  FileText, 
  Download, 
  Trash2, 
  Radio,
  Clock,
  HardDrive,
  RefreshCw,
  Loader2
} from 'lucide-react';
import { api, type Recording } from '@/lib/api';
import { toast } from 'sonner';

export interface FileManagerProps {
  className?: string;
}

/**
 * File Manager Component
 * Manage RF signal recordings from backend and download SigMF files
 */
export function FileManager({ className }: FileManagerProps) {
  const [recordings, setRecordings] = useState<Recording[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  // Fetch recordings from backend
  const fetchRecordings = async () => {
    try {
      setIsRefreshing(true);
      const response = await api.getRecordings();
      setRecordings(response.recordings);
    } catch (error) {
      console.error('Failed to fetch recordings:', error);
      toast.error('Failed to load recordings');
    } finally {
      setIsRefreshing(false);
    }
  };

  // Load recordings on mount
  useEffect(() => {
    fetchRecordings();
  }, []);

  const handleDownloadRecording = async (recording: Recording) => {
    try {
      setDownloadingId(recording.id);
      const blob = await api.downloadRecording(recording.id);
      
      // Create download link
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${recording.name.replace(/\s+/g, '_')}.zip`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      
      toast.success(`Downloaded ${recording.name}`);
    } catch (error) {
      console.error('Failed to download recording:', error);
      toast.error('Failed to download recording');
    } finally {
      setDownloadingId(null);
    }
  };

  const handleDeleteRecording = async (recording: Recording) => {
    if (!confirm(`Delete recording "${recording.name}"?`)) return;

    try {
      await api.deleteRecording(recording.id);
      setRecordings(recordings.filter(r => r.id !== recording.id));
      toast.success('Recording deleted');
    } catch (error) {
      console.error('Failed to delete recording:', error);
      toast.error('Failed to delete recording');
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  const formatDuration = (seconds: number): string => {
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    if (seconds < 3600) return `${(seconds / 60).toFixed(1)}m`;
    return `${(seconds / 3600).toFixed(1)}h`;
  };

  return (
    <div className={className}>
      <div className="space-y-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileText className="w-5 h-5 text-primary" />
            <h2 className="text-lg font-semibold">Recordings</h2>
            <Badge variant="secondary" className="text-xs">
              {recordings.length}
            </Badge>
          </div>
          <Button
            size="sm"
            variant="outline"
            onClick={fetchRecordings}
            disabled={isRefreshing}
            className="gap-2"
          >
            <RefreshCw className={`w-3 h-3 ${isRefreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
        </div>

        {/* Recordings List */}
        <div className="space-y-2">
          {isLoading ? (
            <Card className="p-8 text-center">
              <Loader2 className="w-8 h-8 mx-auto mb-4 animate-spin text-muted-foreground" />
              <p className="text-sm text-muted-foreground">Loading recordings...</p>
            </Card>
          ) : recordings.length === 0 ? (
            <Card className="p-8 text-center">
              <HardDrive className="w-12 h-12 mx-auto mb-4 text-muted-foreground" />
              <p className="text-sm text-muted-foreground">
                No recordings yet. Start recording from the Live Monitoring page.
              </p>
            </Card>
          ) : (
            recordings.map((recording) => (
              <Card key={recording.id} className="p-4">
                <div className="flex items-start justify-between">
                  <div className="flex-1 space-y-2">
                    <div className="flex items-center gap-2">
                      <h3 className="font-semibold">{recording.name}</h3>
                      <Badge 
                        variant={recording.status === 'recording' ? 'default' : 'secondary'} 
                        className="text-xs"
                      >
                        {recording.status === 'recording' ? (
                          <span className="flex items-center gap-1">
                            <span className="w-2 h-2 bg-red-500 rounded-full animate-pulse" />
                            Recording
                          </span>
                        ) : (
                          formatDuration(recording.duration_seconds)
                        )}
                      </Badge>
                    </div>
                    
                    {recording.description && (
                      <p className="text-sm text-muted-foreground">
                        {recording.description}
                      </p>
                    )}

                    <div className="flex items-center gap-4 text-xs text-muted-foreground">
                      <div className="flex items-center gap-1">
                        <Radio className="w-3 h-3" />
                        <span>{(recording.center_freq_hz / 1e6).toFixed(3)} MHz</span>
                      </div>
                      <div className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        <span>{new Date(recording.created_at).toLocaleString()}</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">
                        {(recording.sample_rate_hz / 1e6).toFixed(1)} Msps
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {recording.num_samples.toLocaleString()} samples
                      </Badge>
                      <Badge variant="outline" className="text-xs">
                        {formatFileSize(recording.file_size_bytes)}
                      </Badge>
                    </div>
                  </div>

                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleDownloadRecording(recording)}
                      disabled={recording.status === 'recording' || downloadingId === recording.id}
                      className="gap-2"
                    >
                      {downloadingId === recording.id ? (
                        <Loader2 className="w-3 h-3 animate-spin" />
                      ) : (
                        <Download className="w-3 h-3" />
                      )}
                      Download
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => handleDeleteRecording(recording)}
                      disabled={recording.status === 'recording'}
                      className="gap-2 text-destructive hover:text-destructive"
                    >
                      <Trash2 className="w-3 h-3" />
                    </Button>
                  </div>
                </div>
              </Card>
            ))
          )}
        </div>

        <div className="text-xs text-muted-foreground space-y-1 pt-2 border-t">
          <p className="font-semibold">SigMF Format</p>
          <p>Downloads include .sigmf-meta (JSON metadata) and .sigmf-data (binary IQ samples)</p>
          <p>Datatype: cf32_le (complex float32, little-endian, 8 bytes per sample)</p>
        </div>
      </div>
    </div>
  );
}
