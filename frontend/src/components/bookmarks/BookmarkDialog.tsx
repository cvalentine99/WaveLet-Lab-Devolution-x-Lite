import { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Textarea } from '@/components/ui/textarea';
import { Badge } from '@/components/ui/badge';
import { useBookmarkStore } from '@/stores/bookmarkStore';
import { Bookmark, X } from 'lucide-react';
import { toast } from 'sonner';
import { captureSpectrumSnapshot } from '@/lib/spectrumCapture';
import { Camera, Image as ImageIcon } from 'lucide-react';

interface BookmarkDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  initialData?: {
    centerFreqHz: number;
    powerDbm: number;
    snrDb?: number;
    bandwidthHz?: number;
    modulationType?: string;
    detectionId?: string;
  };
}

export function BookmarkDialog({ open, onOpenChange, initialData }: BookmarkDialogProps) {
  const { addBookmark } = useBookmarkStore();
  const [formData, setFormData] = useState({
    name: '',
    notes: '',
    centerFreqHz: initialData?.centerFreqHz || 900000000,
    bandwidthHz: initialData?.bandwidthHz,
    powerDbm: initialData?.powerDbm || -60,
    snrDb: initialData?.snrDb,
    modulationType: initialData?.modulationType || '',
    tags: [] as string[],
    spectrumSnapshot: undefined as string | undefined,
  });
  const [tagInput, setTagInput] = useState('');
  const [hasSnapshot, setHasSnapshot] = useState(false);

  useEffect(() => {
    if (initialData && open) {
      setFormData((prev) => ({
        ...prev,
        centerFreqHz: initialData.centerFreqHz,
        bandwidthHz: initialData.bandwidthHz,
        powerDbm: initialData.powerDbm,
        snrDb: initialData.snrDb,
        modulationType: initialData.modulationType || '',
        spectrumSnapshot: (initialData as any).spectrumSnapshot,
      }));
      setHasSnapshot(!!(initialData as any).spectrumSnapshot);
    }
  }, [initialData, open]);

  const handleAddTag = () => {
    const tag = tagInput.trim();
    if (tag && !formData.tags.includes(tag)) {
      setFormData({ ...formData, tags: [...formData.tags, tag] });
      setTagInput('');
    }
  };

  const handleRemoveTag = (tag: string) => {
    setFormData({ ...formData, tags: formData.tags.filter((t) => t !== tag) });
  };

  const handleCaptureSnapshot = () => {
    const snapshot = captureSpectrumSnapshot();
    if (snapshot) {
      setFormData({ ...formData, spectrumSnapshot: snapshot });
      setHasSnapshot(true);
      toast.success('Spectrum snapshot captured');
    } else {
      toast.error('Failed to capture spectrum snapshot');
    }
  };

  const handleRemoveSnapshot = () => {
    setFormData({ ...formData, spectrumSnapshot: undefined });
    setHasSnapshot(false);
  };

  const handleSave = () => {
    if (!formData.name.trim()) {
      toast.error('Please enter a name for the bookmark');
      return;
    }

    addBookmark({
      name: formData.name,
      notes: formData.notes || undefined,
      centerFreqHz: formData.centerFreqHz,
      bandwidthHz: formData.bandwidthHz,
      powerDbm: formData.powerDbm,
      snrDb: formData.snrDb,
      modulationType: formData.modulationType || undefined,
      detectionId: initialData?.detectionId,
      tags: formData.tags,
      spectrumSnapshot: formData.spectrumSnapshot,
    });

    toast.success(`Bookmark "${formData.name}" saved`);
    setFormData({
      name: '',
      notes: '',
      centerFreqHz: 900000000,
      bandwidthHz: undefined,
      powerDbm: -60,
      snrDb: undefined,
      modulationType: '',
      tags: [],
      spectrumSnapshot: undefined,
    });
    setTagInput('');
    setHasSnapshot(false);
    onOpenChange(false);
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Bookmark className="w-5 h-5" />
            Bookmark Signal
          </DialogTitle>
          <DialogDescription>
            Save this signal for future reference with notes and tags
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          <div>
            <Label htmlFor="name">Bookmark Name *</Label>
            <Input
              id="name"
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              placeholder="e.g., Suspicious GSM Signal"
            />
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <Label htmlFor="freq">Center Frequency (Hz)</Label>
              <Input
                id="freq"
                type="number"
                value={formData.centerFreqHz}
                onChange={(e) => setFormData({ ...formData, centerFreqHz: Number(e.target.value) })}
              />
              <p className="text-xs text-zinc-500 mt-1">
                {(formData.centerFreqHz / 1e6).toFixed(3)} MHz
              </p>
            </div>

            <div>
              <Label htmlFor="bandwidth">Bandwidth (Hz)</Label>
              <Input
                id="bandwidth"
                type="number"
                value={formData.bandwidthHz || ''}
                onChange={(e) =>
                  setFormData({ ...formData, bandwidthHz: e.target.value ? Number(e.target.value) : undefined })
                }
                placeholder="Optional"
              />
            </div>

            <div>
              <Label htmlFor="power">Power (dBm)</Label>
              <Input
                id="power"
                type="number"
                value={formData.powerDbm}
                onChange={(e) => setFormData({ ...formData, powerDbm: Number(e.target.value) })}
              />
            </div>

            <div>
              <Label htmlFor="snr">SNR (dB)</Label>
              <Input
                id="snr"
                type="number"
                value={formData.snrDb || ''}
                onChange={(e) =>
                  setFormData({ ...formData, snrDb: e.target.value ? Number(e.target.value) : undefined })
                }
                placeholder="Optional"
              />
            </div>
          </div>

          <div>
            <Label htmlFor="modulation">Modulation Type</Label>
            <Input
              id="modulation"
              value={formData.modulationType}
              onChange={(e) => setFormData({ ...formData, modulationType: e.target.value })}
              placeholder="e.g., GSM, LTE, WiFi"
            />
          </div>

          <div>
            <Label>Spectrum Snapshot</Label>
            <div className="flex gap-2 items-center">
              <Button
                type="button"
                onClick={handleCaptureSnapshot}
                variant="outline"
                className="flex-1"
              >
                <Camera className="w-4 h-4 mr-2" />
                {hasSnapshot ? 'Recapture Spectrum' : 'Capture Spectrum'}
              </Button>
              {hasSnapshot && (
                <Button
                  type="button"
                  onClick={handleRemoveSnapshot}
                  variant="ghost"
                  size="icon"
                >
                  <X className="w-4 h-4" />
                </Button>
              )}
            </div>
            {hasSnapshot && formData.spectrumSnapshot && (
              <div className="mt-2 border border-zinc-800 rounded overflow-hidden">
                <img
                  src={formData.spectrumSnapshot}
                  alt="Spectrum snapshot preview"
                  className="w-full h-auto"
                />
              </div>
            )}
          </div>

          <div>
            <Label htmlFor="notes">Notes</Label>
            <Textarea
              id="notes"
              value={formData.notes}
              onChange={(e) => setFormData({ ...formData, notes: e.target.value })}
              placeholder="Add any observations or context..."
              rows={3}
            />
          </div>

          <div>
            <Label htmlFor="tags">Tags</Label>
            <div className="flex gap-2 mb-2">
              <Input
                id="tags"
                value={tagInput}
                onChange={(e) => setTagInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleAddTag();
                  }
                }}
                placeholder="Add a tag and press Enter"
              />
              <Button type="button" onClick={handleAddTag} variant="outline">
                Add
              </Button>
            </div>
            {formData.tags.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {formData.tags.map((tag) => (
                  <Badge key={tag} variant="secondary" className="gap-1">
                    {tag}
                    <button
                      onClick={() => handleRemoveTag(tag)}
                      className="ml-1 hover:text-red-500"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  </Badge>
                ))}
              </div>
            )}
          </div>

          <div className="flex gap-2 pt-4">
            <Button onClick={handleSave} className="flex-1">
              Save Bookmark
            </Button>
            <Button onClick={() => onOpenChange(false)} variant="outline">
              Cancel
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
