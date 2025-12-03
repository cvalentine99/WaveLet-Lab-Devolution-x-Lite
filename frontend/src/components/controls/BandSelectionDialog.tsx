import { useState, useEffect } from 'react';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Badge } from '@/components/ui/badge';
import { api, type Band } from '@/lib/api';
import { toast } from 'sonner';
import { Radio, ArrowLeftRight, ArrowDown, ArrowUp, Star, Search } from 'lucide-react';
import { getFavoriteBands, toggleFavoriteBand } from '@/lib/bandFavorites';

export interface BandSelectionDialogProps {
  open: boolean;
  onClose: () => void;
  onBandSelected?: (band: Band) => void;
}

/**
 * Duplexer Band Selection Dialog
 * Complete list of all uSDR DevBoard duplexer bands with search and favorites
 */
export function BandSelectionDialog({ open, onClose, onBandSelected }: BandSelectionDialogProps) {
  const [bands, setBands] = useState<Band[]>([]);
  const [selectedBand, setSelectedBand] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [favoriteBands, setFavoriteBands] = useState<Set<string>>(new Set());
  const [searchQuery, setSearchQuery] = useState('');

  // Load bands and favorites on open
  useEffect(() => {
    if (open) {
      loadBands();
      setFavoriteBands(getFavoriteBands());
    }
  }, [open]);

  const loadBands = async () => {
    try {
      setLoading(true);
      const bandList = await api.getSDRBands();
      setBands(bandList);
    } catch (error) {
      console.error('Failed to load bands:', error);
      toast.error('Failed to load SDR bands');
    } finally {
      setLoading(false);
    }
  };

  const handleApply = async () => {
    if (!selectedBand) {
      toast.error('Please select a band');
      return;
    }

    try {
      setLoading(true);
      await api.setSDRBand(selectedBand);
      
      const band = bands.find(b => b.name === selectedBand);
      if (band && onBandSelected) {
        onBandSelected(band);
      }
      
      toast.success(`Band set to ${selectedBand}`);
      onClose();
    } catch (error) {
      console.error('Failed to set band:', error);
      toast.error('Failed to set band');
    } finally {
      setLoading(false);
    }
  };

  const handleToggleFavorite = (bandName: string, event: React.MouseEvent) => {
    event.stopPropagation();
    toggleFavoriteBand(bandName);
    setFavoriteBands(getFavoriteBands());
  };

  // Filter bands by search query
  const filterBands = (bandList: Band[]) => {
    if (!searchQuery) return bandList;
    const query = searchQuery.toLowerCase();
    return bandList.filter(b => 
      b.name.toLowerCase().includes(query) ||
      b.description?.toLowerCase().includes(query)
    );
  };

  // Categorize bands (backend uses 'cellular' for FDD duplexer bands)
  const favoriteBandsList = filterBands(bands.filter(b => favoriteBands.has(b.name)));
  const cellularBands = filterBands(bands.filter(b => b.category === 'cellular'));
  const txOnlyBands = filterBands(bands.filter(b => b.category === 'tx_only'));
  const rxOnlyBands = filterBands(bands.filter(b => b.category === 'rx_only'));
  const tddBands = filterBands(bands.filter(b => b.category === 'tdd'));

  const BandCard = ({ band }: { band: Band }) => {
    const isFavorite = favoriteBands.has(band.name);
    const isSelected = selectedBand === band.name;

    return (
      <div
        onClick={() => setSelectedBand(band.name)}
        className={`
          relative p-3 rounded-lg border-2 cursor-pointer transition-all
          ${isSelected ? 'border-primary bg-primary/5' : 'border-border hover:border-primary/50'}
        `}
      >
        <div className="flex items-start justify-between gap-2">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="font-mono font-semibold">{band.name}</span>
              {band.category === 'cellular' && <Badge variant="default" className="text-xs">FDD</Badge>}
              {band.category === 'tx_only' && <Badge variant="secondary" className="text-xs">TX</Badge>}
              {band.category === 'rx_only' && <Badge variant="outline" className="text-xs">RX</Badge>}
              {band.category === 'tdd' && <Badge className="text-xs bg-purple-500">TDD</Badge>}
            </div>
            {band.description && (
              <p className="text-xs text-muted-foreground">{band.description}</p>
            )}
            {/* Frequency range */}
            {band.freq_range_mhz && (
              <p className="text-xs text-muted-foreground mt-1 font-mono">
                {band.freq_range_mhz[0]} - {band.freq_range_mhz[1]} MHz
              </p>
            )}
            {/* PA/LNA indicators */}
            <div className="flex items-center gap-2 mt-1">
              {band.pa_enable !== undefined && (
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${band.pa_enable ? 'bg-green-500/20 text-green-400' : 'bg-muted text-muted-foreground'}`}>
                  PA {band.pa_enable ? 'ON' : 'OFF'}
                </span>
              )}
              {band.lna_enable !== undefined && (
                <span className={`text-[10px] px-1.5 py-0.5 rounded ${band.lna_enable ? 'bg-blue-500/20 text-blue-400' : 'bg-muted text-muted-foreground'}`}>
                  LNA {band.lna_enable ? 'ON' : 'OFF'}
                </span>
              )}
            </div>
          </div>
          <Button
            variant="ghost"
            size="sm"
            className="h-8 w-8 p-0 shrink-0"
            onClick={(e) => handleToggleFavorite(band.name, e)}
          >
            <Star className={`h-4 w-4 ${isFavorite ? 'fill-amber-500 text-amber-500' : ''}`} />
          </Button>
        </div>
      </div>
    );
  };

  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[85vh] overflow-hidden flex flex-col">
        <DialogHeader>
          <DialogTitle>Select Duplexer Band</DialogTitle>
          <DialogDescription>
            Choose a duplexer configuration for the uSDR DevBoard
          </DialogDescription>
        </DialogHeader>

        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search bands..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-9"
          />
        </div>

        {/* Tabs */}
        <Tabs defaultValue="all" className="flex-1 overflow-hidden flex flex-col">
          <TabsList className="grid w-full grid-cols-6">
            <TabsTrigger value="all">All ({bands.length})</TabsTrigger>
            <TabsTrigger value="favorites">
              <Star className="h-3 w-3 mr-1" />
              ({favoriteBandsList.length})
            </TabsTrigger>
            <TabsTrigger value="cellular">FDD ({cellularBands.length})</TabsTrigger>
            <TabsTrigger value="tx">TX ({txOnlyBands.length})</TabsTrigger>
            <TabsTrigger value="rx">RX ({rxOnlyBands.length})</TabsTrigger>
            <TabsTrigger value="tdd">TDD ({tddBands.length})</TabsTrigger>
          </TabsList>

          <div className="flex-1 overflow-y-auto mt-4">
            <TabsContent value="all" className="mt-0">
              <div className="grid grid-cols-2 gap-3">
                {filterBands(bands).map(band => (
                  <BandCard key={band.name} band={band} />
                ))}
              </div>
            </TabsContent>

            <TabsContent value="favorites" className="mt-0">
              {favoriteBandsList.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <Star className="h-12 w-12 mx-auto mb-2 opacity-20" />
                  <p>No favorite bands yet</p>
                  <p className="text-sm">Click the star icon to add favorites</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 gap-3">
                  {favoriteBandsList.map(band => (
                    <BandCard key={band.name} band={band} />
                  ))}
                </div>
              )}
            </TabsContent>

            <TabsContent value="cellular" className="mt-0">
              <div className="grid grid-cols-2 gap-3">
                {cellularBands.map(band => (
                  <BandCard key={band.name} band={band} />
                ))}
              </div>
            </TabsContent>

            <TabsContent value="tx" className="mt-0">
              <div className="grid grid-cols-2 gap-3">
                {txOnlyBands.map(band => (
                  <BandCard key={band.name} band={band} />
                ))}
              </div>
            </TabsContent>

            <TabsContent value="rx" className="mt-0">
              <div className="grid grid-cols-2 gap-3">
                {rxOnlyBands.map(band => (
                  <BandCard key={band.name} band={band} />
                ))}
              </div>
            </TabsContent>

            <TabsContent value="tdd" className="mt-0">
              <div className="grid grid-cols-2 gap-3">
                {tddBands.map(band => (
                  <BandCard key={band.name} band={band} />
                ))}
              </div>
            </TabsContent>
          </div>
        </Tabs>

        {/* Footer */}
        <div className="flex items-center justify-between pt-4 border-t">
          <div className="text-sm text-muted-foreground">
            {selectedBand ? (
              <span>Selected: <span className="font-mono font-semibold">{selectedBand}</span></span>
            ) : (
              <span>No band selected</span>
            )}
          </div>
          <div className="flex gap-2">
            <Button variant="outline" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={handleApply} disabled={!selectedBand || loading}>
              {loading ? 'Applying...' : 'Apply Band'}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  );
}
