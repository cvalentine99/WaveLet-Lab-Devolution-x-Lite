import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useViolationLogStore } from '@/stores/violationLogStore';
import { AlertTriangle, Search, Trash2, Download, FileDown } from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';
import { toast } from 'sonner';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface ViolationHistoryDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  thresholdId?: string; // If provided, show only violations for this threshold
}

export function ViolationHistoryDialog({ open, onOpenChange, thresholdId }: ViolationHistoryDialogProps) {
  const { violations, getViolationsByThreshold, getRecentViolations, clearViolations, deleteViolation, searchViolations } =
    useViolationLogStore();
  const [searchQuery, setSearchQuery] = useState('');

  const violationList = searchQuery
    ? searchViolations(searchQuery)
    : thresholdId
    ? getViolationsByThreshold(thresholdId)
    : getRecentViolations(500);

  const handleClearAll = () => {
    if (confirm('Are you sure you want to clear all violation logs? This cannot be undone.')) {
      clearViolations();
      toast.success('All violation logs cleared');
    }
  };

  const exportToCSV = () => {
    const headers = ['Timestamp', 'Threshold Name', 'Frequency (MHz)', 'Power (dBm)', 'Threshold (dBm)', 'Exceedance (dB)'];
    const rows = violationList.map((v) => [
      v.timestamp.toISOString(),
      v.thresholdName,
      (v.centerFreqHz / 1e6).toFixed(6),
      v.powerDbm.toFixed(2),
      v.thresholdDbm.toFixed(2),
      v.exceedanceDbm.toFixed(2),
    ]);

    const csvContent = [headers.join(','), ...rows.map((row) => row.map((cell) => `"${cell}"`).join(','))].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    link.download = `threshold-violations-${timestamp}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast.success('Violation log exported to CSV');
  };

  const exportToJSON = () => {
    const data = {
      exportDate: new Date().toISOString(),
      exportType: 'Threshold Violation Log',
      count: violationList.length,
      violations: violationList.map((v) => ({
        timestamp: v.timestamp.toISOString(),
        thresholdId: v.thresholdId,
        thresholdName: v.thresholdName,
        centerFreqHz: v.centerFreqHz,
        centerFreqMHz: v.centerFreqHz / 1e6,
        powerDbm: v.powerDbm,
        thresholdDbm: v.thresholdDbm,
        exceedanceDbm: v.exceedanceDbm,
        detectionId: v.detectionId,
      })),
    };

    const jsonContent = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonContent], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    link.download = `threshold-violations-${timestamp}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);

    toast.success('Violation log exported to JSON');
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-4xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <DialogTitle className="flex items-center gap-2">
                <AlertTriangle className="w-5 h-5 text-amber-500" />
                Threshold Violation History
              </DialogTitle>
              <DialogDescription>
                {thresholdId
                  ? 'Violations for selected threshold'
                  : `Complete audit trail of all threshold violations (${violationList.length} total)`}
              </DialogDescription>
            </div>
            <div className="flex gap-2">
              {violationList.length > 0 && (
                <>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="outline" size="sm">
                        <Download className="w-4 h-4 mr-2" />
                        Export
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent>
                      <DropdownMenuItem onClick={exportToCSV}>
                        <FileDown className="w-4 h-4 mr-2" />
                        Export as CSV
                      </DropdownMenuItem>
                      <DropdownMenuItem onClick={exportToJSON}>
                        <FileDown className="w-4 h-4 mr-2" />
                        Export as JSON
                      </DropdownMenuItem>
                    </DropdownMenuContent>
                  </DropdownMenu>
                  <Button variant="outline" size="sm" onClick={handleClearAll}>
                    <Trash2 className="w-4 h-4 mr-2" />
                    Clear All
                  </Button>
                </>
              )}
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-4">
          {/* Search */}
          {!thresholdId && (
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-zinc-500" />
              <Input
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search by threshold name or frequency..."
                className="pl-10"
              />
            </div>
          )}

          {/* Violation List */}
          {violationList.length === 0 ? (
            <Card className="p-8 text-center text-zinc-500">
              <AlertTriangle className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p>No threshold violations recorded</p>
              <p className="text-sm">Violations will appear here when signal power exceeds configured thresholds</p>
            </Card>
          ) : (
            <div className="space-y-2">
              {violationList.map((violation) => (
                <Card key={violation.id} className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className="font-semibold">{violation.thresholdName}</h4>
                        <Badge variant="destructive" className="text-xs">
                          +{violation.exceedanceDbm.toFixed(1)} dB
                        </Badge>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
                        <div>
                          <div className="text-zinc-500 text-xs">Frequency</div>
                          <div className="font-mono">{(violation.centerFreqHz / 1e6).toFixed(3)} MHz</div>
                        </div>
                        <div>
                          <div className="text-zinc-500 text-xs">Detected Power</div>
                          <div className="font-mono text-amber-500">{violation.powerDbm.toFixed(1)} dBm</div>
                        </div>
                        <div>
                          <div className="text-zinc-500 text-xs">Threshold</div>
                          <div className="font-mono">{violation.thresholdDbm.toFixed(1)} dBm</div>
                        </div>
                        <div>
                          <div className="text-zinc-500 text-xs">Time</div>
                          <div className="text-zinc-400">
                            {formatDistanceToNow(violation.timestamp, { addSuffix: true })}
                          </div>
                        </div>
                      </div>

                      <div className="mt-2 text-xs text-zinc-500">
                        {violation.timestamp.toLocaleString()}
                      </div>
                    </div>

                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => {
                        deleteViolation(violation.id);
                        toast.success('Violation log deleted');
                      }}
                    >
                      <Trash2 className="w-4 h-4 text-red-500" />
                    </Button>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
}
