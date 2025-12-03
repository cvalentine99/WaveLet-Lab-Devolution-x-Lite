import { useState } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Card } from '@/components/ui/card';
import { Badge } from '@/components/ui/badge';
import { useThresholdStore, type ThresholdAlert } from '@/stores/thresholdStore';
import { Plus, Trash2, Bell, Volume2, BellRing, Download, FileDown, History } from 'lucide-react';
import { toast } from 'sonner';
import { exportThresholdAlertsToCSV, exportThresholdAlertsToJSON } from '@/lib/export';
import { ViolationHistoryDialog } from './ViolationHistoryDialog';
import { useViolationLogStore } from '@/stores/violationLogStore';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';

interface ThresholdAlertsDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
}

export function ThresholdAlertsDialog({ open, onOpenChange }: ThresholdAlertsDialogProps) {
  const { thresholds, addThreshold, updateThreshold, deleteThreshold } = useThresholdStore();
  const { violations } = useViolationLogStore();
  const [showAddForm, setShowAddForm] = useState(false);
  const [showViolationHistory, setShowViolationHistory] = useState(false);
  const [selectedThresholdId, setSelectedThresholdId] = useState<string | undefined>();
  const [formData, setFormData] = useState({
    name: '',
    freqStartHz: 900000000,
    freqEndHz: 1000000000,
    thresholdDbm: -60,
    alertType: 'both' as 'toast' | 'audio' | 'both',
    enabled: true,
  });

  const handleAdd = () => {
    if (!formData.name.trim()) {
      toast.error('Please enter a name for the threshold alert');
      return;
    }

    if (formData.freqStartHz >= formData.freqEndHz) {
      toast.error('Start frequency must be less than end frequency');
      return;
    }

    addThreshold(formData);
    toast.success(`Threshold alert "${formData.name}" created`);
    setFormData({
      name: '',
      freqStartHz: 900000000,
      freqEndHz: 1000000000,
      thresholdDbm: -60,
      alertType: 'both',
      enabled: true,
    });
    setShowAddForm(false);
  };

  const handleDelete = (id: string, name: string) => {
    deleteThreshold(id);
    toast.success(`Threshold alert "${name}" deleted`);
  };

  const handleToggle = (id: string, enabled: boolean) => {
    updateThreshold(id, { enabled });
    toast.success(enabled ? 'Alert enabled' : 'Alert disabled');
  };

  const thresholdList = Array.from(thresholds.values()).sort(
    (a, b) => b.createdAt.getTime() - a.createdAt.getTime()
  );

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-3xl max-h-[80vh] overflow-y-auto">
        <DialogHeader>
          <div className="flex items-start justify-between">
            <div className="flex-1">
              <DialogTitle>Power Threshold Alerts</DialogTitle>
              <DialogDescription>
                Configure alerts for when signal power exceeds thresholds in specific frequency ranges
              </DialogDescription>
            </div>
            <div className="flex gap-2">
              {violations.size > 0 && (
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => {
                    setSelectedThresholdId(undefined);
                    setShowViolationHistory(true);
                  }}
                >
                  <History className="w-4 h-4 mr-2" />
                  View History ({violations.size})
                </Button>
              )}
              {thresholdList.length > 0 && (
                <DropdownMenu>
                  <DropdownMenuTrigger asChild>
                    <Button variant="outline" size="sm">
                      <Download className="w-4 h-4 mr-2" />
                      Export
                    </Button>
                  </DropdownMenuTrigger>
                  <DropdownMenuContent>
                    <DropdownMenuItem onClick={() => exportThresholdAlertsToCSV(thresholdList)}>
                      <FileDown className="w-4 h-4 mr-2" />
                      Export as CSV
                    </DropdownMenuItem>
                    <DropdownMenuItem onClick={() => exportThresholdAlertsToJSON(thresholdList)}>
                      <FileDown className="w-4 h-4 mr-2" />
                      Export as JSON
                    </DropdownMenuItem>
                  </DropdownMenuContent>
                </DropdownMenu>
              )}
            </div>
          </div>
        </DialogHeader>

        <div className="space-y-4">
          {/* Add New Threshold Button */}
          {!showAddForm && (
            <Button onClick={() => setShowAddForm(true)} className="w-full">
              <Plus className="w-4 h-4 mr-2" />
              Add Threshold Alert
            </Button>
          )}

          {/* Add Form */}
          {showAddForm && (
            <Card className="p-4 space-y-4 border-primary/50">
              <div className="grid grid-cols-2 gap-4">
                <div className="col-span-2">
                  <Label htmlFor="name">Alert Name</Label>
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                    placeholder="e.g., GSM 900 High Power"
                  />
                </div>

                <div>
                  <Label htmlFor="freqStart">Start Frequency (Hz)</Label>
                  <Input
                    id="freqStart"
                    type="number"
                    value={formData.freqStartHz}
                    onChange={(e) => setFormData({ ...formData, freqStartHz: Number(e.target.value) })}
                  />
                  <p className="text-xs text-zinc-500 mt-1">
                    {(formData.freqStartHz / 1e6).toFixed(2)} MHz
                  </p>
                </div>

                <div>
                  <Label htmlFor="freqEnd">End Frequency (Hz)</Label>
                  <Input
                    id="freqEnd"
                    type="number"
                    value={formData.freqEndHz}
                    onChange={(e) => setFormData({ ...formData, freqEndHz: Number(e.target.value) })}
                  />
                  <p className="text-xs text-zinc-500 mt-1">
                    {(formData.freqEndHz / 1e6).toFixed(2)} MHz
                  </p>
                </div>

                <div>
                  <Label htmlFor="threshold">Threshold (dBm)</Label>
                  <Input
                    id="threshold"
                    type="number"
                    value={formData.thresholdDbm}
                    onChange={(e) => setFormData({ ...formData, thresholdDbm: Number(e.target.value) })}
                  />
                </div>

                <div>
                  <Label htmlFor="alertType">Alert Type</Label>
                  <Select
                    value={formData.alertType}
                    onValueChange={(value: 'toast' | 'audio' | 'both') =>
                      setFormData({ ...formData, alertType: value })
                    }
                  >
                    <SelectTrigger id="alertType">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="toast">Toast Only</SelectItem>
                      <SelectItem value="audio">Audio Only</SelectItem>
                      <SelectItem value="both">Toast + Audio</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </div>

              <div className="flex gap-2">
                <Button onClick={handleAdd} className="flex-1">
                  Create Alert
                </Button>
                <Button onClick={() => setShowAddForm(false)} variant="outline">
                  Cancel
                </Button>
              </div>
            </Card>
          )}

          {/* Threshold List */}
          <div className="space-y-2">
            {thresholdList.length === 0 ? (
              <Card className="p-8 text-center text-zinc-500">
                <Bell className="w-8 h-8 mx-auto mb-2 opacity-50" />
                <p>No threshold alerts configured</p>
                <p className="text-sm">Create an alert to monitor signal power levels</p>
              </Card>
            ) : (
              thresholdList.map((threshold) => (
                <Card key={threshold.id} className="p-4">
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-2 mb-2">
                        <h4 className="font-semibold">{threshold.name}</h4>
                        <Badge variant={threshold.enabled ? 'default' : 'secondary'}>
                          {threshold.enabled ? 'Enabled' : 'Disabled'}
                        </Badge>
                        {threshold.alertType === 'toast' && <Bell className="w-4 h-4 text-zinc-500" />}
                        {threshold.alertType === 'audio' && <Volume2 className="w-4 h-4 text-zinc-500" />}
                        {threshold.alertType === 'both' && <BellRing className="w-4 h-4 text-zinc-500" />}
                      </div>

                      <div className="text-sm text-zinc-400 space-y-1">
                        <div>
                          Frequency: {(threshold.freqStartHz / 1e6).toFixed(2)} -{' '}
                          {(threshold.freqEndHz / 1e6).toFixed(2)} MHz
                        </div>
                        <div>Threshold: {threshold.thresholdDbm} dBm</div>
                        {threshold.triggerCount > 0 && (
                          <div className="text-amber-500">
                            Triggered {threshold.triggerCount} times
                            {threshold.lastTriggeredAt &&
                              ` (last: ${new Date(threshold.lastTriggeredAt).toLocaleString()})`}
                          </div>
                        )}
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <Switch
                        checked={threshold.enabled}
                        onCheckedChange={(checked) => handleToggle(threshold.id, checked)}
                      />
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => handleDelete(threshold.id, threshold.name)}
                      >
                        <Trash2 className="w-4 h-4 text-red-500" />
                      </Button>
                    </div>
                  </div>
                </Card>
              ))
            )}
          </div>
        </div>
      </DialogContent>

      <ViolationHistoryDialog
        open={showViolationHistory}
        onOpenChange={setShowViolationHistory}
        thresholdId={selectedThresholdId}
      />
    </Dialog>
  );
}
