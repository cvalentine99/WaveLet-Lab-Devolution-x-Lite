/**
 * Research Report Generator Component
 * Professional RF signal forensics report with charts and statistics
 */

import { useState, useRef, useMemo } from 'react';
import { PerformanceProfiler } from '@/components/performance/PerformanceProfiler';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { FileText, Download, BarChart3 } from 'lucide-react';
import { toast } from 'sonner';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
} from 'chart.js';
import { Bar, Line, Pie } from 'react-chartjs-2';
import { useDetectionStore } from '@/stores/detectionStore';
import { useClusterStore } from '@/stores/clusterStore';
import { useConfigStore } from '@/stores/configStore';
import type { Detection, Cluster } from '@/types';
import {
  calculateStatistics,
  generateFrequencyOccupancyData,
  generateSNRDistributionData,
  generateModulationPieData,
  generateDetectionTimelineData,
  formatFrequency,
  type ReportData
} from '@/lib/reportGenerator';
import { exportReportToPDF } from '@/lib/pdfExport';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  LineElement,
  PointElement,
  ArcElement,
  Title,
  Tooltip,
  Legend,
  Filler
);

export interface ReportGeneratorProps {
  open: boolean;
  onClose: () => void;
}

// Stable selector for SDR config (returns object with primitives)
const selectSdrConfig = (state: { sdrConfig: { centerFreqHz: number; sampleRateHz: number } }) =>
  state.sdrConfig;

export function ReportGenerator({ open, onClose }: ReportGeneratorProps) {
  return (
    <PerformanceProfiler id="ReportGenerator" slowThresholdMs={50}>
      <ReportGeneratorInner open={open} onClose={onClose} />
    </PerformanceProfiler>
  );
}

function ReportGeneratorInner({ open, onClose }: ReportGeneratorProps) {
  const [reportTitle, setReportTitle] = useState('Valentine RF - Signal Forensics Analysis Report');
  const [generating, setGenerating] = useState(false);
  
  // Chart refs for PDF export
  const freqOccupancyRef = useRef<ChartJS<'bar'>>(null);
  const snrDistRef = useRef<ChartJS<'bar'>>(null);
  const modulationPieRef = useRef<ChartJS<'pie'>>(null);
  const timelineRef = useRef<ChartJS<'line'>>(null);
  
  // Subscribe to sizes for re-render triggers
  const detectionCount = useDetectionStore((state) => state.detections.size);
  const clusterCount = useClusterStore((state) => state.clusters.size);
  const sdrConfig = useConfigStore(selectSdrConfig);
  
  // Get data via getState() - avoids unstable selector snapshot
  const detections = useMemo(() => {
    return useDetectionStore.getState().getActiveDetections();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [detectionCount]);
  
  const clusters = useMemo(() => {
    return useClusterStore.getState().getSortedClusters('size');
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [clusterCount]);
  
  // Prepare report data with useMemo to prevent infinite loops
  const reportData: ReportData = useMemo(() => ({
    title: reportTitle,
    generatedAt: new Date(),
    analysisStartTime: new Date(Date.now() - 60000), // Last minute
    analysisEndTime: new Date(),
    duration: 60,
    centerFreqHz: sdrConfig.centerFreqHz,
    sampleRateHz: sdrConfig.sampleRateHz,
    spanHz: sdrConfig.sampleRateHz,
    detections,
    clusters,
    loraPackets: 0, // TODO: Get from decoder store
    blePackets: 0,
  }), [reportTitle, sdrConfig.centerFreqHz, sdrConfig.sampleRateHz, detections, clusters]);
  
  const stats = useMemo(() => calculateStatistics(reportData), [reportData]);
  
  // Chart data with useMemo
  const freqOccupancyData = useMemo(() => generateFrequencyOccupancyData(stats), [stats]);
  const snrDistData = useMemo(() => generateSNRDistributionData(detections), [detections]);
  const modulationPieData = useMemo(() => generateModulationPieData(stats), [stats]);
  const timelineData = useMemo(() => generateDetectionTimelineData(detections, reportData.duration), [detections, reportData.duration]);
  
  const handleGeneratePDF = async () => {
    setGenerating(true);
    
    try {
      // Get chart canvases
      const chartElements: Record<string, HTMLCanvasElement> = {};
      
      if (freqOccupancyRef.current) {
        chartElements.frequencyOccupancy = freqOccupancyRef.current.canvas;
      }
      if (snrDistRef.current) {
        chartElements.snrDistribution = snrDistRef.current.canvas;
      }
      if (modulationPieRef.current) {
        chartElements.modulationPie = modulationPieRef.current.canvas;
      }
      if (timelineRef.current) {
        chartElements.detectionTimeline = timelineRef.current.canvas;
      }
      
      await exportReportToPDF(reportData, stats, chartElements);
      
      toast.success('Report generated successfully');
      onClose();
    } catch (error) {
      console.error('Failed to generate report:', error);
      toast.error('Failed to generate report');
    } finally {
      setGenerating(false);
    }
  };
  
  return (
    <Dialog open={open} onOpenChange={onClose}>
      <DialogContent className="max-w-6xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <FileText className="w-5 h-5" />
            Generate Research Report
          </DialogTitle>
          <DialogDescription>
            Create a comprehensive analysis report with statistics and visualizations
          </DialogDescription>
        </DialogHeader>
        
        <div className="space-y-6">
          {/* Report Configuration */}
          <div className="space-y-4">
            <div>
              <Label htmlFor="title">Report Title</Label>
              <Input
                id="title"
                value={reportTitle}
                onChange={(e) => setReportTitle(e.target.value)}
                placeholder="Enter report title"
              />
            </div>
          </div>
          
          {/* Summary Statistics */}
          <Card className="p-4 bg-slate-900/50 border-slate-700">
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
              <BarChart3 className="w-5 h-5" />
              Summary Statistics
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <div className="text-sm text-slate-400">Total Detections</div>
                <div className="text-2xl font-bold text-blue-400">{stats.totalDetections}</div>
              </div>
              <div>
                <div className="text-sm text-slate-400">Unique Emitters</div>
                <div className="text-2xl font-bold text-green-400">{stats.uniqueEmitters}</div>
              </div>
              <div>
                <div className="text-sm text-slate-400">Average SNR</div>
                <div className="text-2xl font-bold text-purple-400">{stats.avgSnr.toFixed(2)} dB</div>
              </div>
              <div>
                <div className="text-sm text-slate-400">Detection Rate</div>
                <div className="text-2xl font-bold text-orange-400">{stats.detectionsPerSecond.toFixed(2)} /sec</div>
              </div>
            </div>
          </Card>
          
          {/* Charts Preview */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Frequency Occupancy */}
            <Card className="p-4 bg-slate-900/50 border-slate-700">
              <h4 className="text-sm font-semibold mb-3">Frequency Occupancy</h4>
              <Bar
                ref={freqOccupancyRef}
                data={freqOccupancyData}
                options={{
                  responsive: true,
                  maintainAspectRatio: true,
                  plugins: {
                    legend: { display: false },
                    title: { display: false }
                  },
                  scales: {
                    y: { beginAtZero: true, ticks: { color: '#94a3b8' } },
                    x: { ticks: { color: '#94a3b8' } }
                  }
                }}
              />
            </Card>
            
            {/* SNR Distribution */}
            <Card className="p-4 bg-slate-900/50 border-slate-700">
              <h4 className="text-sm font-semibold mb-3">SNR Distribution</h4>
              <Bar
                ref={snrDistRef}
                data={snrDistData}
                options={{
                  responsive: true,
                  maintainAspectRatio: true,
                  plugins: {
                    legend: { display: false },
                    title: { display: false }
                  },
                  scales: {
                    y: { beginAtZero: true, ticks: { color: '#94a3b8' } },
                    x: { ticks: { color: '#94a3b8' } }
                  }
                }}
              />
            </Card>
            
            {/* Modulation Breakdown */}
            <Card className="p-4 bg-slate-900/50 border-slate-700">
              <h4 className="text-sm font-semibold mb-3">Modulation Types</h4>
              <Pie
                ref={modulationPieRef}
                data={modulationPieData}
                options={{
                  responsive: true,
                  maintainAspectRatio: true,
                  plugins: {
                    legend: { position: 'bottom', labels: { color: '#94a3b8' } }
                  }
                }}
              />
            </Card>
            
            {/* Detection Timeline */}
            <Card className="p-4 bg-slate-900/50 border-slate-700">
              <h4 className="text-sm font-semibold mb-3">Detection Timeline</h4>
              <Line
                ref={timelineRef}
                data={timelineData}
                options={{
                  responsive: true,
                  maintainAspectRatio: true,
                  plugins: {
                    legend: { display: false },
                    title: { display: false }
                  },
                  scales: {
                    y: { beginAtZero: true, ticks: { color: '#94a3b8' } },
                    x: { ticks: { color: '#94a3b8', maxTicksLimit: 10 } }
                  }
                }}
              />
            </Card>
          </div>
          
          {/* Frequency Band Table */}
          <Card className="p-4 bg-slate-900/50 border-slate-700">
            <h3 className="text-lg font-semibold mb-4">Frequency Band Analysis</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left py-2">Band</th>
                    <th className="text-left py-2">Frequency Range</th>
                    <th className="text-right py-2">Detections</th>
                    <th className="text-right py-2">Utilization</th>
                    <th className="text-right py-2">Avg SNR</th>
                  </tr>
                </thead>
                <tbody>
                  {stats.frequencyBands.map((band, idx) => (
                    <tr key={idx} className="border-b border-slate-800">
                      <td className="py-2">{band.name}</td>
                      <td className="py-2 text-slate-400">
                        {formatFrequency(band.startHz)} - {formatFrequency(band.endHz)}
                      </td>
                      <td className="text-right py-2">{band.detectionCount}</td>
                      <td className="text-right py-2">{band.utilizationPercent.toFixed(1)}%</td>
                      <td className="text-right py-2">{band.avgSnrDb.toFixed(2)} dB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Card>
        </div>
        
        <DialogFooter>
          <Button variant="outline" onClick={onClose}>
            Cancel
          </Button>
          <Button onClick={handleGeneratePDF} disabled={generating}>
            <Download className="w-4 h-4 mr-2" />
            {generating ? 'Generating...' : 'Export PDF'}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
