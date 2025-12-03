import { useState, useEffect } from 'react';
import { Card } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Activity, Download, RefreshCw, Trash2 } from 'lucide-react';
import {
  getAllPerformanceMetrics,
  clearPerformanceMetrics,
  exportPerformanceReport,
} from './PerformanceProfiler';

/**
 * Performance Dashboard
 * Displays real-time component render metrics
 */
export function PerformanceDashboard() {
  const [metrics, setMetrics] = useState(getAllPerformanceMetrics());
  const [autoRefresh, setAutoRefresh] = useState(true);

  useEffect(() => {
    if (!autoRefresh) return;

    const interval = setInterval(() => {
      setMetrics(getAllPerformanceMetrics());
    }, 1000);

    return () => clearInterval(interval);
  }, [autoRefresh]);

  const handleRefresh = () => {
    setMetrics(getAllPerformanceMetrics());
  };

  const handleClear = () => {
    clearPerformanceMetrics();
    setMetrics([]);
  };

  const handleExport = () => {
    exportPerformanceReport();
  };

  // Sort by average duration (slowest first)
  const sortedMetrics = [...metrics].sort((a, b) => b.avgDuration - a.avgDuration);

  return (
    <Card className="p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-primary" />
          <h3 className="text-sm font-semibold">Performance Monitor</h3>
          <Badge variant="secondary" className="text-xs">
            {metrics.length} components
          </Badge>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setAutoRefresh(!autoRefresh)}
            className="h-8 px-2"
          >
            <RefreshCw className={`w-3 h-3 ${autoRefresh ? 'animate-spin' : ''}`} />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleRefresh}
            className="h-8 px-2"
          >
            <RefreshCw className="w-3 h-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleExport}
            className="h-8 px-2"
          >
            <Download className="w-3 h-3" />
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleClear}
            className="h-8 px-2"
          >
            <Trash2 className="w-3 h-3" />
          </Button>
        </div>
      </div>

      <ScrollArea className="h-[400px]">
        <div className="space-y-2">
          {sortedMetrics.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground text-sm">
              No performance data yet
            </div>
          ) : (
            sortedMetrics.map((metric) => (
              <div
                key={metric.id}
                className="p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
              >
                <div className="flex items-start justify-between mb-2">
                  <div className="flex-1">
                    <div className="font-medium text-sm">{metric.id}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      {metric.renderCount} renders
                    </div>
                  </div>
                  <Badge
                    variant={metric.avgDuration > 16 ? 'destructive' : 'secondary'}
                    className="text-xs"
                  >
                    {metric.avgDuration.toFixed(2)}ms avg
                  </Badge>
                </div>

                <div className="grid grid-cols-3 gap-2 text-xs">
                  <div>
                    <div className="text-muted-foreground">Min</div>
                    <div className="font-medium">{metric.minDuration.toFixed(2)}ms</div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Max</div>
                    <div className="font-medium text-destructive">
                      {metric.maxDuration.toFixed(2)}ms
                    </div>
                  </div>
                  <div>
                    <div className="text-muted-foreground">Last</div>
                    <div className="font-medium">{metric.lastRender.toFixed(2)}ms</div>
                  </div>
                </div>

                {/* Performance bar */}
                <div className="mt-2 h-1 bg-muted rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all ${
                      metric.avgDuration > 16 ? 'bg-destructive' : 'bg-primary'
                    }`}
                    style={{
                      width: `${Math.min((metric.avgDuration / 50) * 100, 100)}%`,
                    }}
                  />
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>

      <div className="mt-4 pt-4 border-t text-xs text-muted-foreground">
        <div className="flex items-center justify-between">
          <span>Target: &lt;16ms per render (60 FPS)</span>
          <span className="text-primary">
            Auto-refresh: {autoRefresh ? 'ON' : 'OFF'}
          </span>
        </div>
      </div>
    </Card>
  );
}
