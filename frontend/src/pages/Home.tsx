import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, Upload, History, Cpu, Radio, CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { Link } from "wouter";
import { useState, useEffect } from "react";
import { api, type SystemStatus, type SDRDevice } from "@/lib/api";

export default function Home() {
  const [status, setStatus] = useState<SystemStatus | null>(null);
  const [devices, setDevices] = useState<SDRDevice[]>([]);
  const [loading, setLoading] = useState(true);
  const [backendOnline, setBackendOnline] = useState(false);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const [statusData, deviceData] = await Promise.all([
          api.getStatus(),
          api.getSDRDevices().catch(() => [])
        ]);
        setStatus(statusData);
        setDevices(deviceData);
        setBackendOnline(true);
      } catch (error) {
        setBackendOnline(false);
      } finally {
        setLoading(false);
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, []);
  return (
    <div className="min-h-screen bg-background">
      {/* Header */}
      <header className="border-b border-border bg-background/95 backdrop-blur">
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center gap-2">
            <img src="/wsdr-logo.svg" alt="wSDR" className="h-8" />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="container mx-auto px-4 py-12">
        <div className="max-w-4xl mx-auto space-y-8">
          {/* Hero Section */}
          <div className="text-center space-y-4">
            <h2 className="text-4xl font-bold tracking-tight">GPU-Accelerated RF Analysis</h2>
            <p className="text-xl text-muted-foreground">
              Real-time spectrum monitoring with CUDA-powered signal processing
            </p>
          </div>

          {/* Feature Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-12">
            <Link href="/analyze">
              <Card className="cursor-pointer hover:border-primary transition-colors h-full">
                <CardHeader>
                  <div className="w-12 h-12 rounded-lg bg-primary/10 flex items-center justify-center mb-2">
                    <Upload className="w-6 h-6 text-primary" />
                  </div>
                  <CardTitle>Analyze Signal</CardTitle>
                  <CardDescription>
                    Upload and analyze RF signal files with GPU acceleration
                  </CardDescription>
                </CardHeader>
              </Card>
            </Link>

            <Link href="/live">
              <Card className="cursor-pointer hover:border-primary transition-colors h-full">
                <CardHeader>
                  <div className="w-12 h-12 rounded-lg bg-blue-500/10 flex items-center justify-center mb-2">
                    <Activity className="w-6 h-6 text-blue-500" />
                  </div>
                  <CardTitle>Live Monitoring</CardTitle>
                  <CardDescription>
                    Real-time spectrum analysis with WebGPU visualization
                  </CardDescription>
                </CardHeader>
              </Card>
            </Link>

            <Link href="/history">
              <Card className="cursor-pointer hover:border-primary transition-colors h-full">
                <CardHeader>
                  <div className="w-12 h-12 rounded-lg bg-cyan-500/10 flex items-center justify-center mb-2">
                    <History className="w-6 h-6 text-cyan-500" />
                  </div>
                  <CardTitle>Analysis History</CardTitle>
                  <CardDescription>
                    View and manage your previous signal analysis results
                  </CardDescription>
                </CardHeader>
              </Card>
            </Link>
          </div>

          {/* System Status */}
          <Card className="mt-8">
            <CardHeader>
              <div className="flex items-center gap-2">
                <Cpu className="w-5 h-5 text-primary" />
                <CardTitle>System Status</CardTitle>
                {loading && <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />}
              </div>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {/* Backend Status */}
                <div className="flex justify-between text-sm items-center">
                  <span className="text-muted-foreground">Backend (Port 8000):</span>
                  {backendOnline ? (
                    <span className="font-mono text-green-500 flex items-center gap-1">
                      <CheckCircle2 className="w-4 h-4" /> Online
                    </span>
                  ) : (
                    <span className="font-mono text-red-500 flex items-center gap-1">
                      <XCircle className="w-4 h-4" /> Offline
                    </span>
                  )}
                </div>

                {/* Pipeline State */}
                <div className="flex justify-between text-sm">
                  <span className="text-muted-foreground">Pipeline State:</span>
                  <span className={`font-mono ${status?.state === 'running' ? 'text-green-500' : status?.state === 'error' ? 'text-red-500' : 'text-yellow-500'}`}>
                    {status?.state || 'unknown'}
                  </span>
                </div>

                {/* GPU Memory */}
                {status && (
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">GPU Memory:</span>
                    <span className="font-mono text-blue-400">
                      {status.gpu_memory_used_gb.toFixed(2)} GB
                    </span>
                  </div>
                )}

                {/* Throughput */}
                {status && status.state === 'running' && (
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Throughput:</span>
                    <span className="font-mono text-cyan-400">
                      {status.current_throughput_msps.toFixed(1)} Msps
                    </span>
                  </div>
                )}

                {/* SDR Devices */}
                <div className="flex justify-between text-sm items-center">
                  <span className="text-muted-foreground flex items-center gap-1">
                    <Radio className="w-3 h-3" /> SDR Devices:
                  </span>
                  {devices.length > 0 ? (
                    <span className="font-mono text-green-400">
                      {devices.length} found ({devices[0]?.model || 'SDR'})
                    </span>
                  ) : (
                    <span className="font-mono text-muted-foreground">
                      None detected
                    </span>
                  )}
                </div>

                {/* Detections Count */}
                {status && (
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground">Total Detections:</span>
                    <span className="font-mono text-purple-400">
                      {status.detections_count.toLocaleString()}
                    </span>
                  </div>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Supported Formats */}
          <div className="mt-12">
            <h3 className="text-lg font-semibold mb-4">Supported Signal Formats</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {[
                { name: 'complex64', desc: 'Float32 I+Q', bytes: '8 bytes/sample' },
                { name: 'int16_iq', desc: 'Int16 I+Q', bytes: '4 bytes/sample' },
                { name: 'uint8_iq', desc: 'Uint8 I+Q (RTL-SDR)', bytes: '2 bytes/sample' },
                { name: 'float32', desc: 'Raw float32', bytes: '4 bytes/sample' },
                { name: 'float32_db', desc: 'FFT magnitude (dB)', bytes: '4 bytes/sample' },
              ].map((format) => (
                <Card key={format.name} className="p-3">
                  <div className="text-sm font-mono text-primary mb-1">{format.name}</div>
                  <div className="text-xs text-muted-foreground">{format.desc}</div>
                  <div className="text-xs text-muted-foreground mt-1">{format.bytes}</div>
                </Card>
              ))}
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
