import { useState, useCallback } from "react";
import { useAuth } from "@/hooks/useAuth";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import { Progress } from "@/components/ui/progress";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ArrowLeft, Upload, FileAudio, CheckCircle2, AlertCircle, Loader2 } from "lucide-react";
import { Link } from "wouter";
import { toast } from "sonner";

import { AdaptiveSpectrumDisplay } from "@/components/spectrum/AdaptiveSpectrumDisplay";
import { DetectionList } from "@/components/detections/DetectionList";
import { api, AnalysisFormat } from "@/lib/api";
import { useDetectionStore } from "@/stores/detectionStore";
import { useSpectrumStore } from "@/stores/spectrumStore";

type AnalysisState = "idle" | "uploading" | "processing" | "complete" | "error";

const SUPPORTED_EXT_FORMATS: Record<string, AnalysisFormat> = {
  "sigmf": "sigmf",
  "sigmf-data": "sigmf",
  "cf32": "complex64",
  "cfile": "complex64",
  "bin": "complex64",
  "cs16": "int16_iq",
  "iq": "int16_iq",
  "cu8": "uint8",
};

export default function Analyze() {
  const { user } = useAuth();
  const addDetection = useDetectionStore((s) => s.addDetection);
  const clearDetections = useDetectionStore((s) => s.clearDetections);
  const updatePsd = useSpectrumStore((s) => s.updatePsd);
  const setFrequencyRange = useSpectrumStore((s) => s.setFrequencyRange);

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [fileFormat, setFileFormat] = useState<AnalysisFormat>("complex64");
  const [analysisState, setAnalysisState] = useState<AnalysisState>("idle");
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const [sampleRateHz, setSampleRateHz] = useState<string>("20000000");
  const [centerFreqHz, setCenterFreqHz] = useState<string>("915000000");
  const [fftSize, setFftSize] = useState<string>("2048");

  const handleDrag = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  }, []);

  const handleFileSelect = (file: File) => {
    const ext = (file.name.split(".").pop() || "").toLowerCase();
    const fmt = SUPPORTED_EXT_FORMATS[ext];
    if (fmt) {
      setFileFormat(fmt);
    }
    setSelectedFile(file);
    setAnalysisState("idle");
    setUploadProgress(0);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelect(e.target.files[0]);
    }
  };

  const handleUploadAndAnalyze = async () => {
    if (!selectedFile) {
      toast.error("Please select a file first");
      return;
    }

    // Require sample_rate_hz unless SigMF
    const sr = parseFloat(sampleRateHz);
    const cf = parseFloat(centerFreqHz);
    const fft = parseInt(fftSize, 10) || 2048;
    if (fileFormat !== "sigmf" && (!Number.isFinite(sr) || sr <= 0)) {
      toast.error("Sample rate is required for non-SigMF files");
      return;
    }

    setAnalysisState("uploading");
    setUploadProgress(0);

    try {
      const uploadResp = await api.uploadAnalysisFile(selectedFile, fileFormat);
      setAnalysisId(uploadResp.analysis_id);
      setUploadProgress(100);
      setAnalysisState("processing");

      const startResp = await api.startAnalysis({
        analysis_id: uploadResp.analysis_id,
        format: fileFormat,
        sample_rate_hz: Number.isFinite(sr) ? sr : undefined,
        center_freq_hz: Number.isFinite(cf) ? cf : undefined,
        fft_size: fft,
      });

      // Push results into stores
      clearDetections();
      startResp.detections.forEach((det) => {
        addDetection({
          id: `det-${det.detection_id}`,
          centerFreqHz: det.center_freq_hz,
          bandwidthHz: det.bandwidth_hz,
          bandwidth3dbHz: det.bandwidth_3db_hz,
          bandwidth6dbHz: det.bandwidth_6db_hz,
          peakPowerDb: det.peak_power_db,
          snrDb: det.snr_db,
          startBin: det.start_bin,
          endBin: det.end_bin,
          timestamp: det.timestamp ? det.timestamp * 1000 : Date.now(),
        });
      });

      if (startResp.spectrum?.magnitude_db) {
        const psdArray = new Float32Array(startResp.spectrum.magnitude_db);
        const spanHz = startResp.sample_rate_hz;
        setFrequencyRange(startResp.center_freq_hz, spanHz, startResp.sample_rate_hz);
        updatePsd(psdArray, Date.now());
      }

      setAnalysisState("complete");
      toast.success("Analysis complete");
    } catch (error: any) {
      console.error(error);
      setAnalysisState("error");
      toast.error(error?.message || "Analysis failed");
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setAnalysisState("idle");
    setAnalysisId(null);
    setUploadProgress(0);
    clearDetections();
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(2)} KB`;
    if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
    return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-zinc-50">
      <header className="border-b border-zinc-800 bg-zinc-900/50 backdrop-blur-sm">
        <div className="container mx-auto px-4 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link href="/">
              <Button variant="ghost" size="icon">
                <ArrowLeft className="h-5 w-5" />
              </Button>
            </Link>
            <div>
              <h1 className="text-2xl font-bold">Signal Analysis</h1>
              <p className="text-sm text-zinc-400">Upload and analyze RF signal files</p>
            </div>
          </div>
          {user && (
            <div className="flex items-center gap-2">
              <span className="text-sm text-zinc-400">{user.name}</span>
            </div>
          )}
        </div>
      </header>

      <main className="container mx-auto px-4 py-8">
        {analysisState === "complete" && analysisId ? (
          <div className="space-y-6">
            <Card className="p-6 bg-green-950/20 border-green-800">
              <div className="flex items-center gap-3">
                <CheckCircle2 className="h-6 w-6 text-green-500" />
                <div>
                  <h3 className="font-semibold text-green-400">Analysis Complete</h3>
                  <p className="text-sm text-zinc-400">Results are displayed below</p>
                </div>
                <Button onClick={handleReset} variant="outline" className="ml-auto">
                  Analyze Another File
                </Button>
              </div>
            </Card>

            <Tabs defaultValue="spectrum" className="w-full">
              <TabsList className="grid w-full grid-cols-2">
                <TabsTrigger value="spectrum">Spectrum</TabsTrigger>
                <TabsTrigger value="detections">Detections</TabsTrigger>
              </TabsList>

              <TabsContent value="spectrum" className="mt-4">
                <AdaptiveSpectrumDisplay width={1200} height={600} />
              </TabsContent>

              <TabsContent value="detections" className="mt-4">
                <DetectionList />
              </TabsContent>
            </Tabs>
          </div>
        ) : (
          <div className="max-w-3xl mx-auto space-y-6">
            <Card className="p-8">
              <h2 className="text-xl font-semibold mb-6">Upload IQ File</h2>

              {!selectedFile ? (
                <div
                  className={`border-2 border-dashed rounded-lg p-12 text-center transition-colors ${
                    dragActive
                      ? "border-primary bg-primary/10"
                      : "border-zinc-700 hover:border-zinc-600"
                  }`}
                  onDragEnter={handleDrag}
                  onDragLeave={handleDrag}
                  onDragOver={handleDrag}
                  onDrop={handleDrop}
                >
                  <Upload className="h-12 w-12 mx-auto mb-4 text-zinc-500" />
                  <p className="text-lg mb-2">Drag and drop your IQ file here</p>
                  <p className="text-sm text-zinc-400 mb-4">or</p>
                  <Button variant="outline" onClick={() => document.getElementById("file-input")?.click()}>
                    Browse Files
                  </Button>
                  <input
                    id="file-input"
                    type="file"
                    className="hidden"
                    accept=".sigmf-data,.sigmf,.cf32,.cfile,.cs16,.iq,.cu8,.bin"
                    onChange={handleFileInputChange}
                  />
                  <p className="text-xs text-zinc-500 mt-4">
                    Supported formats: SigMF, complex64, int16_iq, uint8, .bin
                  </p>
                </div>
              ) : (
                <div className="space-y-4">
                  <div className="flex items-start gap-4 p-4 bg-zinc-900 rounded-lg">
                    <FileAudio className="h-10 w-10 text-primary flex-shrink-0" />
                    <div className="flex-1 min-w-0">
                      <p className="font-medium truncate">{selectedFile.name}</p>
                      <p className="text-sm text-zinc-400">{formatFileSize(selectedFile.size)}</p>
                    </div>
                    {analysisState === "idle" && (
                      <Button variant="ghost" size="sm" onClick={handleReset}>
                        Remove
                      </Button>
                    )}
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="format">File Format</Label>
                    <Select value={fileFormat} onValueChange={(v) => setFileFormat(v as AnalysisFormat)}>
                      <SelectTrigger id="format">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="complex64">Complex64 (float32 I/Q)</SelectItem>
                        <SelectItem value="int16_iq">Int16 I/Q</SelectItem>
                        <SelectItem value="uint8">Uint8 I/Q</SelectItem>
                        <SelectItem value="sigmf">SigMF (with metadata)</SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-zinc-500">Format auto-detected from extension, adjust if needed</p>
                  </div>

                  {fileFormat !== "sigmf" && (
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <Label htmlFor="sampleRate">Sample Rate (Hz)</Label>
                        <Input
                          id="sampleRate"
                          type="number"
                          value={sampleRateHz}
                          onChange={(e) => setSampleRateHz(e.target.value)}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="centerFreq">Center Frequency (Hz)</Label>
                        <Input
                          id="centerFreq"
                          type="number"
                          value={centerFreqHz}
                          onChange={(e) => setCenterFreqHz(e.target.value)}
                        />
                      </div>
                    </div>
                  )}

                  <div className="space-y-2">
                    <Label htmlFor="fftSize">FFT Size</Label>
                    <Input
                      id="fftSize"
                      type="number"
                      value={fftSize}
                      onChange={(e) => setFftSize(e.target.value)}
                    />
                  </div>

                  {analysisState === "uploading" && (
                    <div className="space-y-2">
                      <div className="flex items-center justify-between text-sm">
                        <span>Uploading...</span>
                        <span>{uploadProgress}%</span>
                      </div>
                      <Progress value={uploadProgress} />
                    </div>
                  )}

                  {analysisState === "processing" && (
                    <div className="flex items-center gap-3 p-4 bg-blue-950/20 border border-blue-800 rounded-lg">
                      <Loader2 className="h-5 w-5 animate-spin text-blue-500" />
                      <div>
                        <p className="font-medium text-blue-400">Processing...</p>
                        <p className="text-sm text-zinc-400">Running GPU analysis pipeline</p>
                      </div>
                    </div>
                  )}

                  {analysisState === "error" && (
                    <div className="flex items-center gap-3 p-4 bg-red-950/20 border border-red-800 rounded-lg">
                      <AlertCircle className="h-5 w-5 text-red-500" />
                      <div>
                        <p className="font-medium text-red-400">Analysis Failed</p>
                        <p className="text-sm text-zinc-400">Please try again or check file format</p>
                      </div>
                    </div>
                  )}

                  {analysisState === "idle" && (
                    <Button onClick={handleUploadAndAnalyze} className="w-full" size="lg">
                      <Upload className="h-4 w-4 mr-2" />
                      Upload and Analyze
                    </Button>
                  )}
                </div>
              )}
            </Card>

            <Card className="p-6 bg-zinc-900/50">
              <h3 className="font-semibold mb-3">Supported File Formats</h3>
              <div className="space-y-2 text-sm text-zinc-400">
                <div className="flex justify-between">
                  <span>• SigMF (.sigmf-data + .sigmf-meta)</span>
                  <span className="text-zinc-500">Recommended</span>
                </div>
                <div className="flex justify-between">
                  <span>• Complex64 (.cf32, .cfile)</span>
                  <span className="text-zinc-500">Float32 I/Q pairs</span>
                </div>
                <div className="flex justify-between">
                  <span>• Int16 I/Q (.cs16, .iq)</span>
                  <span className="text-zinc-500">Signed 16-bit I/Q</span>
                </div>
                <div className="flex justify-between">
                  <span>• Uint8 I/Q (.cu8)</span>
                  <span className="text-zinc-500">Unsigned 8-bit I/Q</span>
                </div>
              </div>
            </Card>
          </div>
        )}
      </main>
    </div>
  );
}
