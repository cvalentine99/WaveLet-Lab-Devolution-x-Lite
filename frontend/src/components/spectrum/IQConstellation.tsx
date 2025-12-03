import { useRef, useEffect, forwardRef, useImperativeHandle, useState } from 'react';

export interface IQConstellationProps {
  width: number;
  height: number;
  maxPoints?: number;
  decayRate?: number;
  gridLines?: boolean;
  showStats?: boolean;
}

export interface IQConstellationRef {
  addSamples: (i: Float32Array, q: Float32Array) => void;
  clear: () => void;
  setDecayRate: (rate: number) => void;
}

interface IQPoint {
  i: number;
  q: number;
  age: number;
}

/**
 * IQ Constellation Display
 *
 * Scatter plot visualization of In-phase (I) and Quadrature (Q) samples.
 * Essential for modulation analysis and signal quality assessment.
 *
 * Features:
 * - Real-time IQ sample plotting
 * - Point decay for temporal visualization
 * - Grid overlay for reference
 * - Statistics (EVM, power, etc.)
 */
export const IQConstellation = forwardRef<IQConstellationRef, IQConstellationProps>(
  function IQConstellation(
    {
      width,
      height,
      maxPoints = 2048,
      decayRate = 0.02,
      gridLines = true,
      showStats = true,
    },
    ref
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const pointsRef = useRef<IQPoint[]>([]);
    const decayRateRef = useRef(decayRate);
    const animationFrameRef = useRef<number>(0);
    const [fps, setFps] = useState(0);
    const [stats, setStats] = useState({ power: 0, evm: 0, snr: 0 });

    const frameCountRef = useRef(0);
    const lastFpsUpdateRef = useRef(Date.now());

    /**
     * Add IQ samples to constellation
     */
    const addSamples = (iData: Float32Array, qData: Float32Array) => {
      const len = Math.min(iData.length, qData.length);
      const points = pointsRef.current;

      // Age existing points
      for (const point of points) {
        point.age += decayRateRef.current;
      }

      // Remove old points
      pointsRef.current = points.filter(p => p.age < 1);

      // Add new points (subsample if too many)
      const step = len > 256 ? Math.floor(len / 256) : 1;
      for (let i = 0; i < len; i += step) {
        pointsRef.current.push({
          i: iData[i],
          q: qData[i],
          age: 0,
        });
      }

      // Limit total points
      if (pointsRef.current.length > maxPoints) {
        pointsRef.current = pointsRef.current.slice(-maxPoints);
      }

      // Calculate stats
      calculateStats(iData, qData);

      render();
    };

    /**
     * Calculate signal statistics
     */
    const calculateStats = (iData: Float32Array, qData: Float32Array) => {
      if (iData.length === 0) return;

      let sumPower = 0;
      let sumI = 0;
      let sumQ = 0;

      for (let i = 0; i < iData.length; i++) {
        sumPower += iData[i] * iData[i] + qData[i] * qData[i];
        sumI += iData[i];
        sumQ += qData[i];
      }

      const avgPower = sumPower / iData.length;
      const avgI = sumI / iData.length;
      const avgQ = sumQ / iData.length;

      // EVM calculation (simplified - assumes QAM reference points)
      let evmSum = 0;
      for (let i = 0; i < iData.length; i++) {
        const errorI = iData[i] - avgI;
        const errorQ = qData[i] - avgQ;
        evmSum += errorI * errorI + errorQ * errorQ;
      }
      const evm = avgPower > 0 ? Math.sqrt(evmSum / iData.length / avgPower) * 100 : 0;

      // SNR estimate (simplified)
      const snr = avgPower > 0 ? 10 * Math.log10(avgPower / (evmSum / iData.length + 1e-10)) : 0;

      setStats({
        power: 10 * Math.log10(avgPower + 1e-10),
        evm: evm,
        snr: Math.max(0, Math.min(60, snr)),
      });
    };

    /**
     * Clear constellation
     */
    const clear = () => {
      pointsRef.current = [];
      render();
    };

    /**
     * Set decay rate
     */
    const setDecayRate = (rate: number) => {
      decayRateRef.current = Math.max(0.001, Math.min(0.5, rate));
    };

    /**
     * Render constellation
     */
    const render = () => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      const points = pointsRef.current;
      const cx = width / 2;
      const cy = height / 2;
      const scale = Math.min(width, height) / 3; // Scale factor for IQ values

      // Clear with dark background
      ctx.fillStyle = '#0a0e14';
      ctx.fillRect(0, 0, width, height);

      // Draw grid
      if (gridLines) {
        ctx.strokeStyle = '#2a3040';
        ctx.lineWidth = 1;

        // Axes
        ctx.beginPath();
        ctx.moveTo(cx, 0);
        ctx.lineTo(cx, height);
        ctx.moveTo(0, cy);
        ctx.lineTo(width, cy);
        ctx.stroke();

        // Grid circles
        ctx.strokeStyle = '#1a1f2e';
        for (let r = 0.25; r <= 1; r += 0.25) {
          ctx.beginPath();
          ctx.arc(cx, cy, r * scale, 0, Math.PI * 2);
          ctx.stroke();
        }

        // Grid lines
        for (let i = -1; i <= 1; i += 0.5) {
          if (i === 0) continue;
          ctx.beginPath();
          ctx.moveTo(cx + i * scale, 0);
          ctx.lineTo(cx + i * scale, height);
          ctx.moveTo(0, cy + i * scale);
          ctx.lineTo(width, cy + i * scale);
          ctx.stroke();
        }

        // Labels
        ctx.fillStyle = '#666';
        ctx.font = '11px "IBM Plex Mono", monospace';
        ctx.textAlign = 'center';
        ctx.fillText('I', width - 15, cy - 5);
        ctx.fillText('Q', cx + 10, 15);
      }

      // Draw points
      if (points.length === 0) {
        ctx.fillStyle = '#666';
        ctx.font = '14px "Space Grotesk", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText('Waiting for IQ samples...', cx, cy);
        return;
      }

      // Sort by age (oldest first) so newer points are on top
      const sortedPoints = [...points].sort((a, b) => b.age - a.age);

      for (const point of sortedPoints) {
        const x = cx + point.i * scale;
        const y = cy - point.q * scale; // Y is inverted in canvas

        // Color based on age (wSDR blue to transparent)
        const alpha = Math.max(0.1, 1 - point.age);
        const intensity = Math.floor(alpha * 255);

        // wSDR blue with age-based alpha
        ctx.fillStyle = `rgba(74, 144, 217, ${alpha})`;

        // Point size based on age (newer = larger)
        const size = 2 + (1 - point.age) * 2;

        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }

      // Draw unit circle reference
      ctx.strokeStyle = 'rgba(74, 144, 217, 0.3)';
      ctx.lineWidth = 1;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.arc(cx, cy, scale, 0, Math.PI * 2);
      ctx.stroke();
      ctx.setLineDash([]);

      // Update FPS
      frameCountRef.current++;
      const now = Date.now();
      if (now - lastFpsUpdateRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsUpdateRef.current = now;
      }
    };

    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      addSamples,
      clear,
      setDecayRate,
    }));

    // Initial render
    useEffect(() => {
      render();
    }, [width, height, gridLines]);

    return (
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="border border-border rounded-lg"
        />
        <div className="absolute top-2 left-2 bg-background/80 px-2 py-1 rounded text-xs font-mono text-muted-foreground">
          IQ Constellation
        </div>

        {/* Stats overlay */}
        {showStats && pointsRef.current.length > 0 && (
          <div className="absolute bottom-2 left-2 bg-background/80 px-2 py-1 rounded text-xs font-mono space-y-0.5">
            <div className="text-muted-foreground">
              Power: <span className="text-primary">{stats.power.toFixed(1)} dB</span>
            </div>
            <div className="text-muted-foreground">
              EVM: <span className="text-primary">{stats.evm.toFixed(1)}%</span>
            </div>
            <div className="text-muted-foreground">
              SNR: <span className="text-primary">{stats.snr.toFixed(1)} dB</span>
            </div>
          </div>
        )}

        {fps > 0 && (
          <div className="absolute top-2 right-2 bg-background/80 px-2 py-1 rounded text-xs font-mono text-muted-foreground">
            {fps} FPS
          </div>
        )}
      </div>
    );
  }
);
