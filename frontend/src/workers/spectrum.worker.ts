/**
 * Spectrum Processing Web Worker
 * Handles heavy computation off the main thread:
 * - Peak detection
 * - Averaging
 * - Decimation
 * - dB conversion
 */

// Message types
export type WorkerMessage =
  | { type: 'process'; data: ArrayBuffer; fftSize: number; format: string }
  | { type: 'setAveraging'; count: number }
  | { type: 'setPeakHold'; enabled: boolean }
  | { type: 'clearPeakHold' }
  | { type: 'setDecimation'; factor: number };

export type WorkerResponse =
  | { type: 'psd'; data: Float32Array; peakHold?: Float32Array; average?: Float32Array }
  | { type: 'error'; message: string }
  | { type: 'ready' };

// Processing state
let averagingCount = 1;
let averageBuffer: Float32Array | null = null;
let averageSampleCount = 0;

let peakHoldEnabled = false;
let peakHoldBuffer: Float32Array | null = null;

let decimationFactor = 1;

/**
 * Convert raw IQ samples to power spectrum (simplified)
 * In production, you'd want to do proper FFT here
 */
function processRawIQ(buffer: ArrayBuffer, format: string): Float32Array {
  let samples: Float32Array;

  switch (format) {
    case 'complex64':
      samples = new Float32Array(buffer);
      break;
    case 'int16_iq': {
      const int16 = new Int16Array(buffer);
      samples = new Float32Array(int16.length);
      for (let i = 0; i < int16.length; i++) {
        samples[i] = int16[i] / 32768;
      }
      break;
    }
    case 'uint8': {
      const uint8 = new Uint8Array(buffer);
      samples = new Float32Array(uint8.length);
      for (let i = 0; i < uint8.length; i++) {
        samples[i] = (uint8[i] - 127.5) / 127.5;
      }
      break;
    }
    case 'float32_db':
      // Already in dB format
      return new Float32Array(buffer);
    default:
      samples = new Float32Array(buffer);
  }

  // Simple magnitude calculation (for I/Q pairs)
  const psdLength = Math.floor(samples.length / 2);
  const psd = new Float32Array(psdLength);

  for (let i = 0; i < psdLength; i++) {
    const I = samples[i * 2];
    const Q = samples[i * 2 + 1];
    const magnitude = Math.sqrt(I * I + Q * Q);
    // Convert to dB with floor
    psd[i] = magnitude > 0 ? 20 * Math.log10(magnitude) : -120;
  }

  return psd;
}

/**
 * Decimate PSD data
 */
function decimate(psd: Float32Array, factor: number): Float32Array {
  if (factor <= 1) return psd;

  const outputLength = Math.ceil(psd.length / factor);
  const output = new Float32Array(outputLength);

  for (let i = 0; i < outputLength; i++) {
    let max = -Infinity;
    const start = i * factor;
    const end = Math.min(start + factor, psd.length);

    // Peak hold within decimation bin
    for (let j = start; j < end; j++) {
      if (psd[j] > max) max = psd[j];
    }
    output[i] = max;
  }

  return output;
}

/**
 * Update running average
 */
function updateAverage(psd: Float32Array): Float32Array {
  if (averagingCount <= 1) {
    return psd;
  }

  if (!averageBuffer || averageBuffer.length !== psd.length) {
    averageBuffer = new Float32Array(psd.length);
    averageSampleCount = 0;
  }

  averageSampleCount++;

  if (averageSampleCount === 1) {
    averageBuffer.set(psd);
  } else {
    const alpha = 1 / Math.min(averageSampleCount, averagingCount);
    for (let i = 0; i < psd.length; i++) {
      averageBuffer[i] = averageBuffer[i] * (1 - alpha) + psd[i] * alpha;
    }
  }

  return averageBuffer;
}

/**
 * Update peak hold
 */
function updatePeakHold(psd: Float32Array): Float32Array | undefined {
  if (!peakHoldEnabled) return undefined;

  if (!peakHoldBuffer || peakHoldBuffer.length !== psd.length) {
    peakHoldBuffer = new Float32Array(psd.length);
    peakHoldBuffer.fill(-Infinity);
  }

  for (let i = 0; i < psd.length; i++) {
    if (psd[i] > peakHoldBuffer[i]) {
      peakHoldBuffer[i] = psd[i];
    }
  }

  return peakHoldBuffer;
}

// Handle messages from main thread
self.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const msg = event.data;

  switch (msg.type) {
    case 'process': {
      try {
        // Process raw data to PSD
        let psd = processRawIQ(msg.data, msg.format);

        // Apply decimation
        if (decimationFactor > 1) {
          psd = decimate(psd, decimationFactor);
        }

        // Update average
        const average = averagingCount > 1 ? updateAverage(psd) : undefined;

        // Update peak hold
        const peakHold = updatePeakHold(psd);

        // Send response with transferable arrays
        const response: WorkerResponse = {
          type: 'psd',
          data: psd,
          peakHold: peakHold ? new Float32Array(peakHold) : undefined,
          average: average ? new Float32Array(average) : undefined,
        };

        // Transfer the buffer for zero-copy
        const transfers: Transferable[] = [psd.buffer as ArrayBuffer];
        if (response.peakHold) transfers.push(response.peakHold.buffer as ArrayBuffer);
        if (response.average) transfers.push(response.average.buffer as ArrayBuffer);

        self.postMessage(response, { transfer: transfers });
      } catch (error) {
        self.postMessage({
          type: 'error',
          message: error instanceof Error ? error.message : 'Processing error',
        } as WorkerResponse);
      }
      break;
    }

    case 'setAveraging':
      averagingCount = Math.max(1, msg.count);
      averageSampleCount = 0;
      break;

    case 'setPeakHold':
      peakHoldEnabled = msg.enabled;
      if (!msg.enabled) {
        peakHoldBuffer = null;
      }
      break;

    case 'clearPeakHold':
      if (peakHoldBuffer) {
        peakHoldBuffer.fill(-Infinity);
      }
      break;

    case 'setDecimation':
      decimationFactor = Math.max(1, msg.factor);
      break;
  }
};

// Signal ready
self.postMessage({ type: 'ready' } as WorkerResponse);
