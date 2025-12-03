/**
 * WebGPU Utilities
 * Initialization, feature detection, and helper functions
 */

export interface WebGPUContext {
  adapter: GPUAdapter;
  device: GPUDevice;
  context: GPUCanvasContext;
  format: GPUTextureFormat;
}

// Cache WebGPU availability check result
let webGPUAvailable: boolean | null = null;
let webGPUCheckPromise: Promise<boolean> | null = null;

/**
 * Check if WebGPU is supported (sync - basic check only)
 */
export function isWebGPUSupported(): boolean {
  // If we've already done the full check, use cached result
  if (webGPUAvailable !== null) {
    return webGPUAvailable;
  }
  // Basic sync check - navigator.gpu exists
  return 'gpu' in navigator;
}

/**
 * Check if WebGPU is fully available (async - actually tries to get adapter)
 * This is the reliable check that should be used before rendering
 */
export async function checkWebGPUAvailable(): Promise<boolean> {
  // Return cached result if available
  if (webGPUAvailable !== null) {
    return webGPUAvailable;
  }

  // Return pending check if in progress
  if (webGPUCheckPromise) {
    return webGPUCheckPromise;
  }

  // Start the check
  webGPUCheckPromise = (async () => {
    if (!('gpu' in navigator)) {
      console.log('[WebGPU] navigator.gpu not available');
      webGPUAvailable = false;
      return false;
    }

    try {
      const adapter = await navigator.gpu.requestAdapter({
        powerPreference: 'high-performance',
      });

      if (!adapter) {
        console.log('[WebGPU] No adapter available. Enable via chrome://flags/#enable-unsafe-webgpu');
        webGPUAvailable = false;
        return false;
      }

      // Verify we can get a device
      const device = await adapter.requestDevice();
      if (!device) {
        console.log('[WebGPU] Failed to get device');
        webGPUAvailable = false;
        return false;
      }

      // Cleanup test device
      device.destroy();

      console.log('[WebGPU] Full support confirmed');
      webGPUAvailable = true;
      return true;
    } catch (error) {
      console.log('[WebGPU] Check failed:', error);
      webGPUAvailable = false;
      return false;
    }
  })();

  return webGPUCheckPromise;
}

/**
 * Initialize WebGPU context
 */
export async function initWebGPU(canvas: HTMLCanvasElement): Promise<WebGPUContext | null> {
  if (!isWebGPUSupported()) {
    console.warn('[WebGPU] Not supported in this browser');
    return null;
  }

  try {
    // Request adapter
    const adapter = await navigator.gpu.requestAdapter({
      powerPreference: 'high-performance',
    });

    if (!adapter) {
      console.warn('[WebGPU] Failed to get adapter. WebGPU may not be enabled. Enable chrome://flags/#enable-unsafe-webgpu');
      return null;
    }

    // Request device
    const device = await adapter.requestDevice({
      requiredFeatures: [],
      requiredLimits: {
        maxTextureDimension1D: 8192,
        maxTextureDimension2D: 8192,
      },
    });

    // Configure canvas context
    const context = canvas.getContext('webgpu');
    if (!context) {
      console.error('[WebGPU] Failed to get canvas context');
      return null;
    }

    const format = navigator.gpu.getPreferredCanvasFormat();
    context.configure({
      device,
      format,
      alphaMode: 'premultiplied',
    });

    console.log('[WebGPU] Initialized successfully');
    console.log('[WebGPU] Adapter:', adapter);
    console.log('[WebGPU] Format:', format);

    return { adapter, device, context, format };
  } catch (error) {
    console.error('[WebGPU] Initialization failed:', error);
    return null;
  }
}

/**
 * Create 1D texture for PSD data
 */
export function createPSDTexture(
  device: GPUDevice,
  size: number
): GPUTexture {
  return device.createTexture({
    size: [size, 1, 1],
    format: 'r32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
}

/**
 * Create 2D texture for waterfall data
 */
export function createWaterfallTexture(
  device: GPUDevice,
  width: number,
  height: number
): GPUTexture {
  return device.createTexture({
    size: [width, height, 1],
    format: 'r32float',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
}

/**
 * Create 1D texture for colormap
 */
export function createColormapTexture(
  device: GPUDevice,
  colormapData: Uint8Array
): GPUTexture {
  const texture = device.createTexture({
    size: [256, 1, 1],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });

  device.queue.writeTexture(
    { texture },
    colormapData.buffer,
    { bytesPerRow: 256 * 4 },
    [256, 1, 1]
  );

  return texture;
}

/**
 * Update PSD texture with new data
 */
export function updatePSDTexture(
  device: GPUDevice,
  texture: GPUTexture,
  data: Float32Array
): void {
  device.queue.writeTexture(
    { texture },
    data.buffer,
    { bytesPerRow: data.length * 4 },
    [data.length, 1, 1]
  );
}

/**
 * Update waterfall texture with new line
 */
export function updateWaterfallTexture(
  device: GPUDevice,
  texture: GPUTexture,
  data: Float32Array,
  lineIndex: number,
  width: number
): void {
  device.queue.writeTexture(
    { texture, origin: [0, lineIndex, 0] },
    data.buffer,
    { bytesPerRow: width * 4 },
    [width, 1, 1]
  );
}

/**
 * Create uniform buffer
 */
export function createUniformBuffer(
  device: GPUDevice,
  data: ArrayBuffer
): GPUBuffer {
  const buffer = device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });

  device.queue.writeBuffer(buffer, 0, data);

  return buffer;
}

/**
 * Update uniform buffer
 */
export function updateUniformBuffer(
  device: GPUDevice,
  buffer: GPUBuffer,
  data: ArrayBuffer
): void {
  device.queue.writeBuffer(buffer, 0, data);
}

/**
 * Create sampler for texture sampling
 */
export function createSampler(device: GPUDevice): GPUSampler {
  return device.createSampler({
    magFilter: 'linear',
    minFilter: 'linear',
    addressModeU: 'clamp-to-edge',
    addressModeV: 'clamp-to-edge',
  });
}

/**
 * Convert dB values to normalized 0-1 range
 */
export function normalizeDbValues(
  data: Float32Array,
  minDb: number,
  maxDb: number
): Float32Array {
  const normalized = new Float32Array(data.length);
  const range = maxDb - minDb;

  for (let i = 0; i < data.length; i++) {
    normalized[i] = Math.max(0, Math.min(1, (data[i] - minDb) / range));
  }

  return normalized;
}

/**
 * Parse uint8 quantized dB values to float
 */
export function parseQuantizedDb(
  data: Uint8Array,
  minDb: number = -120,
  maxDb: number = -20
): Float32Array {
  const float = new Float32Array(data.length);
  const range = maxDb - minDb;

  for (let i = 0; i < data.length; i++) {
    float[i] = minDb + (data[i] / 255) * range;
  }

  return float;
}
