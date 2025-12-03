import { useEffect, useRef, useState, forwardRef, useImperativeHandle } from 'react';
import { initWebGPU, createPSDTexture, createColormapTexture, createUniformBuffer, createSampler, updatePSDTexture, updateUniformBuffer } from '@/utils/webgpu';
import { createColormapTexture as createColormapData } from '@/utils/colormaps';
import type { ColorMapType } from '@/utils/colormaps';
import spectrumShaderCode from '@/shaders/spectrum.wgsl?raw';

export interface WebGPUSpectrumRendererProps {
  width: number;
  height: number;
  colormap?: ColorMapType;
  dynamicRangeMin?: number;
  dynamicRangeMax?: number;
  onError?: (error: Error) => void;
}

export interface WebGPUSpectrumRendererRef {
  updatePSD: (psd: Float32Array) => void;
  setColormap: (colormap: ColorMapType) => void;
  setDynamicRange: (min: number, max: number) => void;
}

/**
 * WebGPU Spectrum Renderer
 * GPU-accelerated PSD rendering with WGSL shaders
 */
export const WebGPUSpectrumRenderer = forwardRef<WebGPUSpectrumRendererRef, WebGPUSpectrumRendererProps>(
  function WebGPUSpectrumRenderer(
    {
      width,
      height,
      colormap = 'viridis',
      dynamicRangeMin = -120,
      dynamicRangeMax = -20,
      onError,
    },
    ref
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const [initialized, setInitialized] = useState(false);
    const [fps, setFps] = useState(0);
    
    // WebGPU resources
    const gpuContextRef = useRef<{
      device: GPUDevice;
      context: GPUCanvasContext;
      pipeline: GPURenderPipeline;
      bindGroup: GPUBindGroup;
      psdTexture: GPUTexture;
      colormapTexture: GPUTexture;
      uniformBuffer: GPUBuffer;
      sampler: GPUSampler;
    } | null>(null);
    
    const frameCountRef = useRef(0);
    const lastFpsUpdateRef = useRef(Date.now());

    /**
     * Initialize WebGPU
     */
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      let mounted = true;

      (async () => {
        try {
          const gpuContext = await initWebGPU(canvas);
          if (!gpuContext || !mounted) return;

          const { device, context, format } = gpuContext;

          // Create shader module
          const shaderModule = device.createShaderModule({
            code: spectrumShaderCode,
          });

          // Create textures
          const psdTexture = createPSDTexture(device, 1024);
          const colormapData = createColormapData(colormap);
          const colormapTexture = createColormapTexture(device, colormapData);
          const sampler = createSampler(device);

          // Create uniform buffer
          const uniformData = new Float32Array([
            dynamicRangeMin, // minDb
            dynamicRangeMax, // maxDb
            1024, // numBins
            1.0, // lineThickness
          ]);
          const uniformBuffer = createUniformBuffer(device, uniformData.buffer);

          // Create bind group layout
          const bindGroupLayout = device.createBindGroupLayout({
            entries: [
              { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' as GPUBufferBindingType } },
              { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' as GPUTextureSampleType, viewDimension: '1d' as GPUTextureViewDimension } },
              { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' as GPUTextureSampleType, viewDimension: '1d' as GPUTextureViewDimension } },
              { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' as GPUSamplerBindingType } },
            ],
          });

          // Create bind group
          const bindGroup = device.createBindGroup({
            layout: bindGroupLayout,
            entries: [
              { binding: 0, resource: { buffer: uniformBuffer } },
              { binding: 1, resource: psdTexture.createView({ dimension: '1d' }) },
              { binding: 2, resource: colormapTexture.createView({ dimension: '1d' }) },
              { binding: 3, resource: sampler },
            ],
          });

          // Create pipeline
          const pipeline = device.createRenderPipeline({
            layout: device.createPipelineLayout({
              bindGroupLayouts: [bindGroupLayout],
            }),
            vertex: {
              module: shaderModule,
              entryPoint: 'vs_main',
            },
            fragment: {
              module: shaderModule,
              entryPoint: 'fs_main',
              targets: [{ format }],
            },
            primitive: {
              topology: 'triangle-strip',
            },
          });

          // Store resources
          gpuContextRef.current = {
            device,
            context,
            pipeline,
            bindGroup,
            psdTexture,
            colormapTexture,
            uniformBuffer,
            sampler,
          };

          setInitialized(true);
          console.log('[WebGPU Spectrum] Initialized successfully');
        } catch (error) {
          console.error('[WebGPU Spectrum] Initialization failed:', error);
          onError?.(error as Error);
        }
      })();

      return () => {
        mounted = false;
      };
    }, []);

    /**
     * Update PSD data
     */
    const updatePSD = (psd: Float32Array) => {
      const gpuContext = gpuContextRef.current;
      if (!gpuContext) return;

      const { device, context, pipeline, bindGroup, psdTexture, uniformBuffer } = gpuContext;

      // Update PSD texture
      updatePSDTexture(device, psdTexture, psd);

      // Update uniforms
      const uniformData = new Float32Array([
        dynamicRangeMin,
        dynamicRangeMax,
        psd.length,
        1.0,
      ]);
      updateUniformBuffer(device, uniformBuffer, uniformData.buffer);

      // Render
      const commandEncoder = device.createCommandEncoder();
      const textureView = context.getCurrentTexture().createView();

      const renderPass = commandEncoder.beginRenderPass({
        colorAttachments: [
          {
            view: textureView,
            clearValue: { r: 0.05, g: 0.05, b: 0.05, a: 1.0 },
            loadOp: 'clear' as GPULoadOp,
            storeOp: 'store' as GPUStoreOp,
          },
        ],
      });

      renderPass.setPipeline(pipeline);
      renderPass.setBindGroup(0, bindGroup);
      renderPass.draw(4); // Full-screen quad
      renderPass.end();

      device.queue.submit([commandEncoder.finish()]);

      // Update FPS
      frameCountRef.current++;
      const now = Date.now();
      if (now - lastFpsUpdateRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsUpdateRef.current = now;
      }
    };

    /**
     * Set colormap
     */
    const setColormap = (newColormap: ColorMapType) => {
      const gpuContext = gpuContextRef.current;
      if (!gpuContext) return;

      const { device } = gpuContext;
      const colormapData = createColormapData(newColormap);
      const newColormapTexture = createColormapTexture(device, colormapData);

      // Update bind group with new colormap
      const bindGroup = device.createBindGroup({
        layout: gpuContext.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: gpuContext.uniformBuffer } },
          { binding: 1, resource: gpuContext.psdTexture.createView({ dimension: '1d' }) },
          { binding: 2, resource: newColormapTexture.createView({ dimension: '1d' }) },
          { binding: 3, resource: gpuContext.sampler },
        ],
      });

      gpuContext.bindGroup = bindGroup;
      gpuContext.colormapTexture = newColormapTexture;
    };

    /**
     * Set dynamic range
     */
    const setDynamicRange = (min: number, max: number) => {
      // Will be applied on next updatePSD call
    };

    // Expose methods via ref
    useImperativeHandle(ref, () => ({
      updatePSD,
      setColormap,
      setDynamicRange,
    }));

    return (
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="border border-border rounded-lg"
        />
        {initialized && fps > 0 && (
          <div className="absolute top-2 right-2 bg-background/80 px-2 py-1 rounded text-xs font-mono text-muted-foreground">
            {fps} FPS
          </div>
        )}
        {!initialized && (
          <div className="absolute inset-0 flex items-center justify-center bg-background/50">
            <div className="text-sm text-muted-foreground">Initializing WebGPU...</div>
          </div>
        )}
      </div>
    );
  }
);
