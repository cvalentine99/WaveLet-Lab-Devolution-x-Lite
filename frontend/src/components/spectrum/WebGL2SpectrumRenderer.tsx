import { useEffect, useRef, useState, forwardRef, useImperativeHandle, useCallback } from 'react';
import type { ColorMapType } from '@/utils/colormaps';
import { COLORMAPS } from '@/utils/colormaps';

// Vertex shader for spectrum line
const SPECTRUM_VERTEX_SHADER = `#version 300 es
precision highp float;

in float a_sample;

uniform float u_width;
uniform float u_height;
uniform float u_minDb;
uniform float u_maxDb;
uniform float u_sampleCount;

void main() {
  // Use gl_VertexID for index instead of a separate attribute
  float x = (float(gl_VertexID) / u_sampleCount) * 2.0 - 1.0;
  float normalized = clamp((a_sample - u_minDb) / (u_maxDb - u_minDb), 0.0, 1.0);
  float y = normalized * 2.0 - 1.0;
  gl_Position = vec4(x, y, 0.0, 1.0);
}
`;

// Fragment shader for spectrum line
const SPECTRUM_FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform vec3 u_color;
out vec4 fragColor;

void main() {
  fragColor = vec4(u_color, 1.0);
}
`;

// Vertex shader for filled area
const FILL_VERTEX_SHADER = `#version 300 es
precision highp float;

in vec2 a_position;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

// Fragment shader for filled gradient
const FILL_FRAGMENT_SHADER = `#version 300 es
precision highp float;

uniform vec3 u_color;
uniform float u_alpha;
out vec4 fragColor;

void main() {
  fragColor = vec4(u_color, u_alpha);
}
`;

export interface WebGL2SpectrumRendererProps {
  width: number;
  height: number;
  colormap?: ColorMapType;
  dynamicRangeMin?: number;
  dynamicRangeMax?: number;
  lineColor?: [number, number, number];
  onError?: (error: Error) => void;
}

export interface WebGL2SpectrumRendererRef {
  updatePSD: (psd: Float32Array) => void;
  setColormap: (colormap: ColorMapType) => void;
  setDynamicRange: (min: number, max: number) => void;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('[WebGL2] Shader compile error:', gl.getShaderInfoLog(shader));
    gl.deleteShader(shader);
    return null;
  }

  return shader;
}

function createProgram(gl: WebGL2RenderingContext, vertexShader: WebGLShader, fragmentShader: WebGLShader): WebGLProgram | null {
  const program = gl.createProgram();
  if (!program) return null;

  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('[WebGL2] Program link error:', gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

/**
 * WebGL2 Spectrum Renderer
 * High-performance GPU-accelerated spectrum visualization
 */
export const WebGL2SpectrumRenderer = forwardRef<WebGL2SpectrumRendererRef, WebGL2SpectrumRendererProps>(
  function WebGL2SpectrumRenderer(
    {
      width,
      height,
      colormap = 'viridis',
      dynamicRangeMin = -120,
      dynamicRangeMax = -20,
      lineColor = [0.13, 0.77, 0.37], // green-500
      onError,
    },
    ref
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const glRef = useRef<WebGL2RenderingContext | null>(null);
    const programRef = useRef<WebGLProgram | null>(null);
    const bufferRef = useRef<WebGLBuffer | null>(null);
    const indexBufferRef = useRef<WebGLBuffer | null>(null);
    const vaoRef = useRef<WebGLVertexArrayObject | null>(null);

    const [fps, setFps] = useState(0);
    const [initialized, setInitialized] = useState(false);

    const currentPsdRef = useRef<Float32Array | null>(null);
    const dynamicRangeRef = useRef({ min: dynamicRangeMin, max: dynamicRangeMax });
    const lineColorRef = useRef(lineColor);

    const frameCountRef = useRef(0);
    const lastFpsUpdateRef = useRef(Date.now());
    const rafIdRef = useRef<number | null>(null);
    const needsRenderRef = useRef(false);

    // Initialize WebGL2
    useEffect(() => {
      const canvas = canvasRef.current;
      if (!canvas) return;

      const gl = canvas.getContext('webgl2', {
        alpha: false,
        antialias: true,
        desynchronized: true,
        powerPreference: 'high-performance',
      });

      if (!gl) {
        console.error('[WebGL2] Failed to get context');
        onError?.(new Error('WebGL2 not supported'));
        return;
      }

      glRef.current = gl;

      // Compile shaders
      const vertexShader = compileShader(gl, gl.VERTEX_SHADER, SPECTRUM_VERTEX_SHADER);
      const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, SPECTRUM_FRAGMENT_SHADER);

      if (!vertexShader || !fragmentShader) {
        onError?.(new Error('Failed to compile shaders'));
        return;
      }

      // Create program
      const program = createProgram(gl, vertexShader, fragmentShader);
      if (!program) {
        onError?.(new Error('Failed to create program'));
        return;
      }

      programRef.current = program;

      // Create VAO
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      vaoRef.current = vao;

      // Create sample data buffer
      const sampleBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, sampleBuffer);
      bufferRef.current = sampleBuffer;

      // Setup attribute for samples
      const sampleLoc = gl.getAttribLocation(program, 'a_sample');
      gl.enableVertexAttribArray(sampleLoc);
      gl.vertexAttribPointer(sampleLoc, 1, gl.FLOAT, false, 0, 0);

      // Create and setup index buffer for vertex indices
      const indexBuffer = gl.createBuffer();
      indexBufferRef.current = indexBuffer;

      gl.bindVertexArray(null);

      // Set clear color
      gl.clearColor(0.04, 0.04, 0.04, 1.0);

      // Enable line smoothing
      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      setInitialized(true);
      console.log('[WebGL2] Spectrum renderer initialized');

      return () => {
        if (rafIdRef.current) {
          cancelAnimationFrame(rafIdRef.current);
        }
        gl.deleteProgram(program);
        gl.deleteBuffer(sampleBuffer);
        gl.deleteBuffer(indexBuffer);
        gl.deleteVertexArray(vao);
      };
    }, [onError]);

    // Render loop
    const renderLoop = useCallback(() => {
      if (!needsRenderRef.current || !initialized) {
        rafIdRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      const gl = glRef.current;
      const program = programRef.current;
      const psd = currentPsdRef.current;

      if (!gl || !program || !psd || psd.length === 0) {
        rafIdRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      needsRenderRef.current = false;

      // Update viewport
      gl.viewport(0, 0, width, height);
      gl.clear(gl.COLOR_BUFFER_BIT);

      // Use program
      gl.useProgram(program);

      // Set uniforms
      const { min, max } = dynamicRangeRef.current;
      gl.uniform1f(gl.getUniformLocation(program, 'u_width'), width);
      gl.uniform1f(gl.getUniformLocation(program, 'u_height'), height);
      gl.uniform1f(gl.getUniformLocation(program, 'u_minDb'), min);
      gl.uniform1f(gl.getUniformLocation(program, 'u_maxDb'), max);
      gl.uniform1f(gl.getUniformLocation(program, 'u_sampleCount'), psd.length);
      gl.uniform3fv(gl.getUniformLocation(program, 'u_color'), lineColorRef.current);

      // Bind VAO
      gl.bindVertexArray(vaoRef.current);

      // Upload PSD data
      gl.bindBuffer(gl.ARRAY_BUFFER, bufferRef.current);
      gl.bufferData(gl.ARRAY_BUFFER, psd, gl.DYNAMIC_DRAW);

      // Create index array if needed (for a_index attribute)
      // We'll use gl_VertexID in shader instead for simplicity

      // Draw spectrum line
      gl.lineWidth(2); // Note: lineWidth > 1 may not work on all platforms
      gl.drawArrays(gl.LINE_STRIP, 0, psd.length);

      gl.bindVertexArray(null);

      // Update FPS
      frameCountRef.current++;
      const now = Date.now();
      if (now - lastFpsUpdateRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastFpsUpdateRef.current = now;
      }

      rafIdRef.current = requestAnimationFrame(renderLoop);
    }, [width, height, initialized]);

    // Start render loop
    useEffect(() => {
      if (initialized) {
        rafIdRef.current = requestAnimationFrame(renderLoop);
      }
      return () => {
        if (rafIdRef.current) {
          cancelAnimationFrame(rafIdRef.current);
        }
      };
    }, [renderLoop, initialized]);

    // Use ref for initialized state so callbacks don't need to recreate
    const initializedRef = useRef(false);
    initializedRef.current = initialized;

    // API methods - use refs to avoid stale closures in useImperativeHandle
    useImperativeHandle(ref, () => ({
      updatePSD: (psd: Float32Array) => {
        if (!initializedRef.current) {
          return; // Silently skip if not ready
        }
        currentPsdRef.current = psd;
        needsRenderRef.current = true;
      },
      setColormap: (_newColormap: ColorMapType) => {
        // Colormap affects gradient fill - for line we use fixed color
        needsRenderRef.current = true;
      },
      setDynamicRange: (min: number, max: number) => {
        dynamicRangeRef.current = { min, max };
        needsRenderRef.current = true;
      },
    }), []);

    return (
      <div className="relative">
        <canvas
          ref={canvasRef}
          width={width}
          height={height}
          className="border border-border rounded-lg"
        />
        {fps > 0 && (
          <div className="absolute top-2 right-2 bg-background/80 px-2 py-1 rounded text-xs font-mono text-green-400">
            {fps} FPS (WebGL2)
          </div>
        )}
      </div>
    );
  }
);
