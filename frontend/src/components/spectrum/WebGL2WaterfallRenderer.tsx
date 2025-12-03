import { useEffect, useRef, useState, forwardRef, useImperativeHandle, useCallback } from 'react';
import type { ColorMapType } from '@/utils/colormaps';
import { COLORMAPS } from '@/utils/colormaps';

// Vertex shader for textured quad
const WATERFALL_VERTEX_SHADER = `#version 300 es
precision highp float;

in vec2 a_position;
in vec2 a_texCoord;

out vec2 v_texCoord;

uniform float u_scrollOffset;

void main() {
  gl_Position = vec4(a_position, 0.0, 1.0);
  // Apply scroll offset to texture coordinates
  v_texCoord = vec2(a_texCoord.x, mod(a_texCoord.y + u_scrollOffset, 1.0));
}
`;

// Fragment shader with colormap lookup
const WATERFALL_FRAGMENT_SHADER = `#version 300 es
precision highp float;

in vec2 v_texCoord;
out vec4 fragColor;

uniform sampler2D u_dataTexture;
uniform sampler2D u_colormapTexture;
uniform float u_minDb;
uniform float u_maxDb;

void main() {
  // Sample the data texture (single channel float)
  float dbValue = texture(u_dataTexture, v_texCoord).r;

  // Normalize to 0-1 range
  float normalized = clamp((dbValue - u_minDb) / (u_maxDb - u_minDb), 0.0, 1.0);

  // Look up color from colormap
  vec4 color = texture(u_colormapTexture, vec2(normalized, 0.5));

  fragColor = color;
}
`;

export interface WebGL2WaterfallRendererProps {
  width: number;
  height: number;
  historyDepth?: number;
  colormap?: ColorMapType;
  dynamicRangeMin?: number;
  dynamicRangeMax?: number;
  onError?: (error: Error) => void;
}

export interface WebGL2WaterfallRendererRef {
  addLine: (psd: Float32Array) => void;
  setColormap: (colormap: ColorMapType) => void;
  setDynamicRange: (min: number, max: number) => void;
  clear: () => void;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader | null {
  const shader = gl.createShader(type);
  if (!shader) return null;

  gl.shaderSource(shader, source);
  gl.compileShader(shader);

  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    console.error('[WebGL2 Waterfall] Shader compile error:', gl.getShaderInfoLog(shader));
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
    console.error('[WebGL2 Waterfall] Program link error:', gl.getProgramInfoLog(program));
    gl.deleteProgram(program);
    return null;
  }

  return program;
}

// Generate colormap texture data from ColorMap
function generateColormapTexture(colormapType: ColorMapType): Uint8Array {
  const colormap = COLORMAPS[colormapType] || COLORMAPS.viridis;
  const textureData = new Uint8Array(256 * 4);

  // ColorMap.data is a Uint8Array with RGB triplets (256 * 3 values)
  for (let i = 0; i < 256; i++) {
    textureData[i * 4 + 0] = colormap.data[i * 3 + 0]; // R
    textureData[i * 4 + 1] = colormap.data[i * 3 + 1]; // G
    textureData[i * 4 + 2] = colormap.data[i * 3 + 2]; // B
    textureData[i * 4 + 3] = 255; // A
  }

  return textureData;
}

/**
 * WebGL2 Waterfall Renderer
 * High-performance GPU-accelerated spectrogram with scrolling texture
 */
export const WebGL2WaterfallRenderer = forwardRef<WebGL2WaterfallRendererRef, WebGL2WaterfallRendererProps>(
  function WebGL2WaterfallRenderer(
    {
      width,
      height,
      historyDepth = 256,
      colormap = 'viridis',
      dynamicRangeMin = -120,
      dynamicRangeMax = -20,
      onError,
    },
    ref
  ) {
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const glRef = useRef<WebGL2RenderingContext | null>(null);
    const programRef = useRef<WebGLProgram | null>(null);
    const vaoRef = useRef<WebGLVertexArrayObject | null>(null);
    const dataTextureRef = useRef<WebGLTexture | null>(null);
    const colormapTextureRef = useRef<WebGLTexture | null>(null);

    const [fps, setFps] = useState(0);
    const [initialized, setInitialized] = useState(false);

    // Waterfall data buffer (circular buffer in GPU texture)
    const textureWidthRef = useRef(1024); // FFT size
    const currentLineRef = useRef(0);
    const dynamicRangeRef = useRef({ min: dynamicRangeMin, max: dynamicRangeMax });
    const colormapRef = useRef<ColorMapType>(colormap);

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
        antialias: false,
        desynchronized: true,
        powerPreference: 'high-performance',
      });

      if (!gl) {
        console.error('[WebGL2 Waterfall] Failed to get context');
        onError?.(new Error('WebGL2 not supported'));
        return;
      }

      glRef.current = gl;

      // Check for required extensions
      const floatTexExt = gl.getExtension('OES_texture_float_linear');
      if (!floatTexExt) {
        console.warn('[WebGL2 Waterfall] OES_texture_float_linear not available, using fallback');
      }

      // Compile shaders
      const vertexShader = compileShader(gl, gl.VERTEX_SHADER, WATERFALL_VERTEX_SHADER);
      const fragmentShader = compileShader(gl, gl.FRAGMENT_SHADER, WATERFALL_FRAGMENT_SHADER);

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

      // Create VAO with fullscreen quad
      const vao = gl.createVertexArray();
      gl.bindVertexArray(vao);
      vaoRef.current = vao;

      // Fullscreen quad vertices (position + texcoord)
      const quadVertices = new Float32Array([
        // position    // texcoord
        -1, -1,        0, 1,  // bottom-left
         1, -1,        1, 1,  // bottom-right
        -1,  1,        0, 0,  // top-left
         1,  1,        1, 0,  // top-right
      ]);

      const quadBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, quadVertices, gl.STATIC_DRAW);

      const posLoc = gl.getAttribLocation(program, 'a_position');
      const texLoc = gl.getAttribLocation(program, 'a_texCoord');

      gl.enableVertexAttribArray(posLoc);
      gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 16, 0);

      gl.enableVertexAttribArray(texLoc);
      gl.vertexAttribPointer(texLoc, 2, gl.FLOAT, false, 16, 8);

      gl.bindVertexArray(null);

      // Create data texture (R32F format for dB values)
      const dataTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, dataTexture);
      gl.texImage2D(
        gl.TEXTURE_2D, 0, gl.R32F,
        textureWidthRef.current, historyDepth,
        0, gl.RED, gl.FLOAT, null
      );
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
      dataTextureRef.current = dataTexture;

      // Create colormap texture
      const colormapTexture = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, colormapTexture);
      const colormapData = generateColormapTexture(colormap);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 256, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, colormapData);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      colormapTextureRef.current = colormapTexture;

      gl.clearColor(0.04, 0.04, 0.04, 1.0);

      setInitialized(true);
      console.log('[WebGL2 Waterfall] Renderer initialized');

      return () => {
        if (rafIdRef.current) {
          cancelAnimationFrame(rafIdRef.current);
        }
        gl.deleteProgram(program);
        gl.deleteTexture(dataTexture);
        gl.deleteTexture(colormapTexture);
        gl.deleteVertexArray(vao);
        gl.deleteBuffer(quadBuffer);
      };
    }, [historyDepth, onError]);

    // Render loop
    const renderLoop = useCallback(() => {
      if (!needsRenderRef.current || !initialized) {
        rafIdRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      const gl = glRef.current;
      const program = programRef.current;

      if (!gl || !program) {
        rafIdRef.current = requestAnimationFrame(renderLoop);
        return;
      }

      needsRenderRef.current = false;

      gl.viewport(0, 0, width, height);
      gl.clear(gl.COLOR_BUFFER_BIT);

      gl.useProgram(program);

      // Bind textures
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, dataTextureRef.current);
      gl.uniform1i(gl.getUniformLocation(program, 'u_dataTexture'), 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, colormapTextureRef.current);
      gl.uniform1i(gl.getUniformLocation(program, 'u_colormapTexture'), 1);

      // Set uniforms
      const { min, max } = dynamicRangeRef.current;
      gl.uniform1f(gl.getUniformLocation(program, 'u_minDb'), min);
      gl.uniform1f(gl.getUniformLocation(program, 'u_maxDb'), max);
      gl.uniform1f(gl.getUniformLocation(program, 'u_scrollOffset'), currentLineRef.current / historyDepth);

      // Draw quad
      gl.bindVertexArray(vaoRef.current);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
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
    }, [width, height, historyDepth, initialized]);

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

    // Add new PSD line
    // Use refs for state that callbacks need to avoid stale closures
    const initializedRef = useRef(false);
    initializedRef.current = initialized;
    const historyDepthRef = useRef(historyDepth);
    historyDepthRef.current = historyDepth;

    // API methods - use refs to avoid stale closures in useImperativeHandle
    useImperativeHandle(ref, () => ({
      addLine: (psd: Float32Array) => {
        const gl = glRef.current;
        if (!gl || !initializedRef.current) return;

        // Resize texture if FFT size changed
        if (psd.length !== textureWidthRef.current) {
          textureWidthRef.current = psd.length;
          gl.bindTexture(gl.TEXTURE_2D, dataTextureRef.current);
          gl.texImage2D(
            gl.TEXTURE_2D, 0, gl.R32F,
            psd.length, historyDepthRef.current,
            0, gl.RED, gl.FLOAT, null
          );
        }

        // Upload single line to texture
        gl.bindTexture(gl.TEXTURE_2D, dataTextureRef.current);
        gl.texSubImage2D(
          gl.TEXTURE_2D, 0,
          0, currentLineRef.current,
          psd.length, 1,
          gl.RED, gl.FLOAT, psd
        );

        // Advance line counter (circular buffer)
        currentLineRef.current = (currentLineRef.current + 1) % historyDepthRef.current;
        needsRenderRef.current = true;
      },
      setColormap: (newColormap: ColorMapType) => {
        const gl = glRef.current;
        if (!gl) return;

        colormapRef.current = newColormap;
        const colormapData = generateColormapTexture(newColormap);

        gl.bindTexture(gl.TEXTURE_2D, colormapTextureRef.current);
        gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 256, 1, gl.RGBA, gl.UNSIGNED_BYTE, colormapData);

        needsRenderRef.current = true;
      },
      setDynamicRange: (min: number, max: number) => {
        dynamicRangeRef.current = { min, max };
        needsRenderRef.current = true;
      },
      clear: () => {
        const gl = glRef.current;
        if (!gl) return;

        currentLineRef.current = 0;
        gl.bindTexture(gl.TEXTURE_2D, dataTextureRef.current);
        gl.texImage2D(
          gl.TEXTURE_2D, 0, gl.R32F,
          textureWidthRef.current, historyDepthRef.current,
          0, gl.RED, gl.FLOAT, null
        );
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
          <div className="absolute top-2 right-2 bg-background/80 px-2 py-1 rounded text-xs font-mono text-cyan-400">
            {fps} FPS (WebGL2)
          </div>
        )}
      </div>
    );
  }
);
