// Waterfall (Spectrogram) Shader (WGSL)
// GPU-accelerated scrolling waterfall with ring buffer

struct Uniforms {
  minDb: f32,
  maxDb: f32,
  numBins: u32,
  historyDepth: u32,
  writeIndex: u32,
  scrollOffset: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var waterfallTexture: texture_2d<f32>;
@group(0) @binding(2) var colormapTexture: texture_1d<f32>;
@group(0) @binding(3) var texSampler: sampler;

struct VertexOutput {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
}

// Vertex Shader - Full-screen quad
@vertex
fn vs_main(@builtin(vertex_index) vertexIndex: u32) -> VertexOutput {
  var output: VertexOutput;
  
  let x = f32((vertexIndex & 1u) << 1u) - 1.0;
  let y = f32((vertexIndex & 2u)) - 1.0;
  
  output.position = vec4<f32>(x, y, 0.0, 1.0);
  output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  
  return output;
}

// Fragment Shader - Waterfall rendering with ring buffer wrap-around
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  // Calculate texture coordinates with ring buffer wrap-around
  let binIndex = u32(input.uv.x * f32(uniforms.numBins));
  
  // Apply scroll offset and wrap around
  var timeIndex = u32((input.uv.y + uniforms.scrollOffset) * f32(uniforms.historyDepth));
  timeIndex = (timeIndex + uniforms.writeIndex) % uniforms.historyDepth;
  
  // Sample PSD value from 2D waterfall texture
  let psdValue = textureLoad(waterfallTexture, vec2<u32>(binIndex, timeIndex), 0).r;
  
  // Normalize to 0-1 range
  let normalized = (psdValue - uniforms.minDb) / (uniforms.maxDb - uniforms.minDb);
  let clamped = clamp(normalized, 0.0, 1.0);
  
  // Lookup color from colormap
  let color = textureSample(colormapTexture, texSampler, clamped);
  
  return color;
}
