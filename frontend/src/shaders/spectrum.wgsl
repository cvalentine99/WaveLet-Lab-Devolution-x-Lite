// Spectrum Visualization Shader (WGSL)
// GPU-accelerated PSD rendering with colormap lookup

struct Uniforms {
  minDb: f32,
  maxDb: f32,
  numBins: u32,
  lineThickness: f32,
}

@group(0) @binding(0) var<uniform> uniforms: Uniforms;
@group(0) @binding(1) var psdTexture: texture_1d<f32>;
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
  
  // Generate full-screen quad
  let x = f32((vertexIndex & 1u) << 1u) - 1.0;
  let y = f32((vertexIndex & 2u)) - 1.0;
  
  output.position = vec4<f32>(x, y, 0.0, 1.0);
  output.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
  
  return output;
}

// Fragment Shader - PSD rendering with colormap
@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
  // Sample PSD value from 1D texture
  let binIndex = u32(input.uv.x * f32(uniforms.numBins));
  let psdValue = textureLoad(psdTexture, binIndex, 0).r;
  
  // Normalize to 0-1 range based on dynamic range
  let normalized = (psdValue - uniforms.minDb) / (uniforms.maxDb - uniforms.minDb);
  let clamped = clamp(normalized, 0.0, 1.0);
  
  // Lookup color from colormap
  let color = textureSample(colormapTexture, texSampler, clamped);
  
  return color;
}
