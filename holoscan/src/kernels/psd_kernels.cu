/**
 * PSD CUDA Kernels for RF Forensics Holoscan Pipeline
 *
 * Implements Welch's method for Power Spectral Density estimation:
 * 1. Segment signal into overlapping windows
 * 2. Apply Hann window
 * 3. FFT each segment
 * 4. Compute magnitude squared
 * 5. Average across segments
 */

#include "psd_kernels.cuh"
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>

namespace rf_forensics {
namespace kernels {

// =============================================================================
// Hann Window Generation
// =============================================================================
__global__ void generate_hann_window_kernel(float* window, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float n = static_cast<float>(idx);
        float N = static_cast<float>(size);
        window[idx] = 0.5f * (1.0f - cosf(2.0f * M_PI * n / N));
    }
}

void generate_hann_window(float* window, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    generate_hann_window_kernel<<<blocks, threads, 0, stream>>>(window, size);
}

// =============================================================================
// Signal Segmentation + Windowing (Fused Kernel)
// =============================================================================
__global__ void segment_and_window_kernel(
    const cuComplex* __restrict__ signal,
    cuComplex* __restrict__ segments,
    const float* __restrict__ window,
    int signal_len,
    int fft_size,
    int hop_size,
    int num_segments
) {
    // Grid: (fft_size, num_segments)
    int freq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seg_idx = blockIdx.y;

    if (freq_idx >= fft_size || seg_idx >= num_segments) return;

    int signal_idx = seg_idx * hop_size + freq_idx;
    int out_idx = seg_idx * fft_size + freq_idx;

    cuComplex sample;
    if (signal_idx < signal_len) {
        sample = signal[signal_idx];
    } else {
        sample = make_cuComplex(0.0f, 0.0f);
    }

    // Apply window
    float w = window[freq_idx];
    segments[out_idx] = make_cuComplex(sample.x * w, sample.y * w);
}

void segment_and_window(
    const cuComplex* signal,
    cuComplex* segments,
    const float* window,
    int signal_len,
    int fft_size,
    int hop_size,
    int num_segments,
    cudaStream_t stream
) {
    dim3 threads(256);
    dim3 blocks((fft_size + 255) / 256, num_segments);
    segment_and_window_kernel<<<blocks, threads, 0, stream>>>(
        signal, segments, window, signal_len, fft_size, hop_size, num_segments
    );
}

// =============================================================================
// Magnitude Squared (Complex to Power)
// =============================================================================
__global__ void magnitude_squared_kernel(
    const cuComplex* __restrict__ fft_out,
    float* __restrict__ power,
    int fft_size,
    int num_segments,
    float scale  // 1.0 / (fft_size * window_power_sum)
) {
    int freq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int seg_idx = blockIdx.y;

    if (freq_idx >= fft_size || seg_idx >= num_segments) return;

    int idx = seg_idx * fft_size + freq_idx;
    cuComplex val = fft_out[idx];
    power[idx] = (val.x * val.x + val.y * val.y) * scale;
}

void magnitude_squared(
    const cuComplex* fft_out,
    float* power,
    int fft_size,
    int num_segments,
    float scale,
    cudaStream_t stream
) {
    dim3 threads(256);
    dim3 blocks((fft_size + 255) / 256, num_segments);
    magnitude_squared_kernel<<<blocks, threads, 0, stream>>>(
        fft_out, power, fft_size, num_segments, scale
    );
}

// =============================================================================
// Average PSD Across Segments
// =============================================================================
__global__ void average_psd_kernel(
    const float* __restrict__ segment_power,
    float* __restrict__ psd_out,
    int fft_size,
    int num_segments
) {
    int freq_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (freq_idx >= fft_size) return;

    float sum = 0.0f;
    for (int seg = 0; seg < num_segments; seg++) {
        sum += segment_power[seg * fft_size + freq_idx];
    }
    psd_out[freq_idx] = sum / static_cast<float>(num_segments);
}

void average_psd(
    const float* segment_power,
    float* psd_out,
    int fft_size,
    int num_segments,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (fft_size + threads - 1) / threads;
    average_psd_kernel<<<blocks, threads, 0, stream>>>(
        segment_power, psd_out, fft_size, num_segments
    );
}

// =============================================================================
// Convert to dB
// =============================================================================
__global__ void power_to_db_kernel(
    const float* __restrict__ power,
    float* __restrict__ db_out,
    int size,
    float ref,
    float min_db
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float val = power[idx];
    float db = 10.0f * log10f(fmaxf(val / ref, 1e-20f));
    db_out[idx] = fmaxf(db, min_db);
}

void power_to_db(
    const float* power,
    float* db_out,
    int size,
    float ref,
    float min_db,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (size + threads - 1) / threads;
    power_to_db_kernel<<<blocks, threads, 0, stream>>>(
        power, db_out, size, ref, min_db
    );
}

// =============================================================================
// FFT Shift (swap halves for proper frequency ordering)
// =============================================================================
__global__ void fft_shift_kernel(
    float* __restrict__ data,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half = size / 2;

    if (idx >= half) return;

    float tmp = data[idx];
    data[idx] = data[idx + half];
    data[idx + half] = tmp;
}

void fft_shift(float* data, int size, cudaStream_t stream) {
    int threads = 256;
    int blocks = (size / 2 + threads - 1) / threads;
    fft_shift_kernel<<<blocks, threads, 0, stream>>>(data, size);
}

// =============================================================================
// Full Welch PSD Pipeline (Fused)
// =============================================================================
PSDContext* psd_create_context(int fft_size, int max_segments) {
    PSDContext* ctx = new PSDContext();
    ctx->fft_size = fft_size;
    ctx->max_segments = max_segments;

    // Create FFT plan for batched execution
    int n[1] = {fft_size};
    cufftPlanMany(&ctx->fft_plan, 1, n,
                  nullptr, 1, fft_size,  // input layout
                  nullptr, 1, fft_size,  // output layout
                  CUFFT_C2C, max_segments);

    // Allocate buffers
    cudaMalloc(&ctx->window, fft_size * sizeof(float));
    cudaMalloc(&ctx->segments, fft_size * max_segments * sizeof(cuComplex));
    cudaMalloc(&ctx->segment_power, fft_size * max_segments * sizeof(float));
    cudaMalloc(&ctx->psd_linear, fft_size * sizeof(float));
    cudaMalloc(&ctx->psd_db, fft_size * sizeof(float));

    // Generate window
    generate_hann_window(ctx->window, fft_size, 0);

    // Compute window power sum for scaling
    float* h_window = new float[fft_size];
    cudaMemcpy(h_window, ctx->window, fft_size * sizeof(float), cudaMemcpyDeviceToHost);
    ctx->window_power_sum = 0.0f;
    for (int i = 0; i < fft_size; i++) {
        ctx->window_power_sum += h_window[i] * h_window[i];
    }
    delete[] h_window;

    return ctx;
}

void psd_destroy_context(PSDContext* ctx) {
    if (ctx) {
        cufftDestroy(ctx->fft_plan);
        cudaFree(ctx->window);
        cudaFree(ctx->segments);
        cudaFree(ctx->segment_power);
        cudaFree(ctx->psd_linear);
        cudaFree(ctx->psd_db);
        delete ctx;
    }
}

void psd_compute(
    PSDContext* ctx,
    const cuComplex* signal,
    int signal_len,
    float sample_rate,
    float overlap,
    cudaStream_t stream
) {
    int hop_size = static_cast<int>(ctx->fft_size * (1.0f - overlap));
    int num_segments = (signal_len - ctx->fft_size) / hop_size + 1;
    num_segments = min(num_segments, ctx->max_segments);

    float scale = 1.0f / (ctx->fft_size * ctx->window_power_sum);

    // Set stream for FFT
    cufftSetStream(ctx->fft_plan, stream);

    // 1. Segment and window
    segment_and_window(signal, ctx->segments, ctx->window,
                       signal_len, ctx->fft_size, hop_size, num_segments, stream);

    // 2. Batched FFT
    cufftExecC2C(ctx->fft_plan, ctx->segments, ctx->segments, CUFFT_FORWARD);

    // 3. Magnitude squared
    magnitude_squared(ctx->segments, ctx->segment_power,
                      ctx->fft_size, num_segments, scale, stream);

    // 4. Average across segments
    average_psd(ctx->segment_power, ctx->psd_linear,
                ctx->fft_size, num_segments, stream);

    // 5. Convert to dB
    power_to_db(ctx->psd_linear, ctx->psd_db,
                ctx->fft_size, 1.0f, -200.0f, stream);

    // 6. FFT shift for proper frequency ordering
    fft_shift(ctx->psd_db, ctx->fft_size, stream);
}

} // namespace kernels
} // namespace rf_forensics
