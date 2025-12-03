/**
 * PSD CUDA Kernels Header
 */

#pragma once

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufft.h>

namespace rf_forensics {
namespace kernels {

// PSD computation context (pre-allocated buffers)
struct PSDContext {
    int fft_size;
    int max_segments;
    float window_power_sum;

    cufftHandle fft_plan;
    float* window;
    cuComplex* segments;
    float* segment_power;
    float* psd_linear;
    float* psd_db;
};

// Context management
PSDContext* psd_create_context(int fft_size, int max_segments);
void psd_destroy_context(PSDContext* ctx);

// Full Welch PSD computation
void psd_compute(
    PSDContext* ctx,
    const cuComplex* signal,
    int signal_len,
    float sample_rate,
    float overlap,
    cudaStream_t stream = 0
);

// Individual kernel functions (for custom pipelines)
void generate_hann_window(float* window, int size, cudaStream_t stream = 0);

void segment_and_window(
    const cuComplex* signal,
    cuComplex* segments,
    const float* window,
    int signal_len,
    int fft_size,
    int hop_size,
    int num_segments,
    cudaStream_t stream = 0
);

void magnitude_squared(
    const cuComplex* fft_out,
    float* power,
    int fft_size,
    int num_segments,
    float scale,
    cudaStream_t stream = 0
);

void average_psd(
    const float* segment_power,
    float* psd_out,
    int fft_size,
    int num_segments,
    cudaStream_t stream = 0
);

void power_to_db(
    const float* power,
    float* db_out,
    int size,
    float ref = 1.0f,
    float min_db = -200.0f,
    cudaStream_t stream = 0
);

void fft_shift(float* data, int size, cudaStream_t stream = 0);

} // namespace kernels
} // namespace rf_forensics
