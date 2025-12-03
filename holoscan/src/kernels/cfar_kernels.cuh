/**
 * CFAR CUDA Kernels Header
 */

#pragma once

#include <cuda_runtime.h>

namespace rf_forensics {
namespace kernels {

// CFAR computation context (pre-allocated buffers)
struct CFARContext {
    int max_bins;
    int max_peaks;

    bool* detection_mask;
    float* snr_out;
    float* threshold_out;

    int* peak_indices;
    float* peak_powers;
    float* peak_snrs;
    int* num_peaks;
};

// Context management
CFARContext* cfar_create_context(int max_bins, int max_peaks);
void cfar_destroy_context(CFARContext* ctx);

// Combined detect + extract
void cfar_detect_and_extract(
    CFARContext* ctx,
    const float* psd_db,
    int n_bins,
    int num_reference,
    int num_guard,
    float alpha,
    float min_snr,
    cudaStream_t stream = 0
);

// CA-CFAR detection
void ca_cfar_detect(
    const float* psd_db,
    bool* detection_mask,
    float* snr_out,
    float* threshold_out,
    int n_bins,
    int num_reference,
    int num_guard,
    float alpha,
    cudaStream_t stream = 0
);

// OS-CFAR detection (Ordered Statistic)
void os_cfar_detect(
    const float* psd_db,
    bool* detection_mask,
    float* snr_out,
    float* threshold_out,
    int n_bins,
    int num_reference,
    int num_guard,
    int k_rank,
    float alpha,
    cudaStream_t stream = 0
);

// Peak extraction from detection mask
void extract_peaks(
    const bool* detection_mask,
    const float* psd_db,
    const float* snr_out,
    int* peak_indices,
    float* peak_powers,
    float* peak_snrs,
    int* num_peaks,
    int n_bins,
    int max_peaks,
    float min_snr,
    cudaStream_t stream = 0
);

} // namespace kernels
} // namespace rf_forensics
