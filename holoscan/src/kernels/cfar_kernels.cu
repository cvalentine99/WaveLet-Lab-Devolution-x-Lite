/**
 * CFAR CUDA Kernels for RF Forensics Holoscan Pipeline
 *
 * Implements Cell-Averaging Constant False Alarm Rate (CA-CFAR) detection.
 * Ported from Numba CUDA implementation for native C++ performance.
 */

#include "cfar_kernels.cuh"
#include <cuda_runtime.h>
#include <cmath>

namespace rf_forensics {
namespace kernels {

// =============================================================================
// CA-CFAR Detection Kernel
// =============================================================================
__global__ void ca_cfar_kernel(
    const float* __restrict__ psd_db,
    bool* __restrict__ detection_mask,
    float* __restrict__ snr_out,
    float* __restrict__ threshold_out,
    int n_bins,
    int num_reference,
    int num_guard,
    float alpha  // threshold multiplier
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bins) return;

    int half_ref = num_reference / 2;
    int half_guard = num_guard / 2;

    // Compute reference cell boundaries
    int lead_start = idx - half_ref - half_guard;
    int lead_end = idx - half_guard;
    int lag_start = idx + half_guard + 1;
    int lag_end = idx + half_guard + half_ref + 1;

    // Sum reference cells (handle edge wrapping)
    float sum = 0.0f;
    int count = 0;

    // Leading reference cells
    for (int i = lead_start; i < lead_end; i++) {
        int ref_idx = i;
        if (ref_idx < 0) ref_idx += n_bins;
        if (ref_idx >= 0 && ref_idx < n_bins) {
            sum += psd_db[ref_idx];
            count++;
        }
    }

    // Lagging reference cells
    for (int i = lag_start; i < lag_end; i++) {
        int ref_idx = i;
        if (ref_idx >= n_bins) ref_idx -= n_bins;
        if (ref_idx >= 0 && ref_idx < n_bins) {
            sum += psd_db[ref_idx];
            count++;
        }
    }

    // Compute threshold
    float noise_estimate = (count > 0) ? (sum / count) : psd_db[idx];
    float threshold = noise_estimate + alpha;

    // Cell under test
    float cut_value = psd_db[idx];

    // Detection decision
    bool detected = cut_value > threshold;
    float snr = cut_value - noise_estimate;

    // Store results
    detection_mask[idx] = detected;
    snr_out[idx] = snr;
    threshold_out[idx] = threshold;
}

void ca_cfar_detect(
    const float* psd_db,
    bool* detection_mask,
    float* snr_out,
    float* threshold_out,
    int n_bins,
    int num_reference,
    int num_guard,
    float alpha,
    cudaStream_t stream
) {
    int threads = 256;
    int blocks = (n_bins + threads - 1) / threads;
    ca_cfar_kernel<<<blocks, threads, 0, stream>>>(
        psd_db, detection_mask, snr_out, threshold_out,
        n_bins, num_reference, num_guard, alpha
    );
}

// =============================================================================
// OS-CFAR Detection Kernel (Ordered Statistic)
// =============================================================================
__device__ void insertion_sort(float* arr, int n) {
    for (int i = 1; i < n; i++) {
        float key = arr[i];
        int j = i - 1;
        while (j >= 0 && arr[j] > key) {
            arr[j + 1] = arr[j];
            j--;
        }
        arr[j + 1] = key;
    }
}

__global__ void os_cfar_kernel(
    const float* __restrict__ psd_db,
    bool* __restrict__ detection_mask,
    float* __restrict__ snr_out,
    float* __restrict__ threshold_out,
    int n_bins,
    int num_reference,
    int num_guard,
    int k_rank,  // which ordered statistic to use (k-th smallest)
    float alpha
) {
    extern __shared__ float ref_cells[];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bins) return;

    int half_ref = num_reference / 2;
    int half_guard = num_guard / 2;
    int total_ref = num_reference;

    // Local array for sorting (in shared memory per thread block)
    float* my_refs = &ref_cells[threadIdx.x * total_ref];
    int ref_count = 0;

    // Collect reference cells
    for (int offset = -half_ref - half_guard; offset <= half_ref + half_guard; offset++) {
        if (abs(offset) <= half_guard) continue;  // Skip guard cells

        int ref_idx = idx + offset;
        if (ref_idx < 0) ref_idx += n_bins;
        if (ref_idx >= n_bins) ref_idx -= n_bins;

        if (ref_count < total_ref) {
            my_refs[ref_count++] = psd_db[ref_idx];
        }
    }

    // Sort reference cells
    insertion_sort(my_refs, ref_count);

    // Get k-th ordered statistic
    int k = min(k_rank, ref_count - 1);
    float noise_estimate = my_refs[k];
    float threshold = noise_estimate + alpha;

    float cut_value = psd_db[idx];
    bool detected = cut_value > threshold;

    detection_mask[idx] = detected;
    snr_out[idx] = cut_value - noise_estimate;
    threshold_out[idx] = threshold;
}

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
    cudaStream_t stream
) {
    int threads = 128;  // Reduced for shared memory
    int blocks = (n_bins + threads - 1) / threads;
    size_t shared_mem = threads * num_reference * sizeof(float);

    os_cfar_kernel<<<blocks, threads, shared_mem, stream>>>(
        psd_db, detection_mask, snr_out, threshold_out,
        n_bins, num_reference, num_guard, k_rank, alpha
    );
}

// =============================================================================
// Peak Extraction Kernel
// =============================================================================
__global__ void extract_peaks_kernel(
    const bool* __restrict__ detection_mask,
    const float* __restrict__ psd_db,
    const float* __restrict__ snr_out,
    int* __restrict__ peak_indices,
    float* __restrict__ peak_powers,
    float* __restrict__ peak_snrs,
    int* __restrict__ num_peaks,
    int n_bins,
    int max_peaks,
    float min_snr
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_bins) return;

    if (detection_mask[idx] && snr_out[idx] >= min_snr) {
        // Check if local maximum (simple peak detection)
        bool is_peak = true;
        if (idx > 0 && psd_db[idx - 1] > psd_db[idx]) is_peak = false;
        if (idx < n_bins - 1 && psd_db[idx + 1] > psd_db[idx]) is_peak = false;

        if (is_peak) {
            int peak_idx = atomicAdd(num_peaks, 1);
            if (peak_idx < max_peaks) {
                peak_indices[peak_idx] = idx;
                peak_powers[peak_idx] = psd_db[idx];
                peak_snrs[peak_idx] = snr_out[idx];
            }
        }
    }
}

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
    cudaStream_t stream
) {
    // Reset peak count
    cudaMemsetAsync(num_peaks, 0, sizeof(int), stream);

    int threads = 256;
    int blocks = (n_bins + threads - 1) / threads;
    extract_peaks_kernel<<<blocks, threads, 0, stream>>>(
        detection_mask, psd_db, snr_out,
        peak_indices, peak_powers, peak_snrs, num_peaks,
        n_bins, max_peaks, min_snr
    );
}

// =============================================================================
// CFAR Context Management
// =============================================================================
CFARContext* cfar_create_context(int max_bins, int max_peaks) {
    CFARContext* ctx = new CFARContext();
    ctx->max_bins = max_bins;
    ctx->max_peaks = max_peaks;

    cudaMalloc(&ctx->detection_mask, max_bins * sizeof(bool));
    cudaMalloc(&ctx->snr_out, max_bins * sizeof(float));
    cudaMalloc(&ctx->threshold_out, max_bins * sizeof(float));
    cudaMalloc(&ctx->peak_indices, max_peaks * sizeof(int));
    cudaMalloc(&ctx->peak_powers, max_peaks * sizeof(float));
    cudaMalloc(&ctx->peak_snrs, max_peaks * sizeof(float));
    cudaMalloc(&ctx->num_peaks, sizeof(int));

    return ctx;
}

void cfar_destroy_context(CFARContext* ctx) {
    if (ctx) {
        cudaFree(ctx->detection_mask);
        cudaFree(ctx->snr_out);
        cudaFree(ctx->threshold_out);
        cudaFree(ctx->peak_indices);
        cudaFree(ctx->peak_powers);
        cudaFree(ctx->peak_snrs);
        cudaFree(ctx->num_peaks);
        delete ctx;
    }
}

void cfar_detect_and_extract(
    CFARContext* ctx,
    const float* psd_db,
    int n_bins,
    int num_reference,
    int num_guard,
    float alpha,
    float min_snr,
    cudaStream_t stream
) {
    // Run CFAR detection
    ca_cfar_detect(
        psd_db, ctx->detection_mask, ctx->snr_out, ctx->threshold_out,
        n_bins, num_reference, num_guard, alpha, stream
    );

    // Extract peaks
    extract_peaks(
        ctx->detection_mask, psd_db, ctx->snr_out,
        ctx->peak_indices, ctx->peak_powers, ctx->peak_snrs, ctx->num_peaks,
        n_bins, ctx->max_peaks, min_snr, stream
    );
}

} // namespace kernels
} // namespace rf_forensics
