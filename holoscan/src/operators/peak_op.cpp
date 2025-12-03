/**
 * Peak Operator Implementation - Holoscan v2.6.0 Compatible
 */

#include "peak_op.hpp"
#include <cuda_runtime.h>
#include <chrono>

namespace rf_forensics {
namespace operators {

void PeakOp::setup(holoscan::OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("mask_in");
    spec.output<holoscan::gxf::Entity>("detections_out");

    spec.param(center_freq_, "center_freq", "Center Frequency", "Hz", 915.0e6);
    spec.param(sample_rate_, "sample_rate", "Sample Rate", "Hz", 10.0e6);
    spec.param(fft_size_, "fft_size", "FFT Size", "bins", 1024);
    spec.param(max_detections_, "max_detections", "Max Detections", "per frame", 128);
}

void PeakOp::start() {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);

    int max_det = max_detections_.get();
    cudaMallocHost(&h_peak_indices_, max_det * sizeof(int));
    cudaMallocHost(&h_peak_powers_, max_det * sizeof(float));
    cudaMallocHost(&h_peak_snrs_, max_det * sizeof(float));
    cudaMallocHost(&h_num_peaks_, sizeof(int));

    HOLOSCAN_LOG_INFO("PeakOp started: max_detections={}", max_det);
}

void PeakOp::stop() {
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }

    if (h_peak_indices_) cudaFreeHost(h_peak_indices_);
    if (h_peak_powers_) cudaFreeHost(h_peak_powers_);
    if (h_peak_snrs_) cudaFreeHost(h_peak_snrs_);
    if (h_num_peaks_) cudaFreeHost(h_num_peaks_);

    h_peak_indices_ = nullptr;
    h_peak_powers_ = nullptr;
    h_peak_snrs_ = nullptr;
    h_num_peaks_ = nullptr;

    HOLOSCAN_LOG_INFO("PeakOp stopped: {} total detections in {} frames",
                      total_detections_, total_frames_);
}

void PeakOp::compute(holoscan::InputContext& op_input,
                     holoscan::OutputContext& op_output,
                     holoscan::ExecutionContext& context) {
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t timestamp_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        now.time_since_epoch()).count();

    auto maybe_input = op_input.receive<holoscan::gxf::Entity>("mask_in");
    if (!maybe_input) {
        return;
    }

    // Simplified for v2.6.0 - just pass through for now
    total_frames_++;

    auto out_message = holoscan::gxf::Entity::New(&context);
    if (!out_message) {
        return;
    }

    op_output.emit(out_message.value(), "detections_out");
}

} // namespace operators
} // namespace rf_forensics
