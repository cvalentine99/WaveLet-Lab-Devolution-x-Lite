/**
 * CFAR Operator Implementation - Holoscan v2.6.0 Compatible
 */

#include "cfar_op.hpp"
#include <cuda_runtime.h>
#include <cmath>

namespace rf_forensics {
namespace operators {

CFAROp::~CFAROp() {
    if (cfar_ctx_) {
        kernels::cfar_destroy_context(cfar_ctx_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void CFAROp::setup(holoscan::OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("psd_in");
    spec.output<holoscan::gxf::Entity>("detections_out");

    spec.param(num_reference_, "num_reference", "Reference Cells", "CFAR reference cells", 32);
    spec.param(num_guard_, "num_guard", "Guard Cells", "CFAR guard cells", 4);
    spec.param(pfa_, "pfa", "Pfa", "Probability of false alarm", 1.0e-6);
    spec.param(min_snr_, "min_snr", "Min SNR", "Minimum SNR for detection (dB)", 6.0f);
    spec.param(max_bins_, "max_bins", "Max Bins", "Maximum frequency bins", 2048);
    spec.param(max_peaks_, "max_peaks", "Max Peaks", "Maximum peaks to extract", 128);
}

void CFAROp::start() {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    cfar_ctx_ = kernels::cfar_create_context(max_bins_.get(), max_peaks_.get());

    // Compute alpha from Pfa
    int N = num_reference_.get();
    double pfa = pfa_.get();
    alpha_ = static_cast<float>(N * (std::pow(pfa, -1.0 / N) - 1.0));
    alpha_ = 10.0f * std::log10(alpha_);

    HOLOSCAN_LOG_INFO("CFAROp started: num_ref={}, pfa={:.2e}, alpha={:.2f} dB",
                      num_reference_.get(), pfa_.get(), alpha_);
}

void CFAROp::stop() {
    if (cfar_ctx_) {
        kernels::cfar_destroy_context(cfar_ctx_);
        cfar_ctx_ = nullptr;
    }
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    HOLOSCAN_LOG_INFO("CFAROp stopped");
}

void CFAROp::compute(holoscan::InputContext& op_input,
                     holoscan::OutputContext& op_output,
                     holoscan::ExecutionContext& context) {
    auto maybe_input = op_input.receive<holoscan::gxf::Entity>("psd_in");
    if (!maybe_input) {
        HOLOSCAN_LOG_WARN("CFAROp: No input received");
        return;
    }

    // For now, simplified - would get PSD data from shared context
    // This is a placeholder until full tensor passing is implemented

    auto out_entity = holoscan::gxf::Entity::New(&context);
    if (!out_entity) {
        HOLOSCAN_LOG_ERROR("CFAROp: Failed to create output entity");
        return;
    }

    op_output.emit(out_entity.value(), "detections_out");
}

} // namespace operators
} // namespace rf_forensics
