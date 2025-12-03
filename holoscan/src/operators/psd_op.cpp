/**
 * PSD Operator Implementation - Holoscan v2.6.0 Compatible
 */

#include "psd_op.hpp"
#include <cuda_runtime.h>

namespace rf_forensics {
namespace operators {

PSDOp::~PSDOp() {
    if (psd_ctx_) {
        kernels::psd_destroy_context(psd_ctx_);
    }
    if (stream_) {
        cudaStreamDestroy(stream_);
    }
}

void PSDOp::setup(holoscan::OperatorSpec& spec) {
    spec.input<holoscan::gxf::Entity>("iq_in");
    spec.output<holoscan::gxf::Entity>("psd_out");

    spec.param(fft_size_, "fft_size", "FFT Size", "Number of FFT bins", 1024);
    spec.param(overlap_, "overlap", "Overlap", "Segment overlap ratio", 0.5f);
    spec.param(sample_rate_, "sample_rate", "Sample Rate", "Hz", 10.0e6f);
    spec.param(max_segments_, "max_segments", "Max Segments", "Max averaging segments", 512);
}

void PSDOp::start() {
    cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
    psd_ctx_ = kernels::psd_create_context(fft_size_.get(), max_segments_.get());
    HOLOSCAN_LOG_INFO("PSDOp started: fft_size={}, overlap={:.2f}",
                      fft_size_.get(), overlap_.get());
}

void PSDOp::stop() {
    if (psd_ctx_) {
        kernels::psd_destroy_context(psd_ctx_);
        psd_ctx_ = nullptr;
    }
    if (stream_) {
        cudaStreamSynchronize(stream_);
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
    HOLOSCAN_LOG_INFO("PSDOp stopped");
}

void PSDOp::compute(holoscan::InputContext& op_input,
                    holoscan::OutputContext& op_output,
                    holoscan::ExecutionContext& context) {
    auto maybe_input = op_input.receive<holoscan::gxf::Entity>("iq_in");
    if (!maybe_input) {
        HOLOSCAN_LOG_WARN("PSDOp: No input received");
        return;
    }

    // Get input entity and tensor
    auto& input_entity = maybe_input.value();
    auto maybe_tensor = input_entity.get<holoscan::Tensor>("iq");
    if (!maybe_tensor) {
        HOLOSCAN_LOG_ERROR("PSDOp: No IQ tensor in input");
        return;
    }

    auto input_tensor = *maybe_tensor;
    const cuComplex* iq_data = static_cast<const cuComplex*>(input_tensor.data());
    int signal_len = input_tensor.size();

    // Compute PSD
    kernels::psd_compute(psd_ctx_, iq_data, signal_len,
                         sample_rate_.get(), overlap_.get(), stream_);

    // Create output - simplified for v2.6.0
    auto maybe_out = holoscan::gxf::Entity::New(&context);
    if (!maybe_out) {
        HOLOSCAN_LOG_ERROR("PSDOp: Failed to create output entity");
        return;
    }
    auto out_entity = maybe_out.value();

    // For now, emit the entity directly
    op_output.emit(out_entity, "psd_out");
}

} // namespace operators
} // namespace rf_forensics
