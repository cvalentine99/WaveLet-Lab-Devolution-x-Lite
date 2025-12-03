/**
 * PSD Operator for RF Forensics Holoscan Pipeline
 *
 * Computes Power Spectral Density using Welch's method.
 * Input: IQ samples (cuComplex tensor)
 * Output: PSD in dB (float tensor)
 */

#pragma once

#include <holoscan/holoscan.hpp>
#include "../kernels/psd_kernels.cuh"

namespace rf_forensics {
namespace operators {

class PSDOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(PSDOp)

    PSDOp() = default;
    ~PSDOp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;

private:
    // Parameters
    holoscan::Parameter<int> fft_size_;
    holoscan::Parameter<float> overlap_;
    holoscan::Parameter<float> sample_rate_;
    holoscan::Parameter<int> max_segments_;

    // CUDA resources
    cudaStream_t stream_ = nullptr;
    kernels::PSDContext* psd_ctx_ = nullptr;

    // Output tensor allocator
    holoscan::Parameter<std::shared_ptr<holoscan::Allocator>> allocator_;
};

} // namespace operators
} // namespace rf_forensics
