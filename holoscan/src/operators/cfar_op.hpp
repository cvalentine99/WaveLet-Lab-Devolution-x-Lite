/**
 * CFAR Operator for RF Forensics Holoscan Pipeline
 *
 * Performs CA-CFAR detection on PSD data.
 * Input: PSD in dB (float tensor)
 * Output: Detection mask and SNR (bool + float tensors)
 */

#pragma once

#include <holoscan/holoscan.hpp>
#include "../kernels/cfar_kernels.cuh"

namespace rf_forensics {
namespace operators {

class CFAROp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(CFAROp)

    CFAROp() = default;
    ~CFAROp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;

private:
    // Parameters
    holoscan::Parameter<int> num_reference_;
    holoscan::Parameter<int> num_guard_;
    holoscan::Parameter<double> pfa_;  // Probability of false alarm
    holoscan::Parameter<float> min_snr_;
    holoscan::Parameter<int> max_bins_;
    holoscan::Parameter<int> max_peaks_;

    // CUDA resources
    cudaStream_t stream_ = nullptr;
    kernels::CFARContext* cfar_ctx_ = nullptr;

    // Computed threshold multiplier (alpha)
    float alpha_ = 0.0f;
};

} // namespace operators
} // namespace rf_forensics
