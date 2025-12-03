/**
 * Peak Operator for RF Forensics Holoscan Pipeline
 *
 * Extracts and formats detection results for downstream processing.
 * Converts GPU tensors to structured detection records.
 */

#pragma once

#include <holoscan/holoscan.hpp>
#include <cuda_runtime.h>
#include <vector>

namespace rf_forensics {
namespace operators {

// Detection record structure
struct Detection {
    int bin_index;
    float frequency_hz;
    float power_db;
    float snr_db;
    uint64_t timestamp_ns;
};

class PeakOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(PeakOp)

    PeakOp() = default;
    ~PeakOp() override = default;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;

private:
    // Parameters
    holoscan::Parameter<double> center_freq_;
    holoscan::Parameter<double> sample_rate_;
    holoscan::Parameter<int> fft_size_;
    holoscan::Parameter<int> max_detections_;

    // CUDA resources
    cudaStream_t stream_ = nullptr;

    // Host buffers for D2H transfer
    int* h_peak_indices_ = nullptr;
    float* h_peak_powers_ = nullptr;
    float* h_peak_snrs_ = nullptr;
    int* h_num_peaks_ = nullptr;

    // Statistics
    uint64_t total_detections_ = 0;
    uint64_t total_frames_ = 0;
};

} // namespace operators
} // namespace rf_forensics
