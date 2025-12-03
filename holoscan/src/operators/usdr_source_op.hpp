/**
 * USDR Source Operator for RF Forensics Holoscan Pipeline
 *
 * High-performance IQ acquisition from uSDR PCIe hardware at up to 65 MSPS.
 * Uses memory-mapped DMA ring buffers with optional CUDA host registration
 * for near-zero-copy GPU transfers.
 *
 * Output: IQ samples (cuComplex tensor on GPU)
 */

#pragma once

#include <holoscan/holoscan.hpp>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <atomic>
#include <memory>
#include <cstdint>

// Forward declarations for libusdr types
struct dm_dev;
typedef struct dm_dev* pdm_dev_t;
struct usdr_dms;
typedef struct usdr_dms* pusdr_dms_t;

namespace rf_forensics {
namespace operators {

/**
 * USDR PCIe SDR Source Operator
 *
 * Acquires IQ samples via PCIe DMA and transfers to GPU.
 * Supports sample rates from 0.1 to 65 MSPS.
 *
 * Parameters:
 *   center_freq: Center frequency in Hz (default: 915 MHz)
 *   sample_rate: Sample rate in Hz (default: 50 MHz)
 *   buffer_size: Samples per buffer (default: 65536)
 *   gain_lna: LNA gain 0-30 dB (default: 20)
 *   gain_tia: TIA gain 0,3,9,12 dB (default: 9)
 *   gain_pga: PGA gain 0-32 dB (default: 12)
 *   rx_path: Antenna path LNAH/LNAL/LNAW (default: LNAL)
 *   simulate: Use simulated data if true (default: false)
 *   use_cuda_host_register: Register DMA buffers with CUDA (default: true)
 *
 * Output Ports:
 *   iq_out: GPU tensor of cuComplex samples
 */
class USDRSourceOp : public holoscan::Operator {
public:
    HOLOSCAN_OPERATOR_FORWARD_ARGS(USDRSourceOp)

    USDRSourceOp() = default;
    ~USDRSourceOp() override;

    void setup(holoscan::OperatorSpec& spec) override;
    void start() override;
    void stop() override;
    void compute(holoscan::InputContext& op_input,
                 holoscan::OutputContext& op_output,
                 holoscan::ExecutionContext& context) override;

private:
    // SDR Configuration Parameters
    holoscan::Parameter<double> center_freq_;      // Hz
    holoscan::Parameter<double> sample_rate_;      // Hz (up to 65 MSPS)
    holoscan::Parameter<double> bandwidth_;        // Hz
    holoscan::Parameter<int> buffer_size_;         // Samples per buffer
    holoscan::Parameter<int> gain_lna_;            // LNA gain dB
    holoscan::Parameter<int> gain_tia_;            // TIA gain dB
    holoscan::Parameter<int> gain_pga_;            // PGA gain dB
    holoscan::Parameter<std::string> rx_path_;     // LNAH, LNAL, LNAW
    holoscan::Parameter<bool> simulate_;           // Simulation mode
    holoscan::Parameter<bool> use_cuda_register_;  // CUDA host register DMA buffers

    // CUDA resources
    cudaStream_t cuda_stream_ = nullptr;
    cuComplex* gpu_buffer_ = nullptr;         // Device memory (pre-allocated)
    cuComplex* pinned_buffer_ = nullptr;      // Fallback pinned memory
    bool dma_registered_ = false;             // DMA buffer registered with CUDA

    // libusdr handles
    pdm_dev_t usdr_device_ = nullptr;
    pusdr_dms_t usdr_stream_ = nullptr;
    void* dma_buffer_ = nullptr;              // mmap'd DMA ring buffer
    size_t dma_buffer_size_ = 0;

    // Ring buffer state
    int current_buf_idx_ = 0;
    int num_dma_buffers_ = 0;
    size_t dma_buf_stride_ = 0;

    // Statistics
    std::atomic<uint64_t> total_samples_{0};
    std::atomic<uint64_t> total_frames_{0};
    std::atomic<uint64_t> dropped_samples_{0};
    double last_timestamp_ = 0.0;

    // Internal methods
    bool init_usdr_device();
    bool configure_usdr();
    bool create_usdr_stream();
    void close_usdr();

    int receive_samples(cuComplex* gpu_dest, int max_samples);
    void generate_test_signal(cuComplex* buffer, int num_samples);

    // VFS path constants (matching libusdr)
    static constexpr const char* VFS_RX_FREQ = "/dm/sdr/0/rx/freqency";  // Note: firmware typo
    static constexpr const char* VFS_RX_BW = "/dm/sdr/0/rx/bandwidth";
    static constexpr const char* VFS_RX_PATH = "/dm/sdr/0/rx/path";
    static constexpr const char* VFS_RX_GAIN_LNA = "/dm/sdr/0/rx/gain/lna";
    static constexpr const char* VFS_RX_GAIN_VGA = "/dm/sdr/0/rx/gain/vga";
    static constexpr const char* VFS_RX_GAIN_PGA = "/dm/sdr/0/rx/gain/pga";
    static constexpr const char* VFS_RATE = "/dm/rate/rxtxadcdac";
};

} // namespace operators
} // namespace rf_forensics
