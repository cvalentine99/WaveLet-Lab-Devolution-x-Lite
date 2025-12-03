/**
 * USDR Source Operator Implementation - Holoscan v2.6.0 Compatible
 *
 * High-performance SDR acquisition using libusdr with CUDA integration.
 * Targets 50 MSPS sustained throughput with minimal latency.
 */

#include "usdr_source_op.hpp"
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <chrono>

// libusdr headers (when available)
#ifdef HAS_USDR
extern "C" {
#include <dm_dev.h>
#include <dm_stream.h>
}
#endif

namespace rf_forensics {
namespace operators {

USDRSourceOp::~USDRSourceOp() {
    stop();
}

void USDRSourceOp::setup(holoscan::OperatorSpec& spec) {
    spec.output<holoscan::gxf::Entity>("iq_out");

    spec.param(center_freq_, "center_freq", "Center Frequency", "Hz", 915.0e6);
    spec.param(sample_rate_, "sample_rate", "Sample Rate", "Hz (max 65 MSPS)", 50.0e6);
    spec.param(bandwidth_, "bandwidth", "Bandwidth", "Hz", 40.0e6);
    spec.param(buffer_size_, "buffer_size", "Buffer Size", "samples per buffer", 65536);
    spec.param(gain_lna_, "gain_lna", "LNA Gain", "dB (0-30)", 20);
    spec.param(gain_tia_, "gain_tia", "TIA Gain", "dB (0,3,9,12)", 9);
    spec.param(gain_pga_, "gain_pga", "PGA Gain", "dB (0-32)", 12);
    spec.param(rx_path_, "rx_path", "RX Path", "LNAH, LNAL, LNAW", std::string("LNAL"));
    spec.param(simulate_, "simulate", "Simulate", "Use simulated data", false);
    spec.param(use_cuda_register_, "use_cuda_register", "CUDA Host Register", "Register DMA with CUDA", true);
}

void USDRSourceOp::start() {
    HOLOSCAN_LOG_INFO("USDRSourceOp: Starting (freq={:.3f} MHz, rate={:.2f} MSPS)",
                      center_freq_.get() / 1e6, sample_rate_.get() / 1e6);

    cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking);

    size_t buffer_bytes = buffer_size_.get() * sizeof(cuComplex);
    cudaMalloc(&gpu_buffer_, buffer_bytes);
    cudaMallocHost(&pinned_buffer_, buffer_bytes);

    if (!simulate_.get()) {
        if (!init_usdr_device()) {
            HOLOSCAN_LOG_WARN("USDR hardware not available, using simulation mode");
        } else if (!configure_usdr()) {
            HOLOSCAN_LOG_ERROR("Failed to configure USDR");
            close_usdr();
        } else if (!create_usdr_stream()) {
            HOLOSCAN_LOG_ERROR("Failed to create USDR stream");
            close_usdr();
        }
    }

    HOLOSCAN_LOG_INFO("USDRSourceOp: Started successfully");
}

void USDRSourceOp::stop() {
    HOLOSCAN_LOG_INFO("USDRSourceOp: Stopping...");

    close_usdr();

    if (cuda_stream_) {
        cudaStreamSynchronize(cuda_stream_);
        cudaStreamDestroy(cuda_stream_);
        cuda_stream_ = nullptr;
    }

    if (gpu_buffer_) {
        cudaFree(gpu_buffer_);
        gpu_buffer_ = nullptr;
    }

    if (pinned_buffer_) {
        cudaFreeHost(pinned_buffer_);
        pinned_buffer_ = nullptr;
    }

    HOLOSCAN_LOG_INFO("USDRSourceOp: Stopped ({} total samples, {} dropped)",
                      total_samples_.load(), dropped_samples_.load());
}

void USDRSourceOp::compute(holoscan::InputContext& op_input,
                           holoscan::OutputContext& op_output,
                           holoscan::ExecutionContext& context) {
    int num_samples = buffer_size_.get();
    int received = 0;

    if (usdr_stream_) {
        received = receive_samples(gpu_buffer_, num_samples);
    } else {
        // Simulation mode
        generate_test_signal(pinned_buffer_, num_samples);
        cudaMemcpyAsync(gpu_buffer_, pinned_buffer_,
                        num_samples * sizeof(cuComplex),
                        cudaMemcpyHostToDevice, cuda_stream_);
        received = num_samples;
    }

    if (received <= 0) {
        HOLOSCAN_LOG_WARN("USDRSourceOp: No samples received");
        return;
    }

    total_samples_ += received;
    total_frames_++;

    auto maybe_entity = holoscan::gxf::Entity::New(&context);
    if (!maybe_entity) {
        HOLOSCAN_LOG_ERROR("USDRSourceOp: Failed to create output entity");
        return;
    }
    auto out_entity = maybe_entity.value();

    op_output.emit(out_entity, "iq_out");
}

bool USDRSourceOp::init_usdr_device() {
#ifdef HAS_USDR
    // Discover and open USDR device
    char devlist[4096];
    int result = usdr_dmd_discovery("", sizeof(devlist), devlist);
    if (result < 0 || strlen(devlist) == 0) {
        HOLOSCAN_LOG_ERROR("No USDR devices found");
        return false;
    }

    HOLOSCAN_LOG_INFO("Found USDR device: {}", devlist);

    result = usdr_dmd_create_string("", &usdr_device_);
    if (result < 0) {
        HOLOSCAN_LOG_ERROR("Failed to open USDR device: {}", result);
        usdr_device_ = nullptr;
        return false;
    }

    HOLOSCAN_LOG_INFO("USDR device opened successfully");
    return true;
#else
    HOLOSCAN_LOG_WARN("USDR support not compiled in");
    return false;
#endif
}

bool USDRSourceOp::configure_usdr() {
#ifdef HAS_USDR
    if (!usdr_device_) return false;

    uint64_t rate = static_cast<uint64_t>(sample_rate_.get());
    usdr_dme_set_uint(usdr_device_, "/dm/rate/rxtxadcdac", rate);

    uint64_t freq = static_cast<uint64_t>(center_freq_.get());
    usdr_dme_set_uint(usdr_device_, "/dm/sdr/0/rx/freqency", freq);

    uint64_t bw = static_cast<uint64_t>(bandwidth_.get());
    usdr_dme_set_uint(usdr_device_, "/dm/sdr/0/rx/bandwidth", bw);

    usdr_dme_set_uint(usdr_device_, "/dm/sdr/0/rx/gain/lna", gain_lna_.get());
    usdr_dme_set_uint(usdr_device_, "/dm/sdr/0/rx/gain/vga", gain_tia_.get());
    usdr_dme_set_uint(usdr_device_, "/dm/sdr/0/rx/gain/pga", gain_pga_.get());

    HOLOSCAN_LOG_INFO("USDR configured: freq={:.3f} MHz, rate={:.2f} MSPS",
                      freq / 1e6, rate / 1e6);
    return true;
#else
    return false;
#endif
}

bool USDRSourceOp::create_usdr_stream() {
#ifdef HAS_USDR
    if (!usdr_device_) return false;

    int result = usdr_dms_create(
        usdr_device_,
        "/ll/srx/0",
        "cf32@ci16",
        0x1,
        buffer_size_.get(),
        &usdr_stream_
    );

    if (result < 0) {
        HOLOSCAN_LOG_ERROR("Failed to create USDR stream: {}", result);
        usdr_stream_ = nullptr;
        return false;
    }

    result = usdr_dms_op(usdr_stream_, USDR_DMS_START, 0);
    if (result < 0) {
        HOLOSCAN_LOG_ERROR("Failed to start USDR stream: {}", result);
        usdr_dms_destroy(usdr_stream_);
        usdr_stream_ = nullptr;
        return false;
    }

    HOLOSCAN_LOG_INFO("USDR stream created and started");
    return true;
#else
    return false;
#endif
}

void USDRSourceOp::close_usdr() {
#ifdef HAS_USDR
    if (usdr_stream_) {
        usdr_dms_op(usdr_stream_, USDR_DMS_STOP, 0);
        usdr_dms_destroy(usdr_stream_);
        usdr_stream_ = nullptr;
    }

    if (dma_registered_ && dma_buffer_) {
        cudaHostUnregister(dma_buffer_);
        dma_registered_ = false;
    }

    if (usdr_device_) {
        usdr_dmd_close(usdr_device_);
        usdr_device_ = nullptr;
    }
#endif
}

int USDRSourceOp::receive_samples(cuComplex* gpu_dest, int max_samples) {
#ifdef HAS_USDR
    if (!usdr_stream_) return -1;

    void* recv_buffer = pinned_buffer_;
    void* buffers[1] = {recv_buffer};

    usdr_dms_recv_nfo_t nfo;
    int result = usdr_dms_recv(usdr_stream_, buffers, 100, &nfo);

    if (result < 0 || nfo.totsyms == 0) {
        return 0;
    }

    if (nfo.totlost > 0) {
        dropped_samples_ += nfo.totlost;
        HOLOSCAN_LOG_WARN("USDR: {} samples dropped", nfo.totlost);
    }

    last_timestamp_ = static_cast<double>(nfo.fsymtime) / sample_rate_.get();

    int samples_to_copy = std::min(static_cast<int>(nfo.totsyms), max_samples);
    cudaMemcpyAsync(gpu_dest, recv_buffer,
                    samples_to_copy * sizeof(cuComplex),
                    cudaMemcpyHostToDevice, cuda_stream_);

    return samples_to_copy;
#else
    return -1;
#endif
}

void USDRSourceOp::generate_test_signal(cuComplex* buffer, int num_samples) {
    static uint64_t sample_idx = 0;
    double fs = sample_rate_.get();

    struct TestTone { double offset_hz; double power_dbfs; };
    TestTone tones[] = {
        {0.0, -30.0}, {100e3, -40.0}, {-250e3, -45.0},
        {500e3, -50.0}, {1.0e6, -55.0}, {-2.0e6, -60.0}
    };
    int num_tones = sizeof(tones) / sizeof(tones[0]);

    for (int i = 0; i < num_samples; i++) {
        double t = static_cast<double>(sample_idx + i) / fs;
        float re = 0.0f, im = 0.0f;

        for (int j = 0; j < num_tones; j++) {
            double amplitude = std::pow(10.0, tones[j].power_dbfs / 20.0);
            double phase = 2.0 * M_PI * tones[j].offset_hz * t;
            re += amplitude * std::cos(phase);
            im += amplitude * std::sin(phase);
        }

        float noise_amplitude = std::pow(10.0f, -90.0f / 20.0f);
        float noise_re = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * noise_amplitude;
        float noise_im = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 2.0f * noise_amplitude;

        buffer[i] = make_cuComplex(re + noise_re, im + noise_im);
    }

    sample_idx += num_samples;
    last_timestamp_ = static_cast<double>(sample_idx) / fs;
}

} // namespace operators
} // namespace rf_forensics
