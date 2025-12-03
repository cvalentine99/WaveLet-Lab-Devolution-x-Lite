/**
 * Python Bindings for RF Forensics Holoscan Pipeline
 *
 * Provides Python interface for:
 * - Configuration and initialization
 * - Async execution control
 * - Detection results retrieval
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <holoscan/holoscan.hpp>
#include "../operators/usdr_source_op.hpp"
#include "../operators/psd_op.hpp"
#include "../operators/cfar_op.hpp"
#include "../operators/peak_op.hpp"

#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>

namespace py = pybind11;

namespace rf_forensics {
namespace python {

// Detection record for Python
struct PyDetection {
    int bin_index;
    double frequency_hz;
    float power_db;
    float snr_db;
    uint64_t timestamp_ns;
};

// Thread-safe detection queue
class DetectionQueue {
public:
    void push(const std::vector<PyDetection>& detections) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& det : detections) {
            queue_.push(det);
        }
        cv_.notify_one();
    }

    std::vector<PyDetection> pop_all(int timeout_ms = 100) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) {
            cv_.wait_for(lock, std::chrono::milliseconds(timeout_ms));
        }

        std::vector<PyDetection> result;
        while (!queue_.empty()) {
            result.push_back(queue_.front());
            queue_.pop();
        }
        return result;
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        while (!queue_.empty()) queue_.pop();
    }

private:
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    std::queue<PyDetection> queue_;
};


// Python-friendly wrapper around the Holoscan application
class HoloscanPipeline {
public:
    HoloscanPipeline() : running_(false) {
        detection_queue_ = std::make_shared<DetectionQueue>();
    }

    ~HoloscanPipeline() {
        stop();
    }

    // Configure the pipeline
    void configure(
        double center_freq = 915.0e6,
        double sample_rate = 10.0e6,
        int buffer_size = 500000,
        float gain = 40.0f,
        bool simulate = false,
        int fft_size = 1024,
        float overlap = 0.5f,
        int num_reference = 32,
        int num_guard = 4,
        double pfa = 1e-6,
        float min_snr = 6.0f
    ) {
        config_.center_freq = center_freq;
        config_.sample_rate = sample_rate;
        config_.buffer_size = buffer_size;
        config_.gain = gain;
        config_.simulate = simulate;
        config_.fft_size = fft_size;
        config_.overlap = overlap;
        config_.num_reference = num_reference;
        config_.num_guard = num_guard;
        config_.pfa = pfa;
        config_.min_snr = min_snr;
    }

    // Load configuration from YAML file
    void load_config(const std::string& yaml_path) {
        config_path_ = yaml_path;
    }

    // Start the pipeline (non-blocking)
    void start() {
        if (running_) {
            throw std::runtime_error("Pipeline already running");
        }

        running_ = true;
        detection_queue_->clear();

        // Start pipeline in background thread
        pipeline_thread_ = std::thread([this]() {
            run_pipeline();
        });
    }

    // Stop the pipeline
    void stop() {
        if (!running_) return;

        running_ = false;

        // Signal application to stop
        if (app_) {
            // Holoscan applications can be interrupted
        }

        // Wait for thread to finish
        if (pipeline_thread_.joinable()) {
            pipeline_thread_.join();
        }

        app_.reset();
    }

    // Check if running
    bool is_running() const {
        return running_;
    }

    // Get detections (blocking with timeout)
    std::vector<PyDetection> get_detections(int timeout_ms = 100) {
        return detection_queue_->pop_all(timeout_ms);
    }

    // Get detection queue size
    size_t pending_detections() const {
        return detection_queue_->size();
    }

    // Get statistics
    py::dict get_stats() const {
        py::dict stats;
        stats["running"] = running_.load();
        stats["total_frames"] = total_frames_.load();
        stats["total_detections"] = total_detections_.load();
        stats["center_freq"] = config_.center_freq;
        stats["sample_rate"] = config_.sample_rate;
        return stats;
    }

private:
    // Pipeline configuration
    struct Config {
        double center_freq = 915.0e6;
        double sample_rate = 10.0e6;
        int buffer_size = 500000;
        float gain = 40.0f;
        bool simulate = false;
        int fft_size = 1024;
        float overlap = 0.5f;
        int num_reference = 32;
        int num_guard = 4;
        double pfa = 1e-6;
        float min_snr = 6.0f;
    } config_;

    std::string config_path_;

    // Holoscan application
    std::shared_ptr<holoscan::Application> app_;
    std::thread pipeline_thread_;
    std::atomic<bool> running_;

    // Detection output
    std::shared_ptr<DetectionQueue> detection_queue_;

    // Statistics
    std::atomic<uint64_t> total_frames_{0};
    std::atomic<uint64_t> total_detections_{0};

    void run_pipeline() {
        try {
            // Create inline application with current config
            class ConfiguredApp : public holoscan::Application {
            public:
                ConfiguredApp(const Config& cfg, std::shared_ptr<DetectionQueue> queue)
                    : config_(cfg), detection_queue_(queue) {}

                void compose() override {
                    using namespace holoscan;

                    auto usdr_source = make_operator<operators::USDRSourceOp>(
                        "usdr_source",
                        Arg("center_freq", config_.center_freq),
                        Arg("sample_rate", config_.sample_rate),
                        Arg("buffer_size", config_.buffer_size),
                        Arg("gain", config_.gain),
                        Arg("simulate", config_.simulate)
                    );

                    auto psd_op = make_operator<operators::PSDOp>(
                        "psd",
                        Arg("fft_size", config_.fft_size),
                        Arg("overlap", config_.overlap),
                        Arg("sample_rate", static_cast<float>(config_.sample_rate)),
                        Arg("max_segments", 512)
                    );

                    auto cfar_op = make_operator<operators::CFAROp>(
                        "cfar",
                        Arg("num_reference", config_.num_reference),
                        Arg("num_guard", config_.num_guard),
                        Arg("pfa", config_.pfa),
                        Arg("min_snr", config_.min_snr),
                        Arg("max_bins", config_.fft_size),
                        Arg("max_peaks", 128)
                    );

                    auto peak_op = make_operator<operators::PeakOp>(
                        "peaks",
                        Arg("center_freq", config_.center_freq),
                        Arg("sample_rate", config_.sample_rate),
                        Arg("fft_size", config_.fft_size),
                        Arg("max_detections", 128)
                    );

                    add_flow(usdr_source, psd_op, {{"iq_out", "iq_in"}});
                    add_flow(psd_op, cfar_op, {{"psd_out", "psd_in"}});
                    add_flow(cfar_op, peak_op, {{"detections_out", "mask_in"}});
                }

            private:
                Config config_;
                std::shared_ptr<DetectionQueue> detection_queue_;
            };

            app_ = holoscan::make_application<ConfiguredApp>(config_, detection_queue_);

            if (!config_path_.empty()) {
                app_->config(config_path_);
            }

            app_->run();

        } catch (const std::exception& e) {
            HOLOSCAN_LOG_ERROR("Pipeline error: {}", e.what());
        }

        running_ = false;
    }
};

} // namespace python
} // namespace rf_forensics


// =============================================================================
// Python Module Definition
// =============================================================================
PYBIND11_MODULE(rf_forensics_holoscan, m) {
    m.doc() = "RF Forensics Holoscan Pipeline - High-performance GPU signal processing";

    using namespace rf_forensics::python;

    // Detection struct
    py::class_<PyDetection>(m, "Detection")
        .def(py::init<>())
        .def_readwrite("bin_index", &PyDetection::bin_index)
        .def_readwrite("frequency_hz", &PyDetection::frequency_hz)
        .def_readwrite("power_db", &PyDetection::power_db)
        .def_readwrite("snr_db", &PyDetection::snr_db)
        .def_readwrite("timestamp_ns", &PyDetection::timestamp_ns)
        .def("__repr__", [](const PyDetection& d) {
            return "<Detection freq=" + std::to_string(d.frequency_hz / 1e6) +
                   " MHz, power=" + std::to_string(d.power_db) +
                   " dB, SNR=" + std::to_string(d.snr_db) + " dB>";
        });

    // Main pipeline class
    py::class_<HoloscanPipeline>(m, "HoloscanPipeline")
        .def(py::init<>())
        .def("configure", &HoloscanPipeline::configure,
             py::arg("center_freq") = 915.0e6,
             py::arg("sample_rate") = 10.0e6,
             py::arg("buffer_size") = 500000,
             py::arg("gain") = 40.0f,
             py::arg("simulate") = false,
             py::arg("fft_size") = 1024,
             py::arg("overlap") = 0.5f,
             py::arg("num_reference") = 32,
             py::arg("num_guard") = 4,
             py::arg("pfa") = 1e-6,
             py::arg("min_snr") = 6.0f,
             "Configure pipeline parameters")
        .def("load_config", &HoloscanPipeline::load_config,
             py::arg("yaml_path"),
             "Load configuration from YAML file")
        .def("start", &HoloscanPipeline::start,
             "Start the pipeline (non-blocking)")
        .def("stop", &HoloscanPipeline::stop,
             "Stop the pipeline")
        .def("is_running", &HoloscanPipeline::is_running,
             "Check if pipeline is running")
        .def("get_detections", &HoloscanPipeline::get_detections,
             py::arg("timeout_ms") = 100,
             "Get pending detections (blocking with timeout)")
        .def("pending_detections", &HoloscanPipeline::pending_detections,
             "Get number of pending detections")
        .def("get_stats", &HoloscanPipeline::get_stats,
             "Get pipeline statistics");

    // Version info
    m.attr("__version__") = "1.0.0";
    m.attr("__target_throughput__") = "9.5+ MSPS";
    m.attr("__target_latency__") = "<1ms";
}
