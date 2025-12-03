/**
 * RF Forensics Holoscan Application
 *
 * Main application that composes the GPU processing pipeline:
 * USDR Source -> PSD (Welch) -> CFAR Detection -> Peak Extraction
 *
 * Target: 9.5+ MSPS throughput with <1ms latency
 */

#include <holoscan/holoscan.hpp>
#include "operators/usdr_source_op.hpp"
#include "operators/psd_op.hpp"
#include "operators/cfar_op.hpp"
#include "operators/peak_op.hpp"

namespace rf_forensics {

class RFForensicsApp : public holoscan::Application {
public:
    void compose() override {
        using namespace holoscan;

        // Get configuration parameters (from YAML or command line)
        double center_freq = from_config("sdr.center_freq").as<double>();
        double sample_rate = from_config("sdr.sample_rate").as<double>();
        int buffer_size = from_config("sdr.buffer_size").as<int>();
        float gain = from_config("sdr.gain").as<float>();
        bool simulate = from_config("sdr.simulate").as<bool>();

        int fft_size = from_config("psd.fft_size").as<int>();
        float overlap = from_config("psd.overlap").as<float>();

        int num_reference = from_config("cfar.num_reference").as<int>();
        int num_guard = from_config("cfar.num_guard").as<int>();
        double pfa = from_config("cfar.pfa").as<double>();
        float min_snr = from_config("cfar.min_snr").as<float>();

        // Create operators
        auto usdr_source = make_operator<operators::USDRSourceOp>(
            "usdr_source",
            Arg("center_freq", center_freq),
            Arg("sample_rate", sample_rate),
            Arg("buffer_size", buffer_size),
            Arg("gain", gain),
            Arg("simulate", simulate)
        );

        auto psd_op = make_operator<operators::PSDOp>(
            "psd",
            Arg("fft_size", fft_size),
            Arg("overlap", overlap),
            Arg("sample_rate", static_cast<float>(sample_rate)),
            Arg("max_segments", 512)
        );

        auto cfar_op = make_operator<operators::CFAROp>(
            "cfar",
            Arg("num_reference", num_reference),
            Arg("num_guard", num_guard),
            Arg("pfa", pfa),
            Arg("min_snr", min_snr),
            Arg("max_bins", fft_size),
            Arg("max_peaks", 128)
        );

        auto peak_op = make_operator<operators::PeakOp>(
            "peaks",
            Arg("center_freq", center_freq),
            Arg("sample_rate", sample_rate),
            Arg("fft_size", fft_size),
            Arg("max_detections", 128)
        );

        // Connect the pipeline graph
        // USDR -> PSD -> CFAR -> Peaks
        add_flow(usdr_source, psd_op, {{"iq_out", "iq_in"}});
        add_flow(psd_op, cfar_op, {{"psd_out", "psd_in"}});
        add_flow(cfar_op, peak_op, {{"detections_out", "mask_in"}});
    }

    // Set default configuration
    void set_defaults() {
        // SDR defaults
        config()["sdr"]["center_freq"] = 915.0e6;
        config()["sdr"]["sample_rate"] = 10.0e6;
        config()["sdr"]["buffer_size"] = 500000;
        config()["sdr"]["gain"] = 40.0f;
        config()["sdr"]["simulate"] = false;

        // PSD defaults
        config()["psd"]["fft_size"] = 1024;
        config()["psd"]["overlap"] = 0.5f;

        // CFAR defaults
        config()["cfar"]["num_reference"] = 32;
        config()["cfar"]["num_guard"] = 4;
        config()["cfar"]["pfa"] = 1.0e-6;
        config()["cfar"]["min_snr"] = 6.0f;
    }
};

} // namespace rf_forensics


// =============================================================================
// Main Entry Point
// =============================================================================
int main(int argc, char** argv) {
    // Parse command line arguments
    holoscan::ArgList args;
    args.add(holoscan::Arg("config-file", "rf_forensics.yaml"));

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];

        // Handle --config
        if (arg == "--config" && i + 1 < argc) {
            args.add(holoscan::Arg("config-file", argv[++i]));
        }
        // Handle --simulate
        else if (arg == "--simulate") {
            args.add(holoscan::Arg("sdr.simulate", true));
        }
        // Handle --freq
        else if (arg == "--freq" && i + 1 < argc) {
            args.add(holoscan::Arg("sdr.center_freq", std::stod(argv[++i])));
        }
        // Handle --rate
        else if (arg == "--rate" && i + 1 < argc) {
            args.add(holoscan::Arg("sdr.sample_rate", std::stod(argv[++i])));
        }
        // Handle --help
        else if (arg == "--help" || arg == "-h") {
            std::cout << "RF Forensics Holoscan Pipeline\n"
                      << "Usage: rf_forensics_app [options]\n"
                      << "\nOptions:\n"
                      << "  --config FILE    Load configuration from YAML file\n"
                      << "  --simulate       Use simulated SDR data\n"
                      << "  --freq HZ        Set center frequency\n"
                      << "  --rate HZ        Set sample rate\n"
                      << "  -h, --help       Show this help message\n"
                      << "\nTarget performance: 9.5+ MSPS, <1ms latency\n";
            return 0;
        }
    }

    // Create and run application
    auto app = holoscan::make_application<rf_forensics::RFForensicsApp>();
    app->set_defaults();

    // Load config file if specified
    std::string config_file = args.value<std::string>("config-file", "");
    if (!config_file.empty()) {
        try {
            app->config(config_file);
        } catch (const std::exception& e) {
            HOLOSCAN_LOG_WARN("Failed to load config file '{}': {}", config_file, e.what());
        }
    }

    // Apply command line overrides
    // (handled by Holoscan's arg parsing)

    HOLOSCAN_LOG_INFO("Starting RF Forensics Holoscan Pipeline...");
    HOLOSCAN_LOG_INFO("Target: 9.5+ MSPS throughput, <1ms latency");

    // Run the application
    app->run();

    return 0;
}
