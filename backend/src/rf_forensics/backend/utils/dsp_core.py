import numpy as np


def run_analysis_pipeline(iq_data: np.ndarray, desc):
    """
    Placeholder DSP pipeline.

    Replace with real PSD/CFAR/peaks or call into the existing pipeline.
    """
    return {
        "num_samples": int(len(iq_data)),
        "sample_rate": float(desc.sample_rate_hz),
        "center_freq": float(desc.center_frequency_hz or 0),
        "mean_power_db": float(10 * np.log10(np.mean(np.abs(iq_data) ** 2) + 1e-12)),
    }
