"""
Centralized API Validators

All input validation functions for API endpoints.
Consolidates validation logic for security and consistency.
"""

import re

from fastapi import HTTPException

# =============================================================================
# Hardware Bounds Constants
# =============================================================================

# Frequency bounds (70 MHz to 6 GHz typical for LMS7002M)
MIN_FREQ_HZ = 70_000_000  # 70 MHz
MAX_FREQ_HZ = 6_000_000_000  # 6 GHz

# Sample rate bounds
MIN_SAMPLE_RATE_HZ = 100_000  # 100 kHz
MAX_SAMPLE_RATE_HZ = 61_440_000  # 61.44 MHz (max for LMS7002M)

# Bandwidth bounds
MIN_BANDWIDTH_HZ = 100_000  # 100 kHz
MAX_BANDWIDTH_HZ = 56_000_000  # 56 MHz

# Gain bounds (device-specific, these are typical ranges)
MIN_GAIN_DB = 0
MAX_GAIN_DB = 76

# FFT size bounds (power of 2)
MIN_FFT_SIZE = 256
MAX_FFT_SIZE = 65536

# CFAR bounds
MIN_CFAR_GUARD_CELLS = 1
MAX_CFAR_GUARD_CELLS = 64
MIN_CFAR_REF_CELLS = 2
MAX_CFAR_REF_CELLS = 128


# =============================================================================
# Security Patterns
# =============================================================================

# Safe recording ID pattern (alphanumeric, underscore, hyphen, max 64 chars)
SAFE_RECORDING_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")

# Safe device ID pattern (alphanumeric with colons for USB addresses)
SAFE_DEVICE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_:-]{1,128}$")

# Safe preset name pattern
SAFE_PRESET_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\s]{1,64}$")


# =============================================================================
# Frequency Validators
# =============================================================================


def validate_frequency(
    freq_hz: int,
    min_hz: int = MIN_FREQ_HZ,
    max_hz: int = MAX_FREQ_HZ,
    field_name: str = "Frequency",
) -> int:
    """
    Validate frequency is within safe hardware bounds.

    Args:
        freq_hz: Frequency in Hz to validate
        min_hz: Minimum allowed frequency (default: 70 MHz)
        max_hz: Maximum allowed frequency (default: 6 GHz)
        field_name: Name to use in error message

    Returns:
        Validated frequency

    Raises:
        HTTPException: If frequency is out of bounds
    """
    if not min_hz <= freq_hz <= max_hz:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be between {min_hz / 1e6:.1f} MHz and {max_hz / 1e9:.1f} GHz",
        )
    return freq_hz


def validate_frequency_range(
    start_hz: int, end_hz: int, field_name: str = "Frequency range"
) -> tuple[int, int]:
    """
    Validate a frequency range is valid.

    Args:
        start_hz: Start frequency in Hz
        end_hz: End frequency in Hz
        field_name: Name to use in error message

    Returns:
        Tuple of (start_hz, end_hz)

    Raises:
        HTTPException: If range is invalid
    """
    start_hz = validate_frequency(start_hz, field_name=f"{field_name} start")
    end_hz = validate_frequency(end_hz, field_name=f"{field_name} end")

    if start_hz >= end_hz:
        raise HTTPException(status_code=400, detail=f"{field_name} start must be less than end")
    return start_hz, end_hz


# =============================================================================
# Sample Rate Validators
# =============================================================================


def validate_sample_rate(
    rate_hz: int, min_hz: int = MIN_SAMPLE_RATE_HZ, max_hz: int = MAX_SAMPLE_RATE_HZ
) -> int:
    """
    Validate sample rate is within safe hardware bounds.

    Args:
        rate_hz: Sample rate in Hz to validate
        min_hz: Minimum allowed rate (default: 100 kHz)
        max_hz: Maximum allowed rate (default: 61.44 MHz)

    Returns:
        Validated sample rate

    Raises:
        HTTPException: If sample rate is out of bounds
    """
    if not min_hz <= rate_hz <= max_hz:
        raise HTTPException(
            status_code=400,
            detail=f"Sample rate must be between {min_hz / 1e3:.0f} kHz and {max_hz / 1e6:.2f} MHz",
        )
    return rate_hz


# =============================================================================
# Bandwidth Validators
# =============================================================================


def validate_bandwidth(
    bandwidth_hz: int, min_hz: int = MIN_BANDWIDTH_HZ, max_hz: int = MAX_BANDWIDTH_HZ
) -> int:
    """
    Validate bandwidth is within safe bounds.

    Args:
        bandwidth_hz: Bandwidth in Hz to validate
        min_hz: Minimum allowed bandwidth (default: 100 kHz)
        max_hz: Maximum allowed bandwidth (default: 56 MHz)

    Returns:
        Validated bandwidth

    Raises:
        HTTPException: If bandwidth is out of bounds
    """
    if not min_hz <= bandwidth_hz <= max_hz:
        raise HTTPException(
            status_code=400,
            detail=f"Bandwidth must be between {min_hz / 1e3:.0f} kHz and {max_hz / 1e6:.1f} MHz",
        )
    return bandwidth_hz


# =============================================================================
# Gain Validators
# =============================================================================


def validate_gain(
    gain_db: float, min_db: float = MIN_GAIN_DB, max_db: float = MAX_GAIN_DB
) -> float:
    """
    Validate gain is within safe bounds.

    Args:
        gain_db: Gain in dB to validate
        min_db: Minimum allowed gain (default: 0 dB)
        max_db: Maximum allowed gain (default: 76 dB)

    Returns:
        Validated gain

    Raises:
        HTTPException: If gain is out of bounds
    """
    if not min_db <= gain_db <= max_db:
        raise HTTPException(
            status_code=400, detail=f"Gain must be between {min_db:.0f} dB and {max_db:.0f} dB"
        )
    return gain_db


# =============================================================================
# FFT/DSP Validators
# =============================================================================


def validate_fft_size(fft_size: int) -> int:
    """
    Validate FFT size is a power of 2 within bounds.

    Args:
        fft_size: FFT size to validate

    Returns:
        Validated FFT size

    Raises:
        HTTPException: If FFT size is invalid
    """
    # Check power of 2
    if fft_size <= 0 or (fft_size & (fft_size - 1)) != 0:
        raise HTTPException(status_code=400, detail="FFT size must be a power of 2")

    if not MIN_FFT_SIZE <= fft_size <= MAX_FFT_SIZE:
        raise HTTPException(
            status_code=400, detail=f"FFT size must be between {MIN_FFT_SIZE} and {MAX_FFT_SIZE}"
        )
    return fft_size


def validate_cfar_params(guard_cells: int, ref_cells: int) -> tuple[int, int]:
    """
    Validate CFAR detector parameters.

    Args:
        guard_cells: Number of guard cells
        ref_cells: Number of reference cells

    Returns:
        Tuple of (guard_cells, ref_cells)

    Raises:
        HTTPException: If parameters are invalid
    """
    if not MIN_CFAR_GUARD_CELLS <= guard_cells <= MAX_CFAR_GUARD_CELLS:
        raise HTTPException(
            status_code=400,
            detail=f"CFAR guard cells must be between {MIN_CFAR_GUARD_CELLS} and {MAX_CFAR_GUARD_CELLS}",
        )

    if not MIN_CFAR_REF_CELLS <= ref_cells <= MAX_CFAR_REF_CELLS:
        raise HTTPException(
            status_code=400,
            detail=f"CFAR reference cells must be between {MIN_CFAR_REF_CELLS} and {MAX_CFAR_REF_CELLS}",
        )

    return guard_cells, ref_cells


# =============================================================================
# String/ID Validators (Security)
# =============================================================================


def validate_recording_id(recording_id: str) -> str:
    """
    Validate recording ID format to prevent path traversal attacks.

    Args:
        recording_id: Recording ID to validate

    Returns:
        Validated recording ID

    Raises:
        HTTPException: If ID format is invalid
    """
    if not SAFE_RECORDING_ID_PATTERN.match(recording_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid recording ID format. Must be alphanumeric with underscores/hyphens, max 64 characters.",
        )
    return recording_id


def validate_device_id(device_id: str) -> str:
    """
    Validate device ID format.

    Args:
        device_id: Device ID to validate

    Returns:
        Validated device ID

    Raises:
        HTTPException: If ID format is invalid
    """
    if not SAFE_DEVICE_ID_PATTERN.match(device_id):
        raise HTTPException(status_code=400, detail="Invalid device ID format")
    return device_id


def validate_preset_name(name: str) -> str:
    """
    Validate preset name format.

    Args:
        name: Preset name to validate

    Returns:
        Validated preset name

    Raises:
        HTTPException: If name format is invalid
    """
    if not SAFE_PRESET_NAME_PATTERN.match(name):
        raise HTTPException(
            status_code=400,
            detail="Invalid preset name. Must be alphanumeric with spaces/underscores/hyphens, max 64 characters.",
        )
    return name


# =============================================================================
# Duration Validators
# =============================================================================


def validate_duration_seconds(
    duration: float | None,
    max_duration: float = 3600.0,  # 1 hour default max
) -> float | None:
    """
    Validate duration in seconds.

    Args:
        duration: Duration to validate (None means unlimited)
        max_duration: Maximum allowed duration

    Returns:
        Validated duration or None

    Raises:
        HTTPException: If duration is invalid
    """
    if duration is None:
        return None

    if duration <= 0:
        raise HTTPException(status_code=400, detail="Duration must be positive")

    if duration > max_duration:
        raise HTTPException(
            status_code=400, detail=f"Duration cannot exceed {max_duration:.0f} seconds"
        )

    return duration


# =============================================================================
# Power/dB Validators
# =============================================================================


def validate_power_dbm(
    power_dbm: float, min_dbm: float = -150.0, max_dbm: float = 30.0, field_name: str = "Power"
) -> float:
    """
    Validate power level in dBm.

    Args:
        power_dbm: Power in dBm to validate
        min_dbm: Minimum allowed power
        max_dbm: Maximum allowed power
        field_name: Name to use in error message

    Returns:
        Validated power

    Raises:
        HTTPException: If power is out of bounds
    """
    if not min_dbm <= power_dbm <= max_dbm:
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} must be between {min_dbm:.0f} dBm and {max_dbm:.0f} dBm",
        )
    return power_dbm


def validate_snr_db(snr_db: float, min_db: float = -20.0, max_db: float = 100.0) -> float:
    """
    Validate SNR in dB.

    Args:
        snr_db: SNR in dB to validate
        min_db: Minimum allowed SNR
        max_db: Maximum allowed SNR

    Returns:
        Validated SNR

    Raises:
        HTTPException: If SNR is out of bounds
    """
    if not min_db <= snr_db <= max_db:
        raise HTTPException(
            status_code=400, detail=f"SNR must be between {min_db:.0f} dB and {max_db:.0f} dB"
        )
    return snr_db
