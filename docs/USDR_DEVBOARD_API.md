# uSDR DevBoard API Reference

**Library:** `libusdr.so` (wavelet-lab/usdr-lib)
**Version:** 0.9.9
**Source:** https://github.com/wavelet-lab/usdr-lib

---

## Table of Contents

1. [Device Management](#1-device-management)
2. [Parameter Access (VFS)](#2-parameter-access-vfs)
3. [Stream Management](#3-stream-management)
4. [SDR Control Functions](#4-sdr-control-functions)
5. [VFS Paths Reference](#5-vfs-paths-reference)
6. [Python Bindings](#6-python-bindings)
7. [Usage Examples](#7-usage-examples)

---

## 1. Device Management

### Discovery

```c
int usdr_dmd_discovery(const char* filter_string, unsigned max_buf, char* devlist);
```

**Parameters:**
- `filter_string` - Filter for device search (empty string `""` for all devices)
- `max_buf` - Maximum buffer size for device list
- `devlist` - Output buffer for discovered devices

**Returns:** Number of devices found, or negative error code

**Device list format:** Devices separated by `\v` (vertical tab), fields by `\t` (tab)

**Example:**
```c
char devlist[4096];
int count = usdr_dmd_discovery("", 4096, devlist);
// devlist: "bus=pci,device=usdr0\vbus=usb,device=1234"
```

---

### Create Device

```c
int usdr_dmd_create_string(const char* connection_string, pdm_dev_t* odev);
```

**Parameters:**
- `connection_string` - Device connection string from discovery (empty for auto-select)
- `odev` - Output pointer to device handle

**Returns:** 0 on success, negative error code on failure

**Connection string formats:**
- `""` - Auto-select first available device
- `bus=pci,device=usdr0` - PCIe device
- `bus=usb,device=1234` - USB device
- `usb://0` - USB device by index

---

### Close Device

```c
int usdr_dmd_close(pdm_dev_t dev);
```

**Parameters:**
- `dev` - Device handle from `usdr_dmd_create_string`

**Returns:** 0 on success

---

## 2. Parameter Access (VFS)

The uSDR uses a Virtual File System (VFS) for parameter access. Parameters are accessed via paths like `/dm/sdr/0/rx/frequency`.

### Get Parameter (uint64)

```c
int usdr_dme_get_uint(pdm_dev_t dev, const char* path, uint64_t* oval);
```

### Get Parameter (uint32)

```c
int usdr_dme_get_u32(pdm_dev_t dev, const char* path, uint32_t* oval);
```

### Set Parameter (uint64)

```c
int usdr_dme_set_uint(pdm_dev_t dev, const char* path, uint64_t val);
```

### Set Parameter (string)

```c
int usdr_dme_set_string(pdm_dev_t dev, const char* path, const char* val);
```

### Get Parameter (string)

```c
int usdr_dme_get_string(pdm_dev_t dev, const char* path, const char** oval);
```

---

## 3. Stream Management

### Create Stream

```c
int usdr_dms_create(pdm_dev_t device,
                    const char* sobj,
                    const char* dformat,
                    logical_ch_msk_t channels,
                    unsigned pktsyms,
                    pusdr_dms_t* outu);
```

**Parameters:**
- `device` - Device handle
- `sobj` - Stream object path:
  - `/ll/srx/0` - Low-latency RX stream, channel 0
  - `/ll/stx/0` - Low-latency TX stream, channel 0
- `dformat` - Data format: `HOST_FORMAT@WIRE_FORMAT`
  - `cf32@ci16` - Host: complex float32, Wire: complex int16
  - `ci16@ci16` - Host: complex int16, Wire: complex int16
- `channels` - Bitmask of logical channels (e.g., `0x1` for channel 0)
- `pktsyms` - Packet size in symbols (samples)
- `outu` - Output stream handle

**Returns:** 0 on success

---

### Destroy Stream

```c
int usdr_dms_destroy(pusdr_dms_t stream);
```

---

### Stream Sync

```c
int usdr_dms_sync(pdm_dev_t device,
                  const char* synctype,
                  unsigned scount,
                  pusdr_dms_t* pstream);
```

**Sync types:**
- `"none"` - No syncing between streams
- `"off"` - Disable sync (call before start)
- `"all"` - Sync between all active streams
- `"extall"` - Sync on external event (1PPS)

---

### Stream Operation (Start/Stop)

```c
int usdr_dms_op(pusdr_dms_t stream, unsigned command, dm_time_t tm);
```

**Commands:**
| Command | Value | Description |
|---------|-------|-------------|
| `USDR_DMS_START` | 0 | Start streaming immediately |
| `USDR_DMS_STOP` | 1 | Stop streaming immediately |
| `USDR_DMS_START_AT` | 2 | Start at specified time |
| `USDR_DMS_STOP_AT` | 3 | Stop at specified time |

---

### Receive Samples

```c
int usdr_dms_recv(pusdr_dms_t stream,
                  void** stream_buffs,
                  unsigned timeout_ms,
                  usdr_dms_recv_nfo_t* nfo);
```

**Parameters:**
- `stream` - Stream handle
- `stream_buffs` - Array of buffer pointers (one per channel)
- `timeout_ms` - Timeout in milliseconds
- `nfo` - Output receive info structure

**Receive Info Structure:**
```c
struct usdr_dms_recv_nfo {
    dm_time_t fsymtime;   // First symbol timestamp (sample clock ticks)
    unsigned totsyms;     // Total valid samples received
    unsigned totlost;     // Samples lost (overflow)
    unsigned max_parts;   // Max burst parts
    uint64_t extra;       // Extra info
};
```

---

### Send Samples

```c
int usdr_dms_send(pusdr_dms_t stream,
                  const void** stream_buffs,
                  unsigned samples,
                  dm_time_t timestamp,
                  unsigned timeout);
```

---

## 4. SDR Control Functions

High-level SDR control (alternative to VFS):

```c
// Set bandwidth
int usdr_dmsdr_set_bandwidth(pdm_dev_t dev, const char* entity,
                             usdr_frequency_t start, usdr_frequency_t stop);

// Set frequency
int usdr_dmsdr_set_frequency(pdm_dev_t dev, const char* entity,
                             usdr_frequency_t freq);

// Set gain
int usdr_dmsdr_set_gain(pdm_dev_t dev, const char* entity, unsigned gain);

// Set path (by index)
int usdr_dmsdr_set_path(pdm_dev_t dev, const char* entity, unsigned path);

// Set path (by string)
int usdr_dmsdr_set_path_str(pdm_dev_t dev, const char* entity, const char* p);
```

---

## 5. VFS Paths Reference

### RX Parameters

| Path | Type | Description | Range |
|------|------|-------------|-------|
| `/dm/sdr/0/rx/freqency` | uint64 | Center frequency (Hz) | 70M - 6G |
| `/dm/sdr/0/rx/frequency` | uint64 | Center frequency (alt spelling) | 70M - 6G |
| `/dm/sdr/0/rx/bandwidth` | uint64 | Bandwidth (Hz) | 100K - 56M |
| `/dm/sdr/0/rx/path` | string | RX antenna path | LNAH, LNAL, LNAW |
| `/dm/sdr/0/rx/gain/lna` | uint64 | LNA gain (dB) | 0 - 30 |
| `/dm/sdr/0/rx/gain/vga` | uint64 | TIA/VGA gain (dB) | 0, 3, 9, 12 |
| `/dm/sdr/0/rx/gain/pga` | uint64 | PGA gain (dB) | 0 - 32 |

### Sample Rate

| Path | Type | Description |
|------|------|-------------|
| `/dm/rate/rxtxadcdac` | uint64[4] | Sample rates: [RX, TX, ADC, DAC] |

### Hardware Sensors

| Path | Type | Description |
|------|------|-------------|
| `/dm/sensor/temp` | uint32 | Temperature (raw: divide by 256 for °C) |

### DevBoard Frontend Controls

| Path | VFS Name | Type | Description |
|------|----------|------|-------------|
| `/dm/sdr/0/fe/attn` | attn_ | uint64 | RX attenuator (0-18 dB) |
| `/dm/sdr/0/fe/lna` | lna_ | uint64 | RX LNA enable (+19.5dB) |
| `/dm/sdr/0/fe/pa` | pa_ | uint64 | TX PA enable (+19.5dB) |
| `/dm/sdr/0/dac_vctcxo` | dac_ | uint64 | VCTCXO tuning (0-65535) |
| `/dm/sdr/0/fe/path` | path_ | string | Duplexer band selection |
| `/dm/sdr/0/fe/gps` | gps_ | uint64 | GPS module enable |
| `/dm/sdr/0/fe/osc` | osc_ | uint64 | Reference oscillator enable |
| `/dm/sdr/0/fe/lb` | lb_ | uint64 | RX->TX loopback enable |
| `/dm/sdr/0/fe/uart` | uart_ | uint64 | UART interface enable |

### Duplexer Bands (path_)

| Band | Category | Freq Range | Description |
|------|----------|------------|-------------|
| band2 | cellular | 1850-1990 MHz | PCS / GSM 1900 |
| band3 | cellular | 1710-1880 MHz | DCS / GSM 1800 |
| band5 | cellular | 824-894 MHz | GSM 850 |
| band7 | cellular | 2500-2690 MHz | IMT-E / LTE Band 7 |
| band8 | cellular | 880-960 MHz | GSM 900 |
| rxlpf1200 | rx_only | 0-1200 MHz | RX only, LPF |
| rxlpf2100 | rx_only | 0-2100 MHz | RX only, LPF |
| trx0_400 | tdd | 0-400 MHz | TDD mode |
| trx400_1200 | tdd | 400-1200 MHz | TDD mode |

---

## 6. Python Bindings

Our Python wrapper (`sdr/usdr_driver.py`) provides:

```python
from rf_forensics.sdr.usdr_driver import USDRDriver, USDRConfig, USDRGain

# Create driver
driver = USDRDriver()

# Check library availability
if driver.is_available:
    # Discover devices
    devices = driver.discover()

    # Connect
    driver.connect(devices[0].id)

    # Configure
    config = USDRConfig(
        center_freq_hz=915_000_000,
        sample_rate_hz=10_000_000,
        bandwidth_hz=10_000_000,
        gain=USDRGain(lna_db=15, tia_db=9, pga_db=12),
        rx_path="LNAL"
    )
    driver.configure(config)

    # Start streaming
    def callback(samples, timestamp):
        # samples: np.ndarray[complex64]
        print(f"Got {len(samples)} samples")

    driver.start_streaming(callback)

    # Stop
    driver.stop_streaming()
    driver.disconnect()
```

### Using SDRManager (Recommended)

```python
from rf_forensics.sdr.manager import get_sdr_manager
from rf_forensics.sdr.usdr_driver import USDRConfig, USDRGain

# Get singleton manager
manager = get_sdr_manager()

# Discover and connect
devices = manager.discover()
manager.connect(devices[0].id)

# Configure
manager.configure(USDRConfig(
    center_freq_hz=915_000_000,
    sample_rate_hz=10_000_000,
))

# Start streaming with metrics
manager.start_streaming(my_callback)

# Monitor
metrics = manager.get_metrics()
print(f"Temperature: {metrics.temperature_c}°C")
print(f"Overflows: {metrics.total_overflows}")

# Stop
manager.stop_streaming()
manager.disconnect()
```

---

## 7. Usage Examples

### C Example: Basic RX

```c
#include <dm_dev.h>
#include <dm_stream.h>
#include <stdio.h>
#include <complex.h>

int main() {
    pdm_dev_t dev;
    pusdr_dms_t stream;

    // Discover devices
    char devlist[4096];
    int count = usdr_dmd_discovery("", 4096, devlist);
    printf("Found %d devices\n", count);

    // Create device (auto-select)
    int res = usdr_dmd_create_string("", &dev);
    if (res < 0) {
        printf("Failed to create device: %d\n", res);
        return 1;
    }

    // Set frequency to 915 MHz
    usdr_dme_set_uint(dev, "/dm/sdr/0/rx/freqency", 915000000);

    // Set sample rate to 10 MSPS
    uint64_t rates[4] = {10000000, 10000000, 0, 0};
    usdr_dme_set_uint(dev, "/dm/rate/rxtxadcdac", (uint64_t)rates);

    // Set gains
    usdr_dme_set_uint(dev, "/dm/sdr/0/rx/gain/lna", 15);
    usdr_dme_set_uint(dev, "/dm/sdr/0/rx/gain/vga", 9);
    usdr_dme_set_uint(dev, "/dm/sdr/0/rx/gain/pga", 12);

    // Set RX path
    usdr_dme_set_string(dev, "/dm/sdr/0/rx/path", "LNAL");

    // Create RX stream
    res = usdr_dms_create(dev, "/ll/srx/0", "cf32@ci16", 0x1, 131072, &stream);
    if (res < 0) {
        printf("Failed to create stream: %d\n", res);
        usdr_dmd_close(dev);
        return 1;
    }

    // Sync and start
    pusdr_dms_t streams[1] = {stream};
    usdr_dms_sync(dev, "off", 1, streams);
    usdr_dms_op(stream, USDR_DMS_START, 0);
    usdr_dms_sync(dev, "all", 1, streams);

    // Receive samples
    float complex buffer[131072];
    void* buffers[1] = {buffer};
    usdr_dms_recv_nfo_t nfo;

    for (int i = 0; i < 100; i++) {
        res = usdr_dms_recv(stream, buffers, 100, &nfo);
        if (res >= 0 && nfo.totsyms > 0) {
            printf("Received %u samples, lost %u\n", nfo.totsyms, nfo.totlost);
        }
    }

    // Stop and cleanup
    usdr_dms_op(stream, USDR_DMS_STOP, 0);
    usdr_dms_destroy(stream);
    usdr_dmd_close(dev);

    return 0;
}
```

### Compile

```bash
gcc -o rx_example rx_example.c -lusdr -lm
```

---

## Error Codes

| Code | Description |
|------|-------------|
| 0 | Success |
| -ENOENT | Path not found |
| -EINVAL | Invalid argument |
| -EBUSY | Device busy |
| -ETIMEDOUT | Operation timed out |
| -EIO | I/O error |

---

## References

- **usdr-lib GitHub:** https://github.com/wavelet-lab/usdr-lib
- **DevBoard Docs:** https://docs.wsdr.io/hardware/devboard.html
- **SoapySDR Plugin:** `soapysdr-module-usdr`
