# RF Forensics Pipeline - Engineering Benchmark Report

**Generated:** 2025-12-02T05:46:47.916105

## System Configuration

| Parameter | Value |
|-----------|-------|
| Platform | Linux (Docker + NVIDIA Container Toolkit) |
| GPU | NVIDIA CUDA GPU |
| GPU Memory | N/A GB |
| CUDA Version | N/A |
| SDR | uSDR PCIe (LMS7002M-based) |
| Max Theoretical Rate | 50 MSPS |

## Executive Summary

| Metric | Value |
|--------|-------|
| Tests Run | 4 |
| Tests Passed | 2 |
| Tests Marginal | 0 |
| Tests Failed | 2 |
| **Max Sustainable Rate** | **40.0 MSPS** |
| **Recommended Operating Rate** | **36.0 MSPS** |

### Key Findings

- Pipeline sustains 40.0 MSPS without significant sample loss
- Sample drops observed at 30.0 MSPS and above
- Thermal performance nominal (max 47.0°C)
- Processing latency within spec (P99 5.58ms)

## Test Results Summary

| Rate (MSPS) | Status | Drop Rate | Throughput | Latency P95 | Buffer Fill | Temp |
|-------------|--------|-----------|------------|-------------|-------------|------|
| 30.0 | ❌ FAIL | 4.22% ± 5.03% | 54.8 MSPS | 3.54 ms | 50.0% | 46.9°C |
| 35.0 | ✅ PASS | 0.00% ± 0.00% | 53.8 MSPS | 2.50 ms | 62.5% | 46.9°C |
| 40.0 | ✅ PASS | 0.00% ± 0.00% | 53.1 MSPS | 2.35 ms | 62.5% | 47.0°C |
| 45.0 | ❌ FAIL | 8.72% ± 4.92% | 52.7 MSPS | 2.34 ms | 62.5% | 47.0°C |

## Detailed Results

### 30.0 MSPS Test

**Status:** FAIL
  
**Reason:** Average drop rate 4.22% exceeds 2% threshold

#### Sample Processing
| Metric | Value |
|--------|-------|
| Total Received | 71.20B samples |
| Total Dropped | 4,334,944,256 samples |
| Actual Input Rate | 30.01 MSPS |
| Effective Throughput | 28.79 MSPS |
| Total Detections | 566,431 |

#### Drop Rate Analysis
| Metric | Value |
|--------|-------|
| Mean Drop Rate | 4.2244% |
| Std Dev | 5.0288% |
| Min | 0.0000% |
| Max | 11.6667% |
| Drop-Free Intervals | 24/43 (55.8%) |

#### GPU Processing Performance
| Metric | Value |
|--------|-------|
| Mean Throughput | 54.81 MSPS |
| Std Dev | 0.29 MSPS |
| Min | 54.35 MSPS |
| Max | 55.31 MSPS |
| Headroom Ratio | 1.83x |

#### Latency Distribution
| Percentile | Value |
|------------|-------|
| Mean | 2.350 ms |
| P50 | 2.267 ms |
| P95 | 3.545 ms |
| P99 | 3.680 ms |
| Max | 3.680 ms |

#### Resource Utilization
| Resource | Mean | Max |
|----------|------|-----|
| Buffer Fill | 18.9% | 50.0% |
| GPU Memory | 1.040 GB | 1.189 GB |
| SDR Temperature | 46.9°C | 46.9°C |

### 35.0 MSPS Test

**Status:** PASS

#### Sample Processing
| Metric | Value |
|--------|-------|
| Total Received | 73.11B samples |
| Total Dropped | 4,334,944,256 samples |
| Actual Input Rate | 35.0 MSPS |
| Effective Throughput | 35.0 MSPS |
| Total Detections | 575,869 |

#### Drop Rate Analysis
| Metric | Value |
|--------|-------|
| Mean Drop Rate | 0.0000% |
| Std Dev | 0.0000% |
| Min | 0.0000% |
| Max | 0.0000% |
| Drop-Free Intervals | 44/44 (100.0%) |

#### GPU Processing Performance
| Metric | Value |
|--------|-------|
| Mean Throughput | 53.83 MSPS |
| Std Dev | 0.20 MSPS |
| Min | 53.50 MSPS |
| Max | 54.17 MSPS |
| Headroom Ratio | 1.54x |

#### Latency Distribution
| Percentile | Value |
|------------|-------|
| Mean | 2.227 ms |
| P50 | 2.111 ms |
| P95 | 2.503 ms |
| P99 | 5.233 ms |
| Max | 5.233 ms |

#### Resource Utilization
| Resource | Mean | Max |
|----------|------|-----|
| Buffer Fill | 18.5% | 62.5% |
| GPU Memory | 1.000 GB | 1.078 GB |
| SDR Temperature | 46.8°C | 46.9°C |

### 40.0 MSPS Test

**Status:** PASS

#### Sample Processing
| Metric | Value |
|--------|-------|
| Total Received | 75.28B samples |
| Total Dropped | 4,334,944,256 samples |
| Actual Input Rate | 40.0 MSPS |
| Effective Throughput | 40.0 MSPS |
| Total Detections | 593,251 |

#### Drop Rate Analysis
| Metric | Value |
|--------|-------|
| Mean Drop Rate | 0.0000% |
| Std Dev | 0.0000% |
| Min | 0.0000% |
| Max | 0.0000% |
| Drop-Free Intervals | 44/44 (100.0%) |

#### GPU Processing Performance
| Metric | Value |
|--------|-------|
| Mean Throughput | 53.14 MSPS |
| Std Dev | 0.13 MSPS |
| Min | 52.92 MSPS |
| Max | 53.37 MSPS |
| Headroom Ratio | 1.33x |

#### Latency Distribution
| Percentile | Value |
|------------|-------|
| Mean | 2.043 ms |
| P50 | 1.999 ms |
| P95 | 2.349 ms |
| P99 | 2.525 ms |
| Max | 2.525 ms |

#### Resource Utilization
| Resource | Mean | Max |
|----------|------|-----|
| Buffer Fill | 15.6% | 62.5% |
| GPU Memory | 0.969 GB | 1.038 GB |
| SDR Temperature | 46.9°C | 47.0°C |

### 45.0 MSPS Test

**Status:** FAIL
  
**Reason:** Average drop rate 8.72% exceeds 2% threshold

#### Sample Processing
| Metric | Value |
|--------|-------|
| Total Received | 77.73B samples |
| Total Dropped | 4,514,906,112 samples |
| Actual Input Rate | 45.0 MSPS |
| Effective Throughput | 40.98 MSPS |
| Total Detections | 628,542 |

#### Drop Rate Analysis
| Metric | Value |
|--------|-------|
| Mean Drop Rate | 8.7184% |
| Std Dev | 4.9154% |
| Min | 0.0000% |
| Max | 12.3249% |
| Drop-Free Intervals | 10/44 (22.7%) |

#### GPU Processing Performance
| Metric | Value |
|--------|-------|
| Mean Throughput | 52.66 MSPS |
| Std Dev | 0.12 MSPS |
| Min | 52.46 MSPS |
| Max | 52.84 MSPS |
| Headroom Ratio | 1.17x |

#### Latency Distribution
| Percentile | Value |
|------------|-------|
| Mean | 2.170 ms |
| P50 | 2.055 ms |
| P95 | 2.339 ms |
| P99 | 5.580 ms |
| Max | 5.580 ms |

#### Resource Utilization
| Resource | Mean | Max |
|----------|------|-----|
| Buffer Fill | 22.2% | 62.5% |
| GPU Memory | 1.001 GB | 1.044 GB |
| SDR Temperature | 47.0°C | 47.0°C |
