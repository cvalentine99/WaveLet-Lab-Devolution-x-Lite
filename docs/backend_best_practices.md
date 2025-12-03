# Backend Best Practices (RF Forensics Engine)

Goals: secure, reliable, high-throughput SDR→GPU processing with clear ownership of hardware resources and predictable APIs.

## 1) Architecture & Ownership
- Single ownership of SDR hardware: pipeline holds the driver; API calls route through the pipeline to mutate SDR config/state. Avoid separate globals (`get_usdr_driver`) that can desync or double-open devices.
- Explicit pipeline lifecycle: IDLE → CONFIGURING → RUNNING/PAUSED → STOPPING → IDLE/ERROR, with state surfaced on APIs and enforced in handlers (no start/stop races).
- Separation of concerns: SDR I/O, buffering, DSP/ML, and API layers are modular but coordinated via clear contracts (events, callbacks with backpressure, thread-safe queues).

## 2) Security
- Require authN/Z on REST and Socket.IO (tokens/mTLS); do not expose `/api/*` or SDR control unauthenticated.
- Restrictive CORS and WS origins (no `*`); bind to localhost by default or run behind a reverse proxy with ACLs.
- Input validation and clamping on SDR params (freq/rate/gain) using driver-reported capabilities; reject or round safely.
- Sanitize recording names/paths, enforce quotas, and rate-limit control endpoints to prevent abuse.
- Minimize error leakage to clients; log detailed errors server-side, return sanitized messages.

## 3) SDR Integration
- Capability discovery: query supported freq/rate/gain/bands at startup; cache and expose via `/api/sdr/capabilities`.
- Hardware health: expose PLL lock, overflow counters, dropped buffers, link speed/width; surface in `/status`.
- Backpressure: monitor ring buffer fill; throttle or pause SDR when near full; resumable flow control.
- Single driver instance: reconfigure in place when config changes; serialize reconfigure/start/stop under a lock.
- No placeholder devices: report “no hardware” explicitly; avoid fake discovery entries.

## 4) Data Path & Performance
- Use pinned host buffers consistently for SDR→GPU transfers; batch copies to minimize sync points.
- Consider GPUDirect RDMA/dma-buf path if hardware/driver supports it; otherwise, overlap SDR DMA with host→GPU async copies.
- Provide bounded queues/ring buffers with metrics (overflow/underflow) and configurable segment sizes.
- Avoid broad try/except that leaks resources; use per-iteration try/finally to always release ring segments.
- Profile and expose latency/throughput metrics; add optional tracing for hot paths (PSD/CFAR/peaks).

## 5) Configuration Management
- Typed schemas (Pydantic) with strict validation and clamping to capabilities; reject unsafe or out-of-range updates.
- Presets: managed centrally; applying a preset triggers only the necessary component rebuilds, with rollback on failure.
- Persisted config: load from a single source (YAML/env), expose read/write endpoints with validation and audit/logging.

## 6) API Design
- Surface pipeline state and health explicitly: `/status` returns state, throughput, latency, buffer fill, and hardware health.
- SDR control endpoints route through the pipeline driver and honor current state; reject illegal transitions (e.g., reconfigure while stopped if not supported).
- Provide idempotent start/stop/pause/resume endpoints with clear return codes; include last_error if in ERROR state.
- Recording endpoints sanitize inputs, enforce quotas, and tie metadata to current SDR config for traceability.
- Socket.IO/WebSocket: auth handshake, origin checks, and rate limits for pushed telemetry.

## 7) Error Handling & Recovery
- Distinguish transient vs fatal errors; auto-restart pipeline on transient SDR errors (with backoff), leave in ERROR for fatal.
- Ensure cleanup on all paths: SDR stop/close, stream manager cleanup, ring buffer segment release, GPU memory pool free.
- Surface actionable error codes/messages to clients; keep stack traces in logs only.

## 8) Observability
- Metrics: throughput (MSps), latency, buffer fill, overflows/underflows, SDR overrun flags, GPU utilization/memory.
- Logs: structured logging with context (session, SDR state, pipeline state); configurable levels.
- Health checks: `/health` lightweight; `/status` richer; include a hardware health sub-block.

## 9) Concurrency & Thread Safety
- Protect shared state (config, pipeline state, metrics) with locks or atomics; avoid cross-thread mutation without guards.
- Callbacks that run on SDR threads should be minimal and non-blocking; hand off to async/event loop via queue.
- Reconfigure under a single lock to avoid races with streaming.

## 10) Deployment & Defaults
- Default to safe bindings (localhost) and minimal privileges; no open CORS/WS in production.
- Use persistence mode on GPU (`nvidia-smi -pm 1`) and disable ASPM on SDR slot if latency-critical.
- Provide sane defaults for SDR (freq/rate/gain) but force explicit confirmation for high-power or atypical settings.

## 11) Testing
- Add unit/functional tests for config validation, state transitions, SDR capability clamping, and error paths.
- Include integration tests with a simulated SDR driver stub (not exposed as a real device type) to validate pipeline behavior without hardware.
- Add soak tests for ring buffer overflow/underflow and pipeline restart sequences.
