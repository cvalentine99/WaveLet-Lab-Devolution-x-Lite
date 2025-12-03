# GPUDirect DMA Primer for uSDR (usdr-lib)

Audience: another LLM/engineer designing GPUDirect-style DMA from a uSDR PCIe board (wavelet-lab/usdr-lib) into GPU memory.

## Context & Current Stack
- Hardware: RTX 4090 on CPU root port (Gen4 x16 capable), uSDR PCIe card (usdr-lib) currently limited by slot to Gen2/x2-level bandwidth (~1 GB/s usable).
- Topology: single NUMA node; avoid PCH lanes. GPUDirect P2P requires same CPU root complex and non-isolating ACS/IOMMU.
- Software: usdr-lib kernel driver (usdr/usdr-dsp) + userland API; today DMA targets host memory. NVIDIA driver supports peer memory via `nvidia-peermem`.
- Goal: DMA SDR buffers directly into GPU VRAM (BAR1) to skip host copies.

## High-Level Design Targets
1) **Export GPU memory as dma-buf** (userland):
   - Use CUDA driver API (`cuMemCreate` + `cuMemExportToShareableHandle(fd, handle, CU_MEM_HANDLE_TYPE_DMA_BUF, 0)`) to create an exportable GPU buffer.
   - Pass the dma-buf fd to usdr-lib via a new ioctl (e.g., `USDR_IOC_REGISTER_DMABUF`).
2) **Map GPU dma-buf in kernel driver**:
   - In the usdr kernel driver, accept fd, call `dma_buf_get()`, `dma_buf_attach()`, `dma_buf_map_attachment()` to obtain `sg_table` of GPU pages; `nvidia-peermem` backs this.
   - Program the uSDR DMA engine (XDMA-style) with physical addresses from `sg_table`.
   - Unmap/detach/put on teardown.
3) **Limits/guards**:
   - Enforce max buffer length and scatter-gather depth.
   - Single active mapping per RX/TX queue to simplify.
   - Reject if ACS/IOMMU blocks P2P (detect errors on attach/map).

## Steps to Implement
1) Identify the uSDR DMA path in usdr-lib (kernel): locate where host buffers are allocated/mapped and where DMA descriptors are programmed.
2) Add ioctl/user API to register a dma-buf fd for RX/TX.
3) Implement dma-buf attach/map/unmap lifecycle with robust error handling.
4) Expose a userland helper to allocate/export GPU buffers and pass fds into the driver.
5) Add a small test: issue a short DMA into GPU buffer, verify on GPU (CUDA kernel) and profile to confirm no host memcpy.

## Validation Checklist
- `nvidia-smi topo -m`: uSDR bus shows PIX/PHB to GPU (not SYS).
- `lsmod | grep nvidia_peermem`: loaded; no dmesg faults on dma-buf attach.
- `lspci -vv -s <uSDR>`: link width/speed acceptable; ACS not forcing isolation.
- Functional test passes; throughput near link max; no host-side copies observed.

## Risks / Fallback
- If dma-buf attach fails (ACS/IOMMU, driver limitations), fall back to pinned host buffers + async CUDA copies.
- Gen2/x2 link caps throughput; consider moving the card to a CPU root slot with more lanes if available.

## Open Questions for the Designer
- Exact uSDR DMA engine interface (XDMA? custom): which registers/descriptors to program with physical addrs?
- Maximum SG entries and alignment constraints?
- Do we need bidirectional (TX) GPUDirect or RX-only?
- Can we relax ACS in BIOS for the target slot if P2P is blocked?

## Quick Commands (once driver built)
- Topology: `nvidia-smi topo -m`
- Link info: `lspci -vv -s <uSDR_bus>`
- IOMMU groups: `find /sys/kernel/iommu_groups -type l | sort`
- Check peer module: `lsmod | grep nvidia_peermem`
