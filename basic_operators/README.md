# Triton Basic Operators

This folder contains introductory Triton kernel examples for learning GPU programming fundamentals.

## Notebooks

### vector_add.ipynb
Vector addition demonstrating Triton basics:
- Kernel definition with `@triton.jit`
- Pointer arithmetic and memory access
- Block-based parallelization
- Masked load/store operations

### fused_softmax.ipynb
Memory-bound kernel optimization example covering:
- **Kernel fusion** - Combining multiple operations to reduce memory traffic
- **Occupancy estimation** - Balancing SRAM vs registers usage
- **Software pipelining** - Using `num_stages` to hide memory latency (more stages = more SRAM)
- **Warp configuration** - Trade-off between parallelization and register pressure
- **Persistent kernels** - Processing multiple rows per program for better SM utilization

Key concepts demonstrated:
- Computing optimal grid size based on hardware properties
- Using `kernel.warmup()` for pre-compilation and resource inspection
- Benchmarking against PyTorch native implementations