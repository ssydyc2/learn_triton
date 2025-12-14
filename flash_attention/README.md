
# Flash Attention Benchmarks

This repository contains benchmarks comparing three different implementations of Self-Attention.

## Implementations

### 1. Naive PyTorch
The standard implementation using PyTorch operations:
```python
S = Q @ K.T
P = softmax(S)
O = P @ V
```
This approach materializes both the score matrix `S` ($N \times N$) and the attention matrix `P` ($N \times N$) in HBM (Global Memory). This leads to $O(N^2)$ memory usage, which causes Out-Of-Memory (OOM) errors for long sequences and is slow due to excessive memory I/O.

### 2. Triton With Fusion Only
A custom Triton implementation that partially fuses operations.
1.  **Fused Softmax Kernel**: Computes `P = softmax(Q @ K.T)` in a single kernel. It computes the matrix multiplication block-by-block and applies online softmax fusion. However, it **writes the final attention matrix P to HBM**.
2.  **Matmul Kernel**: Reads `P` from HBM and computes `O = P @ V`.

This approach avoids materializing the raw score matrix `S`, saving some memory compared to Naive PyTorch, but it still materializes the attention matrix `P`. Thus, it still suffers from $O(N^2)$ memory complexity and I/O bottlenecks, making it slower than Flash Attention for large sequences.

### 3. Flash Attention v2
The state-of-the-art implementation that fully fuses the attention operation into a single kernel.
It uses tiling to load blocks of Q, K, V into on-chip SRAM, computes local attention scores, and aggregates the results using the online softmax trick without **ever** writing the full $N \times N$ matrix to HBM.
-   **Memory**: $O(N)$ (linear in sequence length due to output).
-   **Speed**: Significantly faster due to reduced HBM access.

## Benchmark Results

The following chart shows the runtime (ms) vs Sequence Length for the three methods.

![Benchmark Results](benchmark_results.png)

As seen in the results:
-   **Flash Attention v2** is the fastest and scales best with sequence length.
-   **Triton With Fusion Only** performs better than Naive PyTorch (likely due to fusion of score+softmax) but degrades at longer sequences compared to Flash Attention because it is memory-bound by the $O(N^2)$ write/read of the Attention Matrix.
-   **naive pytorch** is the slowest due to full materialization of all intermediate $n \times n$ matrices.
