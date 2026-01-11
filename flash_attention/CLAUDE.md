# Flash Attention

This folder contains Flash Attention implementations comparing three approaches to self-attention.

## Implementations

- **Naive PyTorch**: Baseline with O(N²) memory, materializes full score and attention matrices
- **Triton With Fusion Only**: Two-kernel approach that fuses softmax but still writes attention matrix to HBM
- **Flash Attention v2**: Fully-fused single kernel using tiling and online softmax, O(N) memory

## Key Constants

- `TRITON_BLOCK_M = 128` - Query block size
- `TRITON_BLOCK_N = 64` - Key block size
- `TRITON_BLOCK_D = 64` - Dimension block size

## Validation

Implementations are validated against float32 reference:
```python
validate_attention_impls()  # Compares all implementations
```

## Benchmarking

Uses `triton.testing.Benchmark` decorator with `triton.testing.do_bench()`:
```python
attention_benchmark.run(print_data=True, show_plots=True)
```

Sequence lengths tested: 128 to 32768 (powers of 2)

## Files

- `flash_attention.ipynb` - All three implementations with benchmark infrastructure
- `benchmark_results.png` - Performance comparison chart
