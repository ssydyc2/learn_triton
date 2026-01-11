# Basic Operators

This folder contains introductory Triton kernel examples demonstrating fundamental GPU programming patterns.

## Triton Kernel Pattern

All kernels follow this structure:
```python
@triton.jit
def kernel(input_ptrs, output_ptrs, strides, ..., BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)                           # Program identification
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE) # Offset calculation
    data = tl.load(ptr + offsets, mask=offsets < n)       # Masked load
    result = ...                                           # Computation
    tl.store(out_ptr + offsets, result, mask=mask)        # Masked store
```

## Files

- `vector_add.ipynb` - Element-wise vector addition showing basic load/store and masking
