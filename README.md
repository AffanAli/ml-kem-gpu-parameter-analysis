# ml-kem-gpu-parameter-analysis

## Overview

This repository contains a GPU-oriented implementation of the NIST standardized ML-KEM (formerly CRYSTALS-Kyber) algorithm using Python and PyTorch.

The primary objective of this project is to investigate the feasibility of implementing ML-KEM cryptographic operations on a GPU and to evaluate the impact of selected cryptographic parameters on performance and behavior.

This work is intended for research and educational purposes and forms the implementation foundation for a postgraduate dissertation focused on GPU-based post-quantum cryptography.


## Project Objectives

The repository focuses on three major areas:

### 1. GPU-Based ML-KEM Implementation

Implement core ML-KEM operations using PyTorch tensors executed on NVIDIA GPUs.

Target modules include:

- Modular reduction
- Polynomial arithmetic
- Number Theoretic Transform (NTT)
- Inverse NTT
- Polynomial vector operations
- Noise generation (CBD)
- IND-CPA encryption
- ML-KEM Key Generation
- Encapsulation
- Decapsulation


### 2. Parameter Analysis

Investigate the impact of selected parameters on algorithm behavior.

Examples include:

- Modulus parameter `q`
- Noise distribution parameters (`η`)
- Polynomial coefficient distributions
- Runtime behavior under varying parameter settings


### 3. Experimental Evaluation

Perform experiments comparing:

- CPU reference implementation
- GPU implementation

Metrics include:

- Execution time
- Throughput
- Memory usage
- Correctness validation

---

## GPU Strategy

This project intentionally focuses on GPU acceleration using **PyTorch CUDA tensors**.

Example:

```python
poly = torch.tensor(coefficients, device="cuda")
```

Operations performed on tensors located on the CUDA device are executed on the GPU.

### Not in Scope

This project does **not** focus on:

- Custom CUDA kernel development
- CUDA C/C++ programming
- GPU-specific optimization techniques

The goal is to investigate GPU-based implementation using high-level PyTorch primitives rather than low-level CUDA programming.

---

## Reference Implementations

The following repositories are used as correctness references:

- PQClean
- CRYSTALS-Kyber

These repositories remain unmodified and are used for:

- Validation
- Debugging
- Algorithm understanding
- Correctness comparison

---

## Repository Structure

```text
ml-kem-gpu-parameter-analysis/
│
├── README.md
├── requirements.txt
│
├── reference/
│   ├── pqclean/              # PQClean reference implementation
│   └── kyber/                # Original Kyber reference code
│
├── src/
│   └── mlkem_gpu/
│       ├── __init__.py
│       ├── params.py         # ML-KEM parameter definitions
│       ├── gpu_utils.py      # Shared GPU utility functions
│       │
│       ├── reduce_gpu.py     # Modular reduction operations
│       ├── cbd_gpu.py        # Centered Binomial Distribution sampling
│       ├── ntt_gpu.py        # Forward and inverse NTT
│       ├── poly_gpu.py       # Polynomial operations
│       ├── polyvec_gpu.py    # Polynomial vector operations
│       ├── indcpa_gpu.py     # IND-CPA encryption layer
│       └── kem_gpu.py        # ML-KEM API implementation
│
├── experiments/
│   ├── benchmark_keygen.py   # Key generation benchmarks
│   ├── benchmark_encaps.py   # Encapsulation benchmarks
│   ├── benchmark_decaps.py   # Decapsulation benchmarks
│   ├── q_analysis.py         # Modulus q experiments
│   ├── noise_analysis.py     # Noise parameter experiments
│   └── memory_analysis.py    # GPU memory analysis
│
├── validation/
│   ├── compare_ntt.py        # NTT correctness validation
│   ├── compare_poly.py       # Polynomial validation
│   ├── compare_cbd.py        # CBD validation
│   └── compare_kem.py        # End-to-end validation
│
├── tests/
│   ├── test_ntt_gpu.py
│   ├── test_cbd_gpu.py
│   ├── test_poly_gpu.py
│   └── test_kem_gpu.py
│
├── notebooks/
│   ├── q_distribution.ipynb          # q parameter analysis
│   ├── noise_distribution.ipynb      # Noise distribution analysis
│   └── benchmark_visualization.ipynb # Performance visualization
│
├── reports/
│   ├── figures/              # Dissertation figures
│   ├── tables/               # Dissertation tables
│   └── dissertation_assets/  # Supporting dissertation material
│
└── results/
    ├── timings/              # Benchmark outputs
    ├── q_analysis/           # q experiment results
    └── noise_analysis/       # Noise experiment results
```


## Development Environment

### Development Machine

- macOS
- Apple M3
- Python 3.11

### GPU Testing Machine

- Windows
- NVIDIA GTX 1660 Ti
- CUDA 11.8
- PyTorch CUDA

### Setup on Windows Machine
```
$ conda create -n mlkem_gpu python=3.11

$ conda activate mlkem_gpu

$ conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

$ pip install -r requirements.txt
```

## Current Status

- [ ] Repository setup
- [ ] Parameter module
- [ ] Modular reduction
- [ ] NTT implementation
- [ ] CBD implementation
- [ ] IND-CPA implementation
- [ ] ML-KEM implementation
- [ ] Parameter analysis
- [ ] Benchmarking
- [ ] Dissertation evaluation