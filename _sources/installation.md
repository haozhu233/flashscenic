# Installation

## Basic Install

```bash
pip install flashscenic
```

This installs the core package with dependencies: `numpy`, `torch`, and `regdiffusion`.

## GPU Support

flashscenic requires PyTorch with CUDA support for GPU acceleration. If you don't already have a CUDA-enabled PyTorch installation:

```bash
# Install PyTorch with CUDA (check https://pytorch.org for your CUDA version)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install flashscenic
```

CPU fallback is available by passing `device='cpu'` to all functions.

## Optional Dependencies

For building documentation:

```bash
pip install flashscenic[docs]
```

For running validation tests against pySCENIC/ctxcore:

```bash
pip install pyscenic ctxcore scanpy pandas pyarrow
```

## Development Install

```bash
git clone https://github.com/haozhu233/flashscenic.git
cd flashscenic
pip install -e .
```
