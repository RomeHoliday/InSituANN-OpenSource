# InSituANN Query-Time Source Release

This repository contains the query-time search core for InSituANN.  It focuses
on the serving path: GPU IVF routing, on-device top-k selection, CPU exact
verification over host-resident vectors, and residual-PQ candidate pruning.

Large datasets, generated index artifacts, offline builders, auxiliary
experimental runners, third-party comparison code, plotting code, and paper
result directories are not included.

## What Is Included

- `cuda/`: CUDA/C++ source for the query-time execution path, including IVF
  routing, GPU top-k selection, Exact-CPU fine search, CPU residual-PQ kernels,
  and shared GPU utilities.
- `configs/`: reference configuration templates for the query-time parameters
  used in the paper, such as `nlist`, `nprobe`, batch sizes, CPU threads, and
  residual-PQ code sizes.

## Build

The original CMake project is preserved.  A typical build is:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build -j
```

For older GPUs, replace `120` with the correct CUDA architecture.  The code was
developed with CUDA 13.0 and C++17.

## Scope

This source release is intended to expose the query-time implementation, not to
serve as a complete artifact package for every experiment in the paper.  The
100M/1B evaluation requires external datasets and generated artifacts such as
centroids, assignment arrays, reordered vectors, PQ codebooks, and residual-PQ
codes.  Those artifacts are intentionally excluded because of their size.

Auxiliary experimental runners and the offline accelerated construction
pipeline are not part of this public source package.

## License

No license file is included in this release yet.  Add the intended
project license before publishing the repository publicly.
