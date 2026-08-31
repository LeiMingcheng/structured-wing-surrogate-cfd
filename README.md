# Structured Wing Surrogate CFD

Structured Wing Surrogate CFD is the three-dimensional extension layer for
whole-case surrogate modeling on fixed-topology, multi-block wing CFD grids.
It consumes structured exports from Parametric Swept-Wing CFD and reuses the
training/checkpoint contracts from Surrogate–Newton CFD.

The production path is GPU-first. The maintained baseline is a distributed
whole-case 3D FNO with one prediction spanning all 13 structured blocks. Small
CPU fixtures are used only to verify installation, tensor contracts,
backpropagation, and optimizer wiring in continuous integration.

## Included scope

- HDF5 shard and whole-case multi-block dataset loading;
- six-channel primitive-state normalization, including log-scaled SA state;
- hierarchical global FNO with block-local spectral processing and global
  structured-token mixing;
- single- and multi-GPU supervised training;
- volume-weighted field loss and wall Cp/Cf objectives;
- whole-field, force, surface, restart, and Newton-recovery evaluation;
- differentiable ADFLOW-aligned interior 3D flow and SA residual operators;
- CGNS restart and whole-case payload generation.

The residual implementation covers interior cells. Boundary conditions,
cross-block halo/interface residuals, and residual-loss training are not part
of this snapshot.

## Repository boundaries

This repository does not duplicate the geometry, meshing, CFD orchestration,
or CGNS field exporter from `parametric-swept-wing-cfd`. It also does not copy
the shared two-dimensional model/training framework from
`surrogate-newton-cfd`. Exact staging revisions are recorded in
[`foundation-lock.yaml`](foundation-lock.yaml).

Datasets, CGNS files, checkpoints, validation campaigns, and run directories
are external artifacts and are not tracked by Git.

## Environment

Use an explicit conda environment. The current CFD development environment is
`giao-dm`. Install the two sibling foundations first, then this repository:

```bash
$CONDA_PREFIX/bin/python -m pip install -e ../parametric-swept-wing-cfd
$CONDA_PREFIX/bin/python -m pip install -e ../surrogate-newton-cfd
$CONDA_PREFIX/bin/python -m pip install -e '.[analysis,solver,dev]'
```

For a production GPU run, supply the dataset index and output directory
explicitly:

```bash
DATASET_INDEX=/path/to/structured-dataset/index.csv \
TRAIN_OUTPUT_DIR=/path/to/run \
scripts/train_global_fno_2gpu.sh
```

This two-GPU default is the smallest maintained DDP launch. Increase
`NPROC_PER_NODE` when scaling the same training path to more local GPUs. A
one-step synthetic DDP availability check is provided by
`scripts/smoke_ddp_2gpu.sh`; it does not replace a full 13-block dataset run.

## Release status

This directory is an initial private release-staging snapshot. License and
copyright metadata for the original 3D code must be selected before public
distribution. See [`NOTICE.md`](NOTICE.md).
