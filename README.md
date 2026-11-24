# Structured Wing Surrogate CFD

Structured Wing Surrogate CFD is the three-dimensional extension layer for
whole-case surrogate modeling on fixed-topology, multi-block wing CFD grids.
It consumes structured exports from Parametric Swept-Wing CFD and reuses the
training, checkpoint, and Newton–Krylov contracts from Surrogate–Newton CFD.

The production path is GPU-first. The maintained baseline is a distributed
hierarchical 3D FNO whose prediction spans all 13 structured blocks of one CFD
case. Small CPU fixtures are used only for portable contract tests; the
included two-GPU smoke test exercises the real DDP training path.

## Scope

Included:

- HDF5 shard and whole-case multi-block dataset loading;
- six-channel primitive-state normalization, including log-scaled SA state;
- a hierarchical global FNO with block-local spectral processing and global
  structured-token mixing;
- single- and multi-GPU supervised training;
- volume-weighted field loss and wall Cp/Cf objectives;
- whole-field, force, surface, restart, and Newton-recovery evaluation;
- differentiable ADFLOW-aligned interior 3D flow and SA residual operators;
- CGNS restart and whole-case payload generation.

Not included:

- geometry generation, volume meshing, CFD orchestration, or CGNS field
  export, which belong to `parametric-swept-wing-cfd`;
- the shared two-dimensional surrogate and Newton runtime, which belongs to
  `surrogate-newton-cfd`;
- datasets, CGNS files, checkpoints, run directories, or validation results;
- geometry-OOD campaigns or the planned standalone benchmark suite.

The residual implementation covers interior cells. Boundary conditions,
cross-block halo/interface residuals, and residual-loss training are outside
the current scope.

## Required sibling repositories

Keep the three repositories as siblings. This repository imports the two
foundations instead of copying them:

```text
wing-cfd-workspace/
├── parametric-swept-wing-cfd/     geometry, mesh, CFD, structured export
├── surrogate-newton-cfd/          shared surrogate and ADflow/NK runtime
└── structured-wing-surrogate-cfd/ this 3D model, training, and evaluation
```

The tested revisions are recorded in [`foundation-lock.yaml`](foundation-lock.yaml):

- `parametric-swept-wing-cfd` at `cff53b2fabf89e9ad060337bc9f1ec8ff8d1fc68`;
- `surrogate-newton-cfd` at tag `v0.1.0-rc1`, commit
  `6ff4352407e22cb4461e5d77ad284ba5706c4f58`.

For repositories that require authentication, configure GitHub SSH access and
clone all three into the same workspace:

```bash
mkdir -p /path/to/wing-cfd-workspace
cd /path/to/wing-cfd-workspace

git clone git@github.com:LeiMingcheng/parametric-swept-wing-cfd.git
git clone git@github.com:LeiMingcheng/surrogate-newton-cfd.git
git clone git@github.com:LeiMingcheng/structured-wing-surrogate-cfd.git

git -C parametric-swept-wing-cfd checkout cff53b2fabf89e9ad060337bc9f1ec8ff8d1fc68
git -C surrogate-newton-cfd checkout v0.1.0-rc1
```

## Environment and installation

Python 3.10 and an NVIDIA CUDA build of PyTorch are the maintained baseline.
The recommended reproducibility route is the locked environment supplied by
`surrogate-newton-cfd`. Create it from that repository so its editable package
entry resolves correctly:

```bash
cd /path/to/wing-cfd-workspace/surrogate-newton-cfd
conda env create -f environment.yml
conda activate surrogate-newton-cfd
```

Install the parametric repository's project-specific CST implementation and
then the two 3D repositories:

```bash
cd /path/to/wing-cfd-workspace/parametric-swept-wing-cfd
$CONDA_PREFIX/bin/python -m pip install -e vendor/cst-modeling3d
$CONDA_PREFIX/bin/python -m pip install -e '.[dev]'

cd /path/to/wing-cfd-workspace/structured-wing-surrogate-cfd
$CONDA_PREFIX/bin/python -m pip install -e '.[analysis,dev]'
```

This is sufficient for dataset loading, model training, field inference,
figures, HDF5 payloads, and the Torch residual tests. It does not require an
ADflow source build.

### ADflow, CGNS restart, force evaluation, and NK recovery

Solver-backed operations reuse the complete locked environment and solver
stack maintained by `surrogate-newton-cfd`; this repository does not maintain
a second ADflow installation procedure. From the locked Surrogate–Newton
revision, follow its [`docs/solver_stack.md`](https://github.com/LeiMingcheng/surrogate-newton-cfd/blob/v0.1.0-rc1/docs/solver_stack.md)
guide and run its fetch, build, and verification stages:

```bash
cd /path/to/wing-cfd-workspace/surrogate-newton-cfd
scripts/fetch_solver_stack.sh /path/to/solver-stack
scripts/build_solver_stack.sh /path/to/solver-stack
scripts/verify_solver_stack.sh /path/to/solver-stack
```

Do not run CGNS restart, ADFLOW force evaluation, or NK recovery until the
solver-stack verification succeeds. The active Python, `mpi4py`, PETSc,
compiled solver extensions, and MPI launcher must use the same MPI
implementation.

## Verify the checkout

Run the portable test suite from this repository:

```bash
cd /path/to/wing-cfd-workspace/structured-wing-surrogate-cfd
$CONDA_PREFIX/bin/python -m pytest -q -p no:cacheprovider
$CONDA_PREFIX/bin/python scripts/audit_release.py
```

The release snapshot was accepted with 39 passing tests. These tests use tiny
synthetic blocks and may run on CPU; they verify imports, tensor contracts,
forward/backward, one optimizer step, payloads, force metrics, and analytical
residual cases.

On a machine with at least two visible CUDA GPUs, run the real two-rank DDP
availability check:

```bash
CUDA_VISIBLE_DEVICES=0,1 scripts/smoke_ddp_2gpu.sh
```

The script generates a temporary two-block dataset under `tmp/`, launches
`torchrun --nproc-per-node=2`, performs one optimizer step, and checks that
`last.pt` and `summary.json` were written. It verifies distributed wiring, not
13-block memory use, throughput, convergence, or accuracy.

## Prepare a structured training dataset

First use `parametric-swept-wing-cfd` to generate completed geometry → mesh →
ADflow cases. Its maintained runtime layout is documented in
[`docs/pipeline/geometry-mesh-cfd.md`](https://github.com/LeiMingcheng/parametric-swept-wing-cfd/blob/main/docs/pipeline/geometry-mesh-cfd.md).
The dataset root passed below must contain completed sample/condition trees and
their volume CGNS solutions.

From this repository, regroup those completed cases into block-wise HDF5
shards:

```bash
cd /path/to/wing-cfd-workspace/structured-wing-surrogate-cfd

$CONDA_PREFIX/bin/python scripts/export_structured_dataset.py \
  --dataset-root /path/to/completed-parametric-cfd-dataset \
  --output-dir /path/to/structured-block-dataset \
  --shard-size 16 \
  --compression lzf
```

The output contains:

```text
/path/to/structured-block-dataset/
├── index.csv
├── metadata.json
└── block_*.h5 or block-signature shard files
```

One logical training item is a complete `(sample_id, cond_idx)` case regrouped
across the fixed block catalog. See [`docs/dataset-contract.md`](docs/dataset-contract.md)
for the state, coordinate, volume, flow, force, and geometry schemas.

The production configuration uses normalized primitive fields and PIR-DM
log-scaled SA state. Compute `turbulent_scale_stats.json` next to `index.csv`
before training:

```bash
export STRUCTURED_INDEX=/path/to/structured-block-dataset/index.csv

$CONDA_PREFIX/bin/python - <<'PY'
import os
from surrogate3d.data import write_turbulent_scale_stats

path = write_turbulent_scale_stats(os.environ["STRUCTURED_INDEX"], num_workers=4)
print(path)
PY
```

The geometry feature order, block catalog, and state field order are
compatibility-critical. Do not combine datasets with different schemas or use
a checkpoint against a different block catalog.

## Train the 3D global FNO

The maintained production configuration is
[`configs/global-fno-gpu.yaml`](configs/global-fno-gpu.yaml). It predicts all
six primitive-state channels over the complete block catalog and enables
volume-weighted, wall-Cp, and wall-Cf objectives.

Run the two-GPU launcher from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
DATASET_INDEX=/path/to/structured-block-dataset/index.csv \
TRAIN_OUTPUT_DIR=/path/to/training-run \
scripts/train_global_fno_2gpu.sh
```

`NPROC_PER_NODE` defaults to `2` and can be increased for more local GPUs.
`data.batch_size` in the YAML is the per-rank whole-case batch size. To use a
modified configuration:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
NPROC_PER_NODE=2 \
CONFIG_PATH=/path/to/training-config.yaml \
DATASET_INDEX=/path/to/structured-block-dataset/index.csv \
TRAIN_OUTPUT_DIR=/path/to/training-run \
scripts/train_global_fno_2gpu.sh
```

The main DDP rank writes:

- `resolved_config.json`, the effective configuration;
- `step_XXXXXXXX.pt`, periodic checkpoints when enabled;
- `last.pt`, the final checkpoint;
- `summary.json`, step counts, split sizes, parameter count, device, and time.

The checkpoint embeds the model state, dataset schema, block catalog,
normalizer configuration, split, optimizer, scheduler, and training history.
Keep it with the matching dataset schema.

## Evaluate a checkpoint without ADflow

Pass `--index` when the structured dataset has moved since training. The
following command produces field figures and a whole-case HDF5 payload but
does not write a CGNS restart:

```bash
$CONDA_PREFIX/bin/python -m surrogate3d.cli.evaluate \
  --checkpoint /path/to/training-run/last.pt \
  --index /path/to/structured-block-dataset/index.csv \
  --case-index 0 \
  --output-dir /path/to/evaluation \
  --device cuda \
  --no-cgns-restart
```

Use `--no-figures` or `--no-payload` to disable those products. The command
writes a case summary JSON beneath the requested output directory.

## Export a CGNS restart and evaluate forces

After the Surrogate–Newton solver stack has passed verification, omit
`--no-cgns-restart` and provide the source volume CGNS template when it cannot
be resolved from the dataset index:

```bash
$CONDA_PREFIX/bin/python -m surrogate3d.cli.evaluate \
  --checkpoint /path/to/training-run/last.pt \
  --index /path/to/structured-block-dataset/index.csv \
  --case-index 0 \
  --output-dir /path/to/restart-evaluation \
  --device cuda \
  --template-cgns /path/to/source-case/wing_000_vol.cgns \
  --restart-positivity-projection \
  --run-force-eval \
  --force-eval-ranks 8
```

The positivity projection applies only to the exported restart and does not
modify the raw surrogate prediction used for model metrics.

## Run Newton–Krylov recovery

The convenience command predicts one indexed case, writes its CGNS restart,
and delegates correction to the shared Surrogate–Newton/ADflow runtime:

```bash
$CONDA_PREFIX/bin/python -m surrogate3d.cli.nk_recover \
  --checkpoint /path/to/training-run/last.pt \
  --case-index 0 \
  --output-dir /path/to/nk-recovery \
  --device cuda \
  --mpi-ranks 8 \
  --cycles 10 \
  --inner-cycles 1 \
  --l2-tol 1.0e-8
```

This convenience command currently reads the dataset index stored in the
checkpoint configuration. Keep that path available for recovery; use the
evaluation command's explicit `--index` override when only inference products
are required.

## Repository layout and further documentation

```text
configs/      production and DDP-smoke configurations
docs/         architecture, dataset, residual, and testing contracts
reference/    compact residual-alignment evidence
scripts/      dataset export, training, validation, and release checks
src/          installable surrogate3d package
tests/        portable unit and contract tests
```

- [`docs/architecture.md`](docs/architecture.md): whole-case model flow;
- [`docs/dataset-contract.md`](docs/dataset-contract.md): HDF5/index/schema contract;
- [`docs/residual.md`](docs/residual.md): differentiable 3D residual scope;
- [`docs/testing.md`](docs/testing.md): CPU, GPU, and solver test boundaries;
- [`SOURCE_PROVENANCE.md`](SOURCE_PROVENANCE.md): migration provenance;
- [`foundation-lock.yaml`](foundation-lock.yaml): exact sibling revisions.

## Repository and release status

The repository code is released under the [BSD 3-Clause License](LICENSE).
See [`NOTICE.md`](NOTICE.md) for the licenses of external dependencies.
Workflows that use `parametric-swept-wing-cfd` require access to its private
repository at the revision documented in [`foundation-lock.yaml`](foundation-lock.yaml).
