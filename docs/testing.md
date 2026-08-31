# Testing

The release checks are intentionally split by purpose.

## Portable checks

Portable checks use synthetic structured blocks and run without a production
dataset or compiled CFD solver. They verify imports, data contracts, forward
shape, gradients, one optimizer step, payload serialization, and analytical
residual cases. They may run on CPU because the tensors are deliberately tiny.

## GPU training smoke

The included availability smoke creates a deterministic two-block synthetic
dataset and launches two CUDA processes through `torchrun`:

```bash
scripts/smoke_ddp_2gpu.sh
```

It verifies that both DDP ranks can load the whole-case path, run forward and
backward, apply one optimizer step, and write a reloadable checkpoint. The
small tensors exercise the distributed wiring rather than representative
training cost.

A production-data smoke remains a separate integration check. It uses an
external 13-block shard with `configs/global-fno-gpu.yaml` to verify actual
shapes and memory pressure; Git does not contain that CFD shard.

## Solver integration

ADflow alignment and Newton recovery require the external MPI/solver stack and
CGNS cases. These tests are marked `integration` and are not part of a generic
CPU CI job.
