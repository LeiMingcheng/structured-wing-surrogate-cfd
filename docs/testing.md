# Testing

The release checks are intentionally split by purpose.

## Portable checks

Portable checks use synthetic structured blocks and run without a production
dataset or compiled CFD solver. They verify imports, data contracts, forward
shape, gradients, one optimizer step, payload serialization, and analytical
residual cases. They may run on CPU because the tensors are deliberately tiny.

## GPU training smoke

The GPU smoke uses a small external 13-block shard and the production model
path with `torchrun`. It verifies CUDA execution, DDP synchronization, and one
checkpoint write. It is separate from the portable checks because Git does not
contain the CFD shard.

## Solver integration

ADflow alignment and Newton recovery require the external MPI/solver stack and
CGNS cases. These tests are marked `integration` and are not part of a generic
CPU CI job.
