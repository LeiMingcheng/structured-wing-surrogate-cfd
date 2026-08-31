# Source provenance

The initial source selection was prepared on 2026-08-30 from the internal
`swepr_wing` research workspace. That workspace was not a Git working tree, so
it has no authoritative source commit. The initial migration deliberately
selects the maintained `surrogate3d` runtime packages rather than mirroring the
entire workspace.

Selected areas:

- data loading and normalization;
- hierarchical 3D global FNO models;
- training and checkpoint orchestration;
- field, force, residual, restart, and Newton-recovery evaluation;
- structured-grid geometry, viscous flow, and SA residual operators;
- maintained train/evaluate/recovery entry points.

Not selected in the initial migration:

- generated datasets, validation directories, reports, and run products;
- private checkpoints and normalization assets;
- internal handoff notes and machine-specific campaign launchers;
- geometry-OOD campaign scripts, which are deliberately left to the separate
  benchmark effort.

The two external project revisions used during migration are recorded in
`foundation-lock.yaml`.
