# Dataset contract

The model consumes HDF5 block shards accompanied by `index.csv` and
`metadata.json`. One logical sample is a complete `(sample_id, cond_idx)` CFD
case regrouped from every block in the fixed topology catalog.

Each HDF5 record stores:

- `state`: six primitive-state channels in `(C,I,J,K)` order;
- `coords`: physical `(x,y,z)` cell-center coordinates;
- `cell_volumes`: per-cell geometric volumes;
- `flow`: Mach number, angle of attack, and Reynolds number;
- `forces`: wing lift, drag, and pitching-moment coefficients;
- `geometry`: a versioned geometry-condition vector.

The loader adds normalized structured indices to form model coordinates
`(x,y,z,i_norm,j_norm,k_norm)`. It never combines blocks into a padded dense
cube.

Geometry feature names, their ordering, block shapes, state fields, and the
dataset schema version are compatibility-critical. New checkpoints store this
schema directly so inference with `--index` does not need the original
training machine's metadata path. Historical 48-dimensional checkpoints and
newer five-section exports must not be mixed without an explicit compatibility
mapping and a newly trained model.
