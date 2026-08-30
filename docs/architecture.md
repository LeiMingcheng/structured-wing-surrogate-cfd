# Architecture

The maintained baseline predicts one complete CFD case while preserving the
original structured block topology.

```text
geometry + Mach/AoA/Re
          |
          v
 condition encoder
          |
          +------------------------------+
          |                              |
 block-local coordinates           block-local coordinates
          |                              |
     patchify3d                       patchify3d
          |                              |
 block-specific stem              block-specific stem
          +------- shared 3D spectral trunk -------+
                              |
                 structured block tokens
                              |
                  global Transformer mixer
                              |
              patch-wise context reinjection
                              |
                 block-specific output heads
                              |
                   six-channel 3D fields
```

Each case contains a fixed catalog of structured blocks. Every block keeps its
native `(I,J,K)` shape and has its own stem and head. The spectral trunk is
shared, while a global token mixer exchanges information between blocks before
the final prediction.

Model inputs per block are physical coordinates `(x,y,z)` plus normalized
structured indices `(i,j,k)`. The global condition combines the dataset's
versioned geometry vector with Mach number, angle of attack, and Reynolds
number. Model outputs are density, three velocities, pressure, and the
Spalart–Allmaras working variable.
