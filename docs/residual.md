# Three-dimensional residual scope

The Torch backend evaluates differentiable finite-volume residuals on the
interior cells of each structured block. The primitive state is density,
three velocity components, pressure, and the Spalart–Allmaras working variable.

Included terms are:

- continuity, three momentum equations, and total energy;
- central inviscid flux with scalar Jameson dissipation;
- dual-volume nodal gradients and face-normal gradient correction;
- Sutherland laminar viscosity and SA eddy viscosity;
- viscous work, heat conduction, and the SA transport equation.

The alignment audit compared the Torch operator and freshly exported ADFLOW
residual fields on the same double-precision state. Across 13 blocks and
649,792 two-cell-interior cells, flow-component correlations exceed 0.99997
and relative RMSE remains below 0.7%. The SA residual and eddy viscosity agree
to numerical precision. Machine-readable metrics are in
[`reference/residual-alignment.json`](../reference/residual-alignment.json).

This is not a full-domain residual. Wall/farfield boundary rows,
multi-block halo/interface residuals, batch validation-runner integration, and
residual-loss training remain outside the current implementation.
