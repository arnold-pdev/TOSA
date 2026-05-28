# TOSA

**Topology Optimization Sensitivity Analysis** — compute how global compliance changes with respect to design variables for fixed boundary conditions and loads.

## Sprint workflow

1. **NITO-3D dataset** — reference topologies, shapes, BCs, loads, and volume fractions.
2. **ANSYS** — FEA on each design: apply constraints and loading, export **displacement fields** (and any stress/energy fields needed).
3. **Python** — post-process ANSYS results to compute **compliance** and **∂C/∂ρ** (or equivalent design sensitivities) for plotting and validation.

Details, data layout, and next steps: [database/README.md](database/README.md).
