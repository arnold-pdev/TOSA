# voxel2surf

Convert a **binary voxel topology** (a NITO topology-optimization result, thresholded
to 0/1) into a **smooth, watertight, FEM-ready surface mesh** that (a) is faithful to
the voxel occupancy, (b) preserves thin members, (c) carries the boundary-condition /
load tags, and (d) does not shrink.

The headline idea: surface fairing here is **not a filter, it is the discrete
*obstacle problem*** — minimize a thin-plate bending energy plus a soft fidelity term,
subject to a per-vertex occupancy corridor and an axis-aligned constraint hierarchy,
solved as a bound-constrained quadratic program. This document explains every stage,
the reasoning behind it, and the literature it draws on. The lineage report with full
bibliographic detail is in [`binary_smoothing.md`](binary_smoothing.md).

---

## 0. Why this is hard (and why plain smoothing fails)

Binarizing a density field at `ρ ≥ 0.5` commits the **topology** (which voxels are
solid, how they connect) but discards the **sub-voxel geometry**. The raw "cuberille"
surface is therefore blocky, and recovering a smooth surface from it is a
*reconstruction* problem, not a *filtering* problem. Two failure modes dominate naive
smoothing (e.g. Taubin λ|μ):

- **Shrinkage / thin-member collapse** — an unconstrained smoother has no notion of
  *where the surface should be*, so opposite walls of a one-voxel member diffuse
  together and the member necks off.
- **Residual staircase on shallow edges** — the staircase there is *low-frequency*
  (long runs + occasional steps), so it sits in a low-pass filter's pass band and
  survives any number of iterations.

Both are failures of having **no data term and no occupancy constraint**. Everything
below exists to add those, correctly.

---

## 1. Extraction — `extract.py`

`cuberille_mesh(vox, origin, spacing, density_cutoff)` → `verts, faces (quads), prov`.

- **What:** one quad per *exposed* voxel face (cuberille / "opaque cube" surface,
  **Herman & Liu 1979**). Vertices sit at grid corners; a quad indexes the four
  corners of an exposed face.
- **The weld (why it's not just marching cubes):** corners are welded across voxels
  only where they are **face-connected** (6-connectivity), evaluated locally per grid
  corner from its 8-voxel "rosette" via a precomputed 256×8 component table. This makes
  the **surface graph's connectivity identical to the voxel FEM's face-connectivity**:
  diagonal (edge/vertex) contacts split into separate manifold sheets; face contacts
  fuse. This is the load-bearing reason we use cuberille rather than a dual method
  (Surface Nets / dual contouring): we want *explicit, exact* control of topology at the
  ambiguous configurations, not the implicit decisions a dual extractor makes. Pure
  occupancy also lacks the Hermite normals dual contouring needs (**Ju et al. 2002**).
- **Provenance seeded here:** `prov["vmask"]` is a per-vertex **6-bit box-face mask**
  read *exactly* from the integer grid key (no tolerance) — bit `2·axis+side`. This is
  the seed for the constraint labelling and is propagated through refinement (Stage 3).

---

## 2. Constraint labelling — `label.py`

The smoothing must respect the design-domain box: a surface vertex on a box face may
only slide *in* that face, a vertex on a box edge only *along* that edge, a corner is
fixed. `classify(...)` produces a `Constraints` record encoding this.

- **The taxonomy.** Because the box planes are axis-aligned, each face pins one
  *coordinate*. The admissible motion is therefore set by the **number of distinct axes
  pinned**:

  | distinct axes | class | DOF | projector `P` |
  |---|---|---|---|
  | 0 | free | 3 | `I` |
  | 1 | on_face | 2 | `I − nnᵀ` (slide in plane) |
  | 2 | on_edge | 1 | `d dᵀ` (slide along edge line) |
  | 3 | on_corner | 0 | `0` (fixed) |

  Load anchors are additionally fixed. This is the standard Dirichlet-boundary treatment
  of variational fairing (**Botsch & Sorkine 2008**) specialized to axis-aligned planes;
  the specialization is what makes the later solve *separable per coordinate*.

- **Provenance, not geometry.** The 6-bit mask comes from `face_mask=vmask` (exact),
  not from re-testing vertex coordinates against the box with a tolerance. Through
  subdivision the mask propagates by **bitwise AND** (`mask[midpoint] = mask[a] & mask[b]`)
  — a midpoint lies on a coordinate plane iff *both* parents do, so AND is exactly the
  plane-intersection rule. This keeps constraints exact across refinement levels and
  avoids the tolerance fragility that produced spurious BC residuals.

- **BC / load tags.** `bc_quad_ids` assigns each cuberille quad its support-patch id by
  *exact integer* voxel-face match (no centroid geometry); `bc_vertex_mask` lifts that to
  a per-vertex mask that rides the same subdivision AND; `face_bc_arrays` reads the final
  per-face `patch_id` / `fix_x,y,z` / `boundary_type` back off the mask ("all corners
  agree"). No post-mesh geometric re-derivation, so no spill-over artifacts.

---

## 3. Fairing — `fair.py` (+ `corridor.py`)

Two modes, deliberately **not** on equal footing:

- **`implicit` (primary)** — the discrete obstacle problem. Solves the actual
  reconstruction problem (smooth + occupancy-consistent + non-shrinking).
- **`taubin` (fast fallback)** — the original explicit λ|μ filter. Fast, but has **no
  corridor and no data term**, so it can collapse thin members. For previews/comparison.

The two share the cotangent operator and the constraint pins; `taubin` is a degenerate
special case of `implicit` (explicit, truncated, no corridor, membrane-only).

### 3a. The discrete operators

- **Cotangent Laplacian** `Lₛ = D − A` (`_cotangent_laplacian`), with clamped cotangent
  edge weights (**Pinkall & Polthier 1993**). This is the discretization of the
  Dirichlet energy `xᵀLₛx`.
- **Lumped mass matrix** `M` (`_lumped_mass`), barycentric Voronoi vertex areas
  (**Meyer, Desbrun, Schröder & Barr 2003**). Needed so the bi-Laplacian is
  geometrically correct on irregular triangles.

### 3b. The energy: thin-plate, not membrane

`solve_implicit(..., order=...)` minimizes a smoothness energy:

- **`thin-plate` (default):** `Q = Lₛ M⁻¹ Lₛ` — the **bi-Laplacian** / thin-plate
  bending energy (**Kobbelt 1997**; **Botsch & Sorkine 2008**). It is **non-shrinking**
  (minimizing curvature does not pull toward a point) and **C¹ at constraints** (no
  membrane "tent"/kink — this subsumes the Taubin §4.2 smooth-interpolation trick we
  briefly used and then retired).
- **`membrane`:** `Q = Lₛ` — first-order Dirichlet energy. *Shrinks catastrophically*
  (soap-film/minimal-surface collapse — exactly what Taubin's μ-step exists to fight),
  so it is offered only for comparison.

### 3c. The data term (anti-shrink)

The energy alone still drifts. From **pure binary** the only honest target is "stay on
the cuberille boundary," so we add a **soft positional fidelity term**
`½·w·‖x − x₀‖²` (the `reg` / `--data-weight` argument; `x₀` = cuberille position). This
is the smoothness-plus-fidelity least-squares structure of **Nealen, Igarashi, Sorkine
& Alexa 2006 ("Laplacian Mesh Optimization")**. Crucially, the thin-plate energy's
*shrinking* motion is mostly **normal** to the surface, so an *isotropic* spring to `x₀`
preferentially resists shrinkage while leaving **tangential smoothing free** — and it
keeps the per-coordinate separability. This single term is what fixed the shrinkage; it
also regularizes the solve enough to remove self-intersections.

### 3d. The occupancy corridor — `corridor.py`

`derive_bounds(verts, vox, origin, spacing, ...)` → per-vertex axis-aligned box
`[lo, hi]`. This is the hard inequality constraint that makes it an **obstacle problem**,
and the direct descendant of **Gibson's Constrained Elastic Surface Nets (1998)** —
per-cell containment to keep thin features from collapsing — generalized from
cell-centered nodes to our corner vertices via a **regularized signed-distance transform
of the binary** (the binary SDT is severely quantized, so it is Gaussian-smoothed first).
The corridor is **asymmetric/thickness-adaptive**: tight on the inward (toward-solid)
side at thin members, generous tangentially. After the data term took over the
anti-shrink role, the corridor is demoted to a **loose safety leash + thin-feature
protection** (`--corridor-voxels`).

### 3e. The formulation and the solver

Putting 3a–3d together, per coordinate axis we solve a **bound-constrained convex QP**

> minimize `½ xᵀQx + ½·ε·‖x − x₀‖²`  s.t.  `x[pinned]=target`,  `lo ≤ x ≤ hi`

which is the **discrete obstacle problem** (**Lions & Stampacchia 1967**); `Q` is SPD.
Because the pins and corridor are axis-aligned, this **separates into three independent
scalar QPs** sharing `Q`.

- **Solver: primal-dual active set (PDAS) = semismooth Newton** (`_solve_axis_bounded`,
  **Hintermüller, Ito & Kunisch 2002**). Each round predicts the active set from the
  bound multiplier `μ + c₀(x − bound)` — which lets a vertex *leave* a bound, not just
  join it — fixes active vertices to their bound, and solves the SPD system on the
  inactive set. This converges to the *true* constrained minimizer, so the surface meets
  the corridor **tangentially** (C¹ contact) instead of creasing — fixing the
  self-intersections a naive "clamp-and-freeze" active set produces. A **best-feasible-
  energy globalization** guards against PDAS's occasional cycling. (Robust alternative:
  Moré–Toraldo GPCG; scalability escalation past ~10⁶ vertices: Kornhuber monotone
  multigrid — both in [`binary_smoothing.md`](binary_smoothing.md).)

### 3f. Taubin (fallback) — `constrained_taubin`, `smooth_interpolate`

The original explicit λ|μ filter (**Taubin 1995**): `x ← x + f·P·(Wx − x)` with row-
normalized neighbour operator `W`, the constraint projector `P`, and Taubin's
shrink/anti-shrink coefficients. `λ=0.5` annihilates the mesh Nyquist mode (the
staircase); `μ` is derived from a pass-band frequency. An optional §4.2 "smooth
interpolation" (`smooth_interpolate`, fixed uniform precomputed kernel, eqn 10) gives
non-tent constraint interpolation. Kept only as the fast path; the implicit solver
supersedes all of it.

---

## 4. Refinement — `subdivide`, `decimate_to`

- **`subdivide`** — one 1→4 midpoint subdivision, implemented in numpy so the provenance
  masks (box-face + BC) ride through by AND. Quads are triangulated here.
- **`decimate_to`** — quadric edge-collapse to a triangle budget (pyvista), boundary-
  preserving. The BC mask is carried across by nearest-source transfer (decimation drops
  vertex parentage). Scheduling (`--sub-levels`, `--target-faces`) lives in `mesh_surface`.

The constraint and corridor are **re-derived per resolution level**, so the formulation
is resolution-robust: the data term anchors the *physical* geometry, decoupling "where
the surface is" from the iteration/face count.

---

## 5. Orchestration & validation — `main.py`

`mesh_surface(...)` runs: extract → BC label → native fair → optional subdivisions →
optional decimate → `validate` → BC cell-data → return. `validate` gates the result on
**watertightness**, **non-self-intersection** (a vectorized Möller broadphase that
excludes the flat BC patches), **inter-body overlap**, **BC-plane residual**,
**load-on-surface**, and reports volume-fraction delta and free-skin dihedral. The
`Options` dataclass is the single source of tunables; `cli.py` is a thin wrapper.

---

## 6. Why not just Taubin? — limitations and how they are addressed

| Taubin limitation | resolution here |
|---|---|
| No data term — only band-limits; low-frequency staircase unremovable; drifts | **Soft data term** `½w‖x−x₀‖²` gives the surface a target and resists drift (§3c) |
| Shrinkage, no per-location lower bound → thin members collapse | **Thin-plate energy** (non-shrinking, §3b) + **corridor** (§3d) + **data term** (the decisive lever) |
| Explicit relaxation: slow, spectrum-limited, CFL step cap | **Implicit solve** — one SPD system, all frequencies at once (§3e) |
| Constraints via per-step "reset" → tent/kink at point constraints | Thin-plate gives **C¹ at constraints** natively; constraints enter exactly (§3b, §3e) |
| Output is a resolution-coupled *transient*, not a defined object | The variational form has a **well-defined minimizer** (§3e) |
| Over-constrains box edges (pins them) | **Taxonomy** gives correct per-class motion; axis-alignment makes it separable (§2) |

Conceptual continuity: **Taubin's iteration *is* gradient descent on the membrane
energy; this pipeline replaces the relaxation with an implicit *solve* of a richer
energy (thin-plate + data) over a *constraint set* (pins + corridor).** It is an
evolution of Taubin, not a different family — which is why the cotangent operator and a
subordinate explicit fast-path are retained.

---

## 7. CLI quick reference

```
PYTHONPATH=scripts python -m lib.converters.voxel2surf.cli \
    --index 119 -o output/surfaces/small/119.vtp
```

| flag | meaning |
|---|---|
| `--fair-mode implicit\|taubin` | primary obstacle-problem solve vs fast Taubin fallback |
| `--smooth-order thin-plate\|membrane` | bending (non-shrinking) vs stretching (shrinks) energy |
| `--data-weight` | soft pull to the boundary — the **hug ↔ smooth** knob (anti-shrink) |
| `--corridor-voxels` | SDF corridor half-width (loose safety + thin-feature protection) |
| `--implicit-iters` | PDAS active-set rounds |
| `--sub-levels N` | unconditional ×4 midpoint subdivisions |
| `--target-faces T` | decimate to T triangles (omit → keep resolution) |
| `--on-fail warn\|raise` | behaviour when validation gates fail |

---

## References

- Herman & Liu (1979), *Three-dimensional display of human organs from computed tomograms*, CGIP. — cuberille.
- Taubin (1995), *A Signal Processing Approach to Fair Surface Design*, SIGGRAPH. — λ|μ filter.
- Pinkall & Polthier (1993), *Computing Discrete Minimal Surfaces and Their Conjugates*, Exp. Math. — cotangent weights.
- Desbrun, Meyer, Schröder & Barr (1999), *Implicit Fairing of Irregular Meshes using Diffusion and Curvature Flow*, SIGGRAPH. — implicit fairing.
- Meyer, Desbrun, Schröder & Barr (2003), *Discrete Differential-Geometry Operators for Triangulated 2-Manifolds*, VisMath. — mixed-Voronoi mass matrix.
- Kobbelt (1997), *Discrete Fairing*, IMA Surfaces. — discrete membrane/thin-plate energies.
- Botsch & Sorkine (2008), *On Linear Variational Surface Deformation Methods*, IEEE TVCG. — `Lₛ M⁻¹ Lₛ`, membrane vs thin-plate, `C^{k−1}`.
- Botsch, Kobbelt, Pauly, Alliez & Lévy (2010), *Polygon Mesh Processing*, AK Peters. — textbook; fairing as constrained energy minimization.
- Sorkine et al. (2004), *Laplacian Surface Editing*, SGP. — Dirichlet positional constraints, sparse SPD.
- Nealen, Igarashi, Sorkine & Alexa (2006), *Laplacian Mesh Optimization*, GRAPHITE. — smoothness + soft positional fidelity (the data term).
- Gibson (1998), *Constrained Elastic Surface Nets: Generating Smooth Surfaces from Binary Segmented Data*, MICCAI. — the corridor's ancestor.
- Frisken (2022), *SurfaceNets for Multi-Label Segmentations…*, JCGT; vtkSurfaceNets3D (Schroeder et al., 2023). — modern SurfaceNets.
- Ju, Losasso, Schaefer & Warren (2002), *Dual Contouring of Hermite Data*, SIGGRAPH. — lineage; needs Hermite data we don't have.
- Lions & Stampacchia (1967), *Variational inequalities*, CPAM. — obstacle-problem theory.
- Moré & Toraldo (1991), *On the Solution of Large QP with Bound Constraints*, SIAM J. Optim. — GPCG.
- Hintermüller, Ito & Kunisch (2002), *The Primal-Dual Active Set Strategy as a Semismooth Newton Method*, SIAM J. Optim. — the solver.
- Kornhuber (1994/1996), *Monotone multigrid methods for elliptic variational inequalities I/II*, Numer. Math. — scalable obstacle solver.

> The exact pipeline — cuberille-corner mesh + provenance constraints + per-axis
> bi-Laplacian box-QP + soft data term from pure binary — is a *synthesis* of the above;
> no single prior paper states it. See [`binary_smoothing.md`](binary_smoothing.md) for
> the full lineage analysis and solver survey.
