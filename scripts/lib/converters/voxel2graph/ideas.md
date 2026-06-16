**Full volume vs. surface — build both, default to full volume.** This we settled before but it's worth restating with the implementation consequence. The full-volume graph is the primary object: nodes = filled voxels, edges = 6-connectivity face-adjacency. It carries the internal connectivity that compliance depends on, and it's the thing your homology runs on. The surface graph is a *derived view* — the subgraph induced by filled voxels having at least one empty 6-neighbor — and you extract it with a one-line mask, never compute it independently. Surface nodes are where the frontier lives, where boundary normals attach, where the overhang signal evaluates. So: full volume as the stored representation, surface as a cheap induced subgraph. Both, but one is a filter on the other.

**voxel2graph for 32³.** At 32³ you have 32768 possible cells; a 30%-fill design is ~10⁴ nodes, ~3×10⁴ edges under 6-connectivity. That's tiny for PyG — don't over-engineer it. The key implementation choice is to compute edges *vectorized* via the grid structure, not by per-voxel neighbor search (which is what you'd naively write and what gets slow). The pattern:

```python
import torch
import numpy as np
from torch_geometric.data import Data

def voxel2graph(vox, seed_mask=None):
    # vox: (32,32,32) bool tensor; seed_mask: (32,32,32) bool, BC/load cells
    coords = torch.nonzero(vox, as_tuple=False)          # (N,3) long, the cell indices
    N = coords.shape[0]

    # dense lookup: grid position -> node index, -1 if empty
    lin = lambda c: (c[:,0]*32 + c[:,1])*32 + c[:,2]
    idx_grid = torch.full((32*32*32,), -1, dtype=torch.long)
    idx_grid[lin(coords)] = torch.arange(N)

    # 3 positive offsets suffice (each undirected edge found once), then symmetrize
    offsets = torch.tensor([[1,0,0],[0,1,0],[0,0,1]])
    edges = []
    for off in offsets:
        nbr = coords + off
        inb = (nbr < 32).all(dim=1)                       # in-bounds (off is positive)
        src = torch.arange(N)[inb]
        dst = idx_grid[lin(nbr[inb])]
        valid = dst >= 0                                  # neighbor is filled
        edges.append(torch.stack([src[valid], dst[valid]]))
    ei = torch.cat(edges, dim=1)
    ei = torch.cat([ei, ei.flip(0)], dim=1)               # make undirected

    x = coords.float() / 31.0                             # normalized position features
    data = Data(x=x, edge_index=ei.contiguous(), pos=coords.float())
    data.coord_index = idx_grid                           # keep for trivial graph2voxel
    if seed_mask is not None:
        data.is_seed = seed_mask[coords[:,0],coords[:,1],coords[:,2]]
    return data
```

Two things to note. The `idx_grid` dense lookup is your coord-map — O(M) memory (M=32768, trivial) and gives O(1) neighbor resolution, which is what makes this vectorized rather than a dict-lookup loop. Keeping it on the `Data` object makes graph2voxel a one-liner (scatter node values back to `lin(coords)`, reshape to 32³). The "3 positive offsets then symmetrize" trick is the standard way to avoid double-counting undirected edges — you'll see it throughout PyG grid-graph code.

graph2voxel is the inverse and the round-trip test is just `assert (graph2voxel(voxel2graph(v)) == v).all()` over random volumes — build that test in the same sitting, as planned.

**Computing Betti numbers.** Three different tools for three different numbers, and the cheap ones are worth doing separately rather than running full homology for everything:

- **b₀ (components):** don't use homology — use the graph you already built. `scipy.sparse.csgraph.connected_components` on the adjacency, or `torch_geometric.utils` to SciPy. This is union-find under the hood, milliseconds at 10⁴ nodes, and it's the number you check most often (connectivity guarantee). It runs on the 6-connectivity *graph*, which is the mechanically honest connectivity.

- **b₁ and b₂ (loops and voids):** these need actual cubical homology, because b₁ counts independent cycles and b₂ counts enclosed cavities — neither is a plain graph property of the 1-skeleton (b₁ of a graph is just edges−nodes+components, but that counts graph cycles, not topological loops of the solid, which differ once you fill 2-cells and 3-cells of the cubical complex). Use **GUDHI's `CubicalComplex`**, which takes the voxel array directly:

```python
import gudhi
def betti_numbers(vox):
    # GUDHI cubical complex from the binary field
    cc = gudhi.CubicalComplex(
        dimensions=vox.shape,
        top_dimensional_cells=(~vox).flatten().astype(float)  # 0 = filled, 1 = empty
    )
    cc.compute_persistence()
    return cc.betti_numbers()   # [b0, b1, b2]
```

The filtration-value convention (filled cells get low value) matters — you want the sublevel set at threshold to be the solid. Sanity-check it on known shapes: a solid block → [1,0,0], a torus → [1,1,0] (wait — a solid torus is [1,1,0], a hollow shell torus differs), a hollow sphere → [1,0,1]. Build a tiny library of known-topology test volumes and assert their Betti numbers; that's the b₁/b₂ analogue of the round-trip test, and it's how you catch convention bugs (the filled-vs-empty polarity is the classic one).

One honest caveat on b₀ consistency: GUDHI's cubical b₀ and your 6-connectivity graph b₀ should agree, *but* cubical complexes treat face-adjacency as connection by default, so they'll match the 6-connectivity graph, not an 18- or 26-connectivity one. Keep that straight — if you ever compute build-connectivity (18-conn downward), that's a *different* graph and won't match GUDHI's cubical b₀. The mechanical b₀ (6-conn) is the one homology gives you natively.

**Resolving part location against the domain-face constraint.** This is the cleanest of your questions because, per your own earlier conclusion, for NITO's prescribed axis-aligned domain-face BCs the plane is *analytically known* — you do not infer it. The part occupies a known position in the 32³ domain, and the BC face is a known coordinate plane (say z=0). So "resolve the location" reduces to:

1. The voxel grid *is* the domain frame — coordinates are already in domain coordinates, no registration needed. The part's location is given.
2. The coplanarity constraint ("edges coplanar with a domain face") means the seed cells on that face have a known coordinate (z=0 layer), and you tag them as seeds with `seed_mask`. You don't solve for the plane; you assert it.
3. The only real operation is *snapping the BC inset plane to the nearest voxel z-boundary* (which you flagged in the TOSA meshing work) so the constrained layer sits on a clean voxel interface. That's a rounding operation, not an inference.

So there's no localization/registration problem here in the prescribed-BC case — the over-engineered "infer the contact plane via joint variational optimization" framing you correctly rejected for TOSA applies identically here. The plane is data, not an unknown. The part location is the grid frame. The only thing to get right is consistent indexing convention (which axis is build-z, which face is the BC face) and tagging the seed layer — both bookkeeping, not computation.

**How graphs are used in similar ML problems — the parts that bear on your choices.** A few conventions worth borrowing rather than reinventing:

- **PyG `Data` + `InMemoryDataset` is the standard container**, and the standard batching is block-diagonal (graphs of different node counts concatenated with an offset `batch` vector). At 32³ with ~10⁴ nodes per graph you can fit many per batch; this is well within normal GNN-on-3D-shape practice (e.g. mesh/point-cloud GNNs routinely handle 10⁴–10⁵ nodes).
- **The voxel-as-graph vs. point-cloud-as-graph distinction matters for your features.** Point-cloud GNNs (PointNet++, DGCNN) build edges by k-NN in continuous space; you have a *regular grid*, so your edges are exact lattice adjacency, not k-NN — simpler and exact. Don't import k-NN graph construction; you don't need it. Your structure is closer to how 3D CNNs are sometimes "graphified" for sparse convolution (Minkowski/SparseConvNet treat occupied voxels as a sparse set with lattice neighbors) — that sparse-voxel literature is the right mental reference, not point-cloud GNNs.
- **Sparse convolution is the relevant alternative architecture to keep in mind.** If you later find message-passing over the full-volume graph has too small a receptive field (the global-equilibrium concern from before), Minkowski/torchsparse-style sparse 3D convolution over the same occupied-voxel set is the natural upgrade — it's CNN-like spatial inductive bias on exactly your sparse lattice data, and it consumes the same coordinate representation you're already building. Worth knowing it's there; not worth building now.
- **Store node features as you'll consume them:** normalized position (`coords/31`), a seed flag, and later the boundary-normal and strain-energy scalars. All attach to the same `Data.x` by concatenation. Keep `pos` (integer coords) separate from `x` (learned features) — PyG convention, and it keeps graph2voxel exact regardless of what features you add.

**Net recommendation:** full-volume 6-connectivity graph as the stored object via the vectorized dense-lookup construction above; surface as an induced-subgraph mask; b₀ from `connected_components` on the graph, b₁/b₂ from GUDHI cubical homology with a known-shape test suite guarding the polarity convention; part location is given by the grid frame and the BC plane is asserted not inferred (just snap and tag the seed layer); PyG `Data`/`InMemoryDataset` for storage, with sparse convolution noted as the future architectural fallback if message-passing receptive field becomes limiting. This is all within the [1]+[2] first-push scope plus the seed-tagging, so it slots directly into the to-do list.

If you want, I can launch a proper literature dig specifically on graph/sparse-voxel representations in 3D generative ML (Minkowski-style sparse conv, voxel-graph GNNs, autoregressive 3D-shape generation) to pressure-test the architectural fallback before you commit — but for the conversion/homology/localization tooling you asked about, the above is the implementation and you don't need a search to start building it.