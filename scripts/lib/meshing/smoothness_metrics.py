"""Free-boundary smoothness proxies for stairstep / marching-cubes QA."""

from __future__ import annotations

import numpy as np
import trimesh


def free_boundary_smoothness(
    mesh: trimesh.Trimesh,
    tri_labels: np.ndarray,
) -> dict[str, float]:
    """
    Roughness metrics on unlabeled (free) triangles only.

    Higher ``axis_aligned_edge_frac`` and high ``free_dihedral_p95_deg`` often
    indicate visible voxel stairstepping on the exterior skin.
    """
    labels = np.asarray(tri_labels, dtype=int)
    free = labels < 0
    n_free = int(np.count_nonzero(free))
    nan = {
        "free_dihedral_mean_deg": float("nan"),
        "free_dihedral_p95_deg": float("nan"),
        "axis_aligned_edge_frac": float("nan"),
        "free_faces": float(n_free),
    }
    if n_free == 0 or len(mesh.faces) == 0:
        return nan

    adj = mesh.face_adjacency
    if adj.size == 0:
        return nan

    e0, e1 = adj[:, 0], adj[:, 1]
    both_free = free[e0] & free[e1]
    if np.any(both_free):
        angles_rad = mesh.face_adjacency_angles[both_free]
        dihedral_deg = np.degrees(np.abs(angles_rad))
        nan["free_dihedral_mean_deg"] = float(np.mean(dihedral_deg))
        nan["free_dihedral_p95_deg"] = float(np.percentile(dihedral_deg, 95))

        adj_edges = mesh.face_adjacency_edges[both_free]
        v0 = mesh.vertices[adj_edges[:, 0]]
        v1 = mesh.vertices[adj_edges[:, 1]]
        edge_vec = v1 - v0
        lengths = np.linalg.norm(edge_vec, axis=1, keepdims=True)
        lengths = np.maximum(lengths, 1e-12)
        edge_dir = edge_vec / lengths
        axis_aligned = np.zeros(len(edge_dir), dtype=bool)
        for ax in range(3):
            axis_aligned |= np.abs(edge_dir[:, ax]) > 0.98
        nan["axis_aligned_edge_frac"] = float(np.mean(axis_aligned))
    return nan
