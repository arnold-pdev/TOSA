"""
voxel2surf — occupancy corridor for the implicit fairing solver.

Binary voxels carry no sub-voxel geometry, so the only honest anchor for the
smoothed surface is *occupancy consistency*: it must stay near the cuberille
boundary — neither drifting off it nor collapsing a thin member. We express that
as a per-vertex axis-aligned box ``[lo, hi]`` the implicit solver may not leave.

A *uniform* radial clamp can't work (proven empirically and in the obstacle-problem
literature): tight enough to protect a 1-voxel wall (<½ voxel) it blocks all
smoothing; loose enough to smooth it gives no protection. The fix — Gibson's
Constrained Elastic Surface Nets containment, generalized from cell-centered nodes
to our corner vertices — is an **asymmetric, thickness-adaptive** corridor built
from a regularized signed-distance transform of the binary:

  - the *inward* move (toward solid) is capped at the local half-thickness, so the
    two walls of a thin member can't meet (this is the anti-collapse / anti-shrink
    lever — the binding constraint);
  - *outward* and *tangential* moves stay generous, so de-staircasing is free.

The binary SDT is severely quantized, so it is Gaussian-regularized before the
gradient (normal) and bounds are read off it.
"""

from __future__ import annotations

import numpy as np


def _trilin(field, u, *, grad=False):
    """Trilinear sample of a cell-centered (nx,ny,nz) field at index coords ``u``
    (V,3). Returns ``val`` (V,), or ``(val, gidx)`` with index-space gradient."""
    field = np.asarray(field, dtype=float)
    n = np.asarray(field.shape)
    base = np.clip(np.floor(u).astype(int), 0, n - 2)
    f = np.clip(u - base, 0.0, 1.0)
    b0, b1, b2 = base[:, 0], base[:, 1], base[:, 2]

    def g(dx, dy, dz):
        return field[b0 + dx, b1 + dy, b2 + dz]

    c000, c100, c010, c110 = g(0, 0, 0), g(1, 0, 0), g(0, 1, 0), g(1, 1, 0)
    c001, c101, c011, c111 = g(0, 0, 1), g(1, 0, 1), g(0, 1, 1), g(1, 1, 1)
    x, y, z = f[:, 0], f[:, 1], f[:, 2]
    x0, y0, z0 = 1 - x, 1 - y, 1 - z
    val = (c000 * x0 * y0 * z0 + c100 * x * y0 * z0 + c010 * x0 * y * z0 + c110 * x * y * z0
           + c001 * x0 * y0 * z + c101 * x * y0 * z + c011 * x0 * y * z + c111 * x * y * z)
    if not grad:
        return val
    gx = (c100 - c000) * y0 * z0 + (c110 - c010) * y * z0 + (c101 - c001) * y0 * z + (c111 - c011) * y * z
    gy = (c010 - c000) * x0 * z0 + (c110 - c100) * x * z0 + (c011 - c001) * x0 * z + (c111 - c101) * x * z
    gz = (c001 - c000) * x0 * y0 + (c101 - c100) * x * y0 + (c011 - c010) * x0 * y + (c111 - c110) * x * y
    return val, np.stack([gx, gy, gz], 1)


def derive_bounds(verts, vox, origin, spacing, *,
                  normal_voxels: float = 0.6, tangential_voxels: float = 1.5,
                  inward_margin: float = 0.5, min_inward: float = 0.1, sigma: float = 0.75):
    """Per-vertex axis-aligned corridor ``[lo, hi]`` (each V×3).

    The surface must stay *near the occupancy boundary*, so the **normal** budget
    (along ≈ the surface normal) is small — ``normal_voxels`` — which is the
    anti-shrink lever: it caps how far the surface migrates off the cuberille
    boundary toward the medial axis. The *inward* side is additionally tightened to
    the local half-thickness minus ``inward_margin`` (anti-collapse at thin members,
    via the regularized binary SDT). The **tangential** budget ``tangential_voxels``
    stays generous so vertices can redistribute / de-staircase in-plane. ``sigma``
    regularizes the quantized binary SDT before normals/thickness are read.
    """
    from scipy.ndimage import distance_transform_edt, gaussian_filter

    verts = np.asarray(verts, dtype=float)
    origin = np.asarray(origin, dtype=float)
    spacing = np.asarray(spacing, dtype=float)
    solid = np.asarray(vox) > 0
    s = float(np.min(spacing))

    d_in = distance_transform_edt(solid)          # voxels into solid → boundary distance
    d_out = distance_transform_edt(~solid)
    phi = gaussian_filter((d_out - d_in).astype(float), sigma)  # +void / −solid, regularized

    u = (verts - origin) / spacing - 0.5          # node → cell-center index coords
    _, gidx = _trilin(phi, u, grad=True)
    grad = gidx / spacing
    nout = grad / np.maximum(np.linalg.norm(grad, axis=1, keepdims=True), 1e-12)  # outward normal

    # local half-thickness: running max of d_in along a short inward ray
    ht = _trilin(d_in, u)
    x = verts.copy()
    for _ in range(int(np.ceil(normal_voxels / 0.25)) + 4):
        x = x - 0.25 * spacing * nout
        ht = np.maximum(ht, _trilin(d_in, (x - origin) / spacing - 0.5))

    i_w = np.clip(ht - inward_margin, min_inward, normal_voxels) * s  # inward: small + thin-adaptive
    o_w = normal_voxels * s                                           # outward: small (limit bulge)
    t_w = tangential_voxels * s                                       # tangential: generous

    # tangential generous on all axes, then override the dominant (≈ normal) axis
    # with the small asymmetric [−i_w, +o_w] — the anti-shrink / anti-collapse lever.
    lo = verts - t_w
    hi = verts + t_w
    idx = np.arange(len(verts))
    a = np.argmax(np.abs(nout), axis=1)
    pos = nout[idx, a] >= 0                        # outward points +axis?
    val = verts[idx, a]
    hi[idx, a] = np.where(pos, val + o_w, val + i_w)
    lo[idx, a] = np.where(pos, val - i_w, val - o_w)
    return lo, hi
