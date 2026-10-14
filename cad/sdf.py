"""Small vectorised signed-distance-field (SDF) toolkit used by the mouse generator.

Convention: negative inside, positive outside, units in millimetres.
All functions accept numpy arrays that broadcast against each other.
"""
import numpy as np


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def bump(u):
    """Cosine window: 1 at u=0, 0 for |u|>=1, C1 continuous."""
    u = np.clip(np.abs(u), 0.0, 1.0)
    return 0.5 * (1.0 + np.cos(np.pi * u))


def plateau(x, a, b, ramp):
    """1 inside [a, b], smooth ramps of width `ramp` outside, 0 beyond."""
    return smoothstep(a - ramp, a, x) * (1.0 - smoothstep(b, b + ramp, x))


def smin(a, b, k):
    """Polynomial smooth union (blend width k)."""
    if k <= 0:
        return np.minimum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def smax(a, b, k):
    """Polynomial smooth intersection (blend width k); k may be an array."""
    k = np.asarray(k)
    if k.ndim == 0 and float(k) <= 0:
        return np.maximum(a, b)
    h = np.maximum(k - np.abs(a - b), 0.0) / np.maximum(k, 1e-9)
    return np.maximum(a, b) + h * h * k * 0.25


def slab(v, lo, hi):
    """Distance-like field of the slab lo <= v <= hi."""
    return np.maximum(lo - v, v - hi)


def sd_polygon(px, py, poly, chunk=64):
    """Exact signed distance to a closed 2D polygon (N,2), negative inside."""
    px = np.asarray(px, dtype=np.float64)
    py = np.asarray(py, dtype=np.float64)
    a = poly
    b = np.roll(poly, -1, axis=0)
    d2 = np.full(np.broadcast(px, py).shape, np.inf)
    inside = np.zeros(d2.shape, dtype=bool)
    for i0 in range(0, len(a), chunk):
        for (ax, ay), (bx, by) in zip(a[i0:i0 + chunk], b[i0:i0 + chunk]):
            ex, ey = bx - ax, by - ay
            wx, wy = px - ax, py - ay
            t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
            dx, dy = wx - ex * t, wy - ey * t
            d2 = np.minimum(d2, dx * dx + dy * dy)
            # even-odd crossing test
            c1 = (ay > py) != (by > py)
            xint = ax + (py - ay) * ex / np.where(ey == 0, 1e-12, ey)
            inside ^= c1 & (px < xint)
    d = np.sqrt(d2)
    return np.where(inside, -d, d)


def sd_roundbox2d(u, v, hu, hv, r):
    """Rounded rectangle centred at origin, half sizes hu, hv, corner radius r."""
    r = min(r, hu, hv)
    qx = np.abs(u) - (hu - r)
    qy = np.abs(v) - (hv - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def sd_segment2d(px, py, ax, ay, bx, by):
    ex, ey = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    t = np.clip((wx * ex + wy * ey) / (ex * ex + ey * ey), 0.0, 1.0)
    return np.hypot(wx - ex * t, wy - ey * t)


def rot2(u, v, deg):
    c, s = np.cos(np.radians(deg)), np.sin(np.radians(deg))
    return c * u + s * v, -s * u + c * v


def sd_rounded_cylinder_x(x, y, z, cx, cy, cz, radius, half_width, edge_r):
    """Cylinder with axis along X, rounded rims (a wheel)."""
    rr = np.hypot(y - cy, z - cz)
    dx = np.abs(x - cx) - (half_width - edge_r)
    dr = rr - (radius - edge_r)
    return np.minimum(np.maximum(dx, dr), 0) + np.hypot(np.maximum(dx, 0), np.maximum(dr, 0)) - edge_r


def sd_cylinder_z(x, y, z, cx, cy, r, z0, z1):
    d = np.hypot(x - cx, y - cy) - r
    return np.maximum(d, slab(z, z0, z1))
