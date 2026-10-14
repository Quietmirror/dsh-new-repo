"""NURBS reference skin: loft through Y-sections of the outer skin -> STEP (B-rep, editable in CAD).

This is the closed outer body (skin + flat bottom at the pad plane) WITHOUT windows, skeleton or side-button
bosses. It is meant as a master surface for the assembly version (shell / split / cut in Fusion, SolidWorks...).
The printable trial grip is the STL, not this body.
"""
import numpy as np
from skimage.measure import find_contours


def section_curve(g, y, n=80):
    h = g.h
    iy = int(np.clip(np.searchsorted(g.ys, y) - 1, 0, len(g.ys) - 2))
    t = (y - g.ys[iy]) / h
    S2 = (1 - t) * g.s[:, iy, :] + t * g.s[:, iy + 1, :]
    cs = find_contours(S2, 0.0)
    if not cs:
        return None
    c = max(cs, key=len)
    x = np.interp(c[:, 0], np.arange(len(g.xs)), g.xs)
    z = np.interp(c[:, 1], np.arange(len(g.zs)), g.zs)
    # clip to z >= 0 (keep the arc above the pad plane)
    above = z >= 0
    if above.all() or not above.any():
        return None
    k = np.nonzero(above)[0]
    i0, i1 = k[0], k[-1]
    pts = []
    if i0 > 0:
        a = (0 - z[i0 - 1]) / (z[i0] - z[i0 - 1])
        pts.append((x[i0 - 1] + a * (x[i0] - x[i0 - 1]), 0.0))
    pts += list(zip(x[i0:i1 + 1], z[i0:i1 + 1]))
    if i1 < len(z) - 1:
        a = (0 - z[i1]) / (z[i1 + 1] - z[i1])
        pts.append((x[i1] + a * (x[i1 + 1] - x[i1]), 0.0))
    p = np.array(pts)
    if p[0, 0] > p[-1, 0]:
        p = p[::-1]
    # arc-length resampling with denser points near the bottom corners
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(p, axis=0).T))]
    u = (1 - np.cos(np.linspace(0, np.pi, n))) / 2 * d[-1]
    xs = np.interp(u, d, p[:, 0])
    zs = np.interp(u, d, p[:, 1])
    zs[0] = zs[-1] = 0.0
    return np.column_stack([xs, np.full(n, y), zs])


def build_loft(g, nsec=56, end_gap=0.25):
    import cadquery as cq
    sk = g.skin_verts[g.skin_verts[:, 2] >= 0]
    y0, y1 = sk[:, 1].min() + end_gap, sk[:, 1].max() - end_gap
    th = np.linspace(0, np.pi, nsec)
    ys = y0 + (y1 - y0) * (1 - np.cos(th)) / 2
    wires = []
    used = []
    for y in ys:
        p = section_curve(g, y)
        if p is None or np.ptp(p[:, 0]) < 1.0:
            continue
        sp = cq.Edge.makeSpline([cq.Vector(*q) for q in p])
        ln = cq.Edge.makeLine(cq.Vector(*p[-1]), cq.Vector(*p[0]))
        wires.append(cq.Wire.assembleEdges([sp, ln]))
        used.append(float(y))
    solid = cq.Solid.makeLoft(wires, ruled=False)
    return solid, used


def deviation(g, solid, nsample=3000):
    """Exact point-to-surface distance (OCC projection onto the B-spline faces) for skin sample points."""
    from OCP.BRep import BRep_Tool
    from OCP.GeomAPI import GeomAPI_ProjectPointOnSurf
    from OCP.gp import gp_Pnt
    surfs = [BRep_Tool.Surface_s(f.wrapped) for f in solid.Faces() if f.geomType() == "BSPLINE"]
    ys = [v.Y for v in solid.Vertices()]
    ylo, yhi = min(ys), max(ys)
    sk = g.skin_verts[(g.skin_verts[:, 2] > 0.3) & (g.skin_verts[:, 1] > ylo + 0.05) & (g.skin_verts[:, 1] < yhi - 0.05)]
    rng = np.random.default_rng(0)
    pts = sk[rng.choice(len(sk), min(nsample, len(sk)), replace=False)]
    d = np.full(len(pts), np.inf)
    for i, p in enumerate(pts):
        for S in surfs:
            pr = GeomAPI_ProjectPointOnSurf(gp_Pnt(*map(float, p)), S)
            if pr.NbPoints():
                d[i] = min(d[i], pr.LowerDistance())
    d = d[np.isfinite(d)]
    worst = pts[np.argsort(-d)[:3]] if len(d) else []
    return {"samples": int(len(d)), "max_mm": float(d.max()), "p99_mm": float(np.percentile(d, 99)),
            "mean_mm": float(d.mean()), "worst_at": np.round(worst, 1).tolist(),
            "note": "skin beyond the first/last section (small flat end caps) excluded"}


def export(g, path):
    import cadquery as cq
    solid, ys = build_loft(g)
    valid = solid.isValid()
    cq.exporters.export(cq.Workplane().add(solid), path)
    dev = deviation(g, solid)
    return {"file": path, "sections": len(ys), "y_range": [ys[0], ys[-1]], "valid_brep": bool(valid),
            "volume_mm3": float(solid.Volume()), "deviation_vs_skin": dev}
