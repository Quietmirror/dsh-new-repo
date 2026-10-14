"""Manufacturability / geometry checks for the generated trial grip."""
import numpy as np
from scipy.interpolate import RegularGridInterpolator


def mesh_checks(mesh):
    from trimesh.grouping import group_rows
    e = mesh.edges_sorted
    _, cnt = np.unique(e, axis=0, return_counts=True)
    parts = mesh.split(only_watertight=False)
    out = {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "watertight": bool(mesh.is_watertight),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "bodies": int(len(parts)),
        "boundary_edges": int(len(group_rows(e, require_count=1))),
        "non_manifold_edges": int((cnt > 2).sum()),
        "degenerate_faces": int((mesh.area_faces < 1e-12).sum()),
        "euler_number": int(mesh.euler_number),
        "volume_mm3": float(mesh.volume),
        "area_mm2": float(mesh.area),
        "bounds": mesh.bounds.tolist(),
    }
    out["self_intersecting_faces"] = self_intersections(mesh)
    try:
        import manifold3d as mf
        man = mf.Manifold(mf.Mesh(vert_properties=np.asarray(mesh.vertices, np.float32),
                                  tri_verts=np.asarray(mesh.faces, np.uint32)))
        out["manifold3d_status"] = str(man.status())
        out["manifold3d_volume_mm3"] = float(man.volume())
    except Exception as ex:  # pragma: no cover
        out["manifold3d_status"] = f"error: {ex}"
    return out


def self_intersections(mesh):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, np.float64),
                               face_matrix=np.asarray(mesh.faces, np.int32)))
    ms.compute_selection_by_self_intersections_per_face()
    return int(ms.current_mesh().selected_face_number())


def pad_checks(g, mesh, skate_d=7.5):
    """Five skate platforms: bottom faces present, exactly coplanar, each holds a full skate disc."""
    P = g.P
    rp = P["bottom"]["pad_d"] / 2
    c = mesh.triangles_center
    n = mesh.face_normals
    bottom = (n[:, 2] < -0.9999) & (np.abs(c[:, 2]) < 1e-3)
    Fi = RegularGridInterpolator((g.xs, g.ys, g.zs), g.F, bounds_error=False, fill_value=1.0)
    res = {}
    allz = []
    for name, (px, py) in g.pads.items():
        r = np.hypot(c[:, 0] - px, c[:, 1] - py)
        sel = bottom & (r < rp - 0.2)
        vz = mesh.vertices[mesh.faces[sel]].reshape(-1, 3)[:, 2]
        allz.append(vz)
        # does a full skate disc fit on the flat bottom face around the pad centre?
        th = np.linspace(0, 2 * np.pi, 72, endpoint=False)
        pts = [(px + rr * np.cos(a), py + rr * np.sin(a), 0.5 * g.h) for rr in (0.0, skate_d / 4, skate_d / 2) for a in th]
        inside = Fi(np.array(pts)) < 0
        res[name] = {
            "center_xy": [round(px, 3), round(py, 3)],
            "bottom_face_area_mm2_within_pad": float(mesh.area_faces[sel].sum()),
            "z_min": float(vz.min()), "z_max": float(vz.max()),
            f"skate_{skate_d}mm_disc_fully_supported": bool(inside.all()),
        }
    allz = np.concatenate(allz)
    res["coplanarity_max_dev_mm"] = float(allz.max() - allz.min())
    res["global_min_z"] = float(mesh.vertices[:, 2].min())
    res["pad_plane_z"] = 0.0
    return res


def ray_thickness(g, mesh, step=0.04, max_len=8.0, chunk=40000):
    """Wall thickness at every vertex: march along the inward normal through the solid field F
    (trilinear) until it exits; the exit is refined linearly -> sub-voxel accuracy.
    Rays that do not exit within max_len (pads, bosses seen end-on) return max_len."""
    Fi = RegularGridInterpolator((g.xs, g.ys, g.zs), g.F, bounds_error=False, fill_value=1.0)
    v = np.asarray(mesh.vertices)
    n = np.asarray(mesh.vertex_normals)
    out = np.full(len(v), max_len)
    ks = np.arange(1, int(max_len / step) + 1) * step
    for a in range(0, len(v), chunk):
        vv, nn = v[a:a + chunk], n[a:a + chunk]
        prev = np.full(len(vv), -1e-3)
        done = np.zeros(len(vv), bool)
        res = np.full(len(vv), max_len)
        for t in ks:
            f = Fi(vv - nn * t)
            hit = (~done) & (f > 0) & (t > 0.1)
            if hit.any():
                res[hit] = t - step + step * (-prev[hit]) / (f[hit] - prev[hit])
                done |= hit
            prev = np.where(f <= 0, f, prev)
            if done.all():
                break
        out[a:a + chunk] = res
    return out


def thickness_report(g, mesh, th, thin=1.4, edge_band=1.5):
    v = mesh.vertices
    Si = RegularGridInterpolator((g.xs, g.ys, g.zs), g.s, bounds_error=False, fill_value=10.0)
    Wi = RegularGridInterpolator((g.xs, g.ys, g.zs), np.minimum(np.minimum(g.W, g.Wwheel), 1e3), bounds_error=False, fill_value=1e3)
    sv = Si(v)
    outer = np.abs(sv) < 0.25                     # vertex on the outer skin
    inner = np.abs(sv + g.P["shell"]["wall"]) < 0.25
    near_edge = np.abs(Wi(v)) < edge_band          # on/near a rounded window or wheel-opening edge
    shell_pts = (outer | inner) & (v[:, 2] > 3.5)
    def stats(x):
        return {"n": int(len(x)), "min": float(x.min()), "p1": float(np.percentile(x, 1)), "p5": float(np.percentile(x, 5)),
                "median": float(np.median(x))}
    body = shell_pts & ~near_edge
    thin_pts = v[body & (th < thin)]
    clusters = []
    if len(thin_pts):
        from scipy.cluster.hierarchy import fcluster, linkage
        idx = np.nonzero(body & (th < thin))[0]
        if len(idx) > 1:
            lab = fcluster(linkage(v[idx], "single"), 2.0, "distance")
        else:
            lab = np.ones(1, int)
        for k in np.unique(lab):
            ii = idx[lab == k]
            clusters.append({"vertices": int(len(ii)), "min_mm": float(th[ii].min()), "at_xyz": np.round(v[ii].mean(0), 1).tolist()})
        clusters.sort(key=lambda c: c["min_mm"])
    edge = shell_pts & near_edge
    return {
        "method": "inward-normal ray through the SDF (trilinear, sub-voxel), voxel %.2f mm" % g.h,
        "design_wall_mm": g.P["shell"]["wall"],
        "shell_wall_excluding_window_edges": stats(th[body]),
        "rounded_window_and_wheel_edges": stats(th[edge]) if edge.any() else None,
        f"thin_spots_below_{thin}mm_excluding_edges": clusters[:15],
    }


def wheel_gap(g):
    """Minimum clearance between the fixed dummy wheel disc and the shell (opening edges)."""
    if not hasattr(g, "wheel_disc"):
        return None
    m = g.wheel_disc < 0
    return float(g.shell_final[m].min())


def keepout_checks(g, margin, wheel_margin):
    """Distance from each finger contact zone (sampled on the skin) to the nearest window / wheel opening."""
    P = g.P
    L = P["size"]["length"]
    sk = g.skin_verts[g.skin_verts[:, 2] > 0.3]
    x, y, z = sk[:, 0], sk[:, 1], sk[:, 2]
    c = P["contact"]
    lp = P["lip"]
    zones = {}
    for key in ("lmb", "rmb"):
        d = c[key]
        zones[key + "_dish"] = (np.hypot((x - d["x"]) / d["rx"], (y - d["y"]) / d["ry"]) < 1.0) & (z > 12)
    t = c["thumb"]
    zones["thumb_groove"] = (x < -10) & (np.abs(y - t["y"]) < t["ay"]) & (np.abs(z - t["z"]) < t["az"])
    r = c["ring"]
    zones["ring_rest"] = (x > 10) & (np.abs(y - r["y"]) < r["ay"]) & (np.abs(z - r["z"]) < r["az"])
    zones["pinky_lip"] = (x > 10) & (y > lp["right_y0"] * L) & (y < lp["right_y1"] * L) & (z < lp["right_z"])
    for bf, sb in zip(g.boss_frames, P["side_buttons"]):
        d = sk - bf["p0"]
        du, dv = d @ bf["u"], d @ bf["v"]
        zones["boss_" + sb["name"]] = (np.abs(du) < sb["length"] / 2 + 1.5) & (np.abs(dv) < sb["height"] / 2 + 1.5) & (np.abs(d @ bf["n"]) < 5)
    Wi = RegularGridInterpolator((g.xs, g.ys, g.zs), np.minimum(g.W, 1e3), bounds_error=False, fill_value=1e3)
    Ww = RegularGridInterpolator((g.xs, g.ys, g.zs), g.Wwheel, bounds_error=False, fill_value=1e3)
    out = {}
    for name, msk in zones.items():
        pts = sk[msk]
        if len(pts) == 0:
            out[name] = {"samples": 0}
            continue
        dw = Wi(pts)
        dwh = Ww(pts)
        out[name] = {"samples": int(len(pts)), "min_dist_to_window_mm": float(dw.min()),
                     "min_dist_to_wheel_opening_mm": float(dwh.min()),
                     "ok": bool(dw.min() >= margin and dwh.min() >= wheel_margin)}
    return out


def overhang_report(mesh, angle_deg=45.0):
    """Faces that face downward more steeply than the overhang limit (excluding the bed contact)."""
    n = mesh.face_normals
    c = mesh.triangles_center
    lim = -np.cos(np.radians(angle_deg))
    on_bed = (np.abs(c[:, 2]) < 1e-3) & (n[:, 2] < -0.999)
    first_layer = c[:, 2] < 0.8          # first layers + the 0.6 mm 45-deg bottom-edge chamfer print without support
    over = (n[:, 2] < lim) & ~on_bed & ~first_layer
    a = mesh.area_faces
    zc = c[over, 2]
    return over, {
        "overhang_limit_deg_from_vertical": angle_deg,
        "excluded": "faces below Z=0.8 (first layers and the 0.6 mm 45-deg bottom chamfer)",
        "overhang_area_mm2": float(a[over].sum()),
        "bed_contact_area_mm2": float(a[on_bed].sum()),
        "overhang_z_range": [float(zc.min()), float(zc.max())] if over.any() else None,
        "overhang_area_by_z_band_mm2": {f"{lo}-{lo+5}": float(a[over & (c[:, 2] >= lo) & (c[:, 2] < lo + 5)].sum()) for lo in range(0, 35, 5)},
    }
