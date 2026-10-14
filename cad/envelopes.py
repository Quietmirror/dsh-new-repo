"""Electronics reference envelopes for the assembly version + interference analysis against the V0.1 skeleton.

Every envelope carries a source tag (see params_v01.P["electronics"]):
  M measured from an open-source file, B from a reference BOM/README, S standard nominal, E ESTIMATE.
Estimated envelopes only reserve space; they are NOT assembly dimensions or hole positions.
"""
import numpy as np


class Env:
    def __init__(self, name, kind, group, src, color, **kw):
        self.name, self.kind, self.group, self.src, self.color = name, kind, group, src, color
        self.kw = kw

    # ---------------- geometry helpers
    def frame(self):
        R = np.array(self.kw.get("R", np.eye(3)), float)   # rows: local u, v, n axes in world coords
        return R

    def bbox(self):
        k = self.kw
        if self.kind == "box":
            R = self.frame()
            hs = np.array(k["size"]) / 2
            corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]) * hs
            w = corners @ R + np.array(k["center"])
            return w.min(0), w.max(0)
        if self.kind == "cyl":
            c, a = np.array(k["center"]), np.array(k["axis"], float)
            a /= np.linalg.norm(a)
            e = np.sqrt(np.clip(1 - a * a, 0, 1)) * k["r"] + np.abs(a) * k["length"] / 2
            return c - e, c + e
        if self.kind == "tube":
            p = np.array(k["points"])
            return p.min(0) - k["r"], p.max(0) + k["r"]

    def sdf(self, X, Y, Z):
        k = self.kw
        if self.kind == "box":
            R = self.frame()
            dx, dy, dz = X - k["center"][0], Y - k["center"][1], Z - k["center"][2]
            q = [np.abs(dx * R[i, 0] + dy * R[i, 1] + dz * R[i, 2]) - k["size"][i] / 2 for i in range(3)]
            outside = np.sqrt(sum(np.maximum(qi, 0) ** 2 for qi in q))
            inside = np.minimum(np.maximum(np.maximum(q[0], q[1]), q[2]), 0)
            return outside + inside
        if self.kind == "cyl":
            a = np.array(k["axis"], float)
            a /= np.linalg.norm(a)
            dx, dy, dz = X - k["center"][0], Y - k["center"][1], Z - k["center"][2]
            t = dx * a[0] + dy * a[1] + dz * a[2]
            r = np.sqrt(np.maximum(dx * dx + dy * dy + dz * dz - t * t, 0))
            qa, qr = np.abs(t) - k["length"] / 2, r - k["r"]
            return np.minimum(np.maximum(qa, qr), 0) + np.hypot(np.maximum(qa, 0), np.maximum(qr, 0))
        if self.kind == "tube":
            p = np.array(k["points"], float)
            d = np.full(np.broadcast(X, Y, Z).shape, 1e3)
            for a, b in zip(p[:-1], p[1:]):
                e = b - a
                wx, wy, wz = X - a[0], Y - a[1], Z - a[2]
                t = np.clip((wx * e[0] + wy * e[1] + wz * e[2]) / e.dot(e), 0, 1)
                d = np.minimum(d, np.sqrt((wx - e[0] * t) ** 2 + (wy - e[1] * t) ** 2 + (wz - e[2] * t) ** 2))
            return d - k["r"]

    def trimesh(self):
        import trimesh
        k = self.kw
        if self.kind == "box":
            T = np.eye(4)
            T[:3, :3] = self.frame().T
            T[:3, 3] = k["center"]
            return trimesh.creation.box(extents=k["size"], transform=T)
        if self.kind == "cyl":
            a = np.array(k["axis"], float)
            a /= np.linalg.norm(a)
            T = trimesh.geometry.align_vectors([0, 0, 1], a)
            T[:3, 3] = k["center"]
            return trimesh.creation.cylinder(radius=k["r"], height=k["length"], transform=T, sections=48)
        if self.kind == "tube":
            p = np.array(k["points"], float)
            parts = [trimesh.creation.icosphere(2, k["r"]).apply_translation(q) for q in p]
            for a, b in zip(p[:-1], p[1:]):
                parts.append(trimesh.creation.cylinder(radius=k["r"], segment=[a, b], sections=24))
            return trimesh.util.concatenate(parts)

    def cadquery(self):
        import cadquery as cq
        k = self.kw
        if self.kind == "box":
            s = k["size"]
            solid = cq.Solid.makeBox(s[0], s[1], s[2], pnt=cq.Vector(-s[0] / 2, -s[1] / 2, -s[2] / 2))
            R = self.frame().T
            M = cq.Matrix([[R[0, 0], R[0, 1], R[0, 2], k["center"][0]],
                           [R[1, 0], R[1, 1], R[1, 2], k["center"][1]],
                           [R[2, 0], R[2, 1], R[2, 2], k["center"][2]]])
            return solid.transformGeometry(M) if not np.allclose(R, np.eye(3)) else solid.translate(cq.Vector(*k["center"]))
        if self.kind == "cyl":
            a = np.array(k["axis"], float)
            a /= np.linalg.norm(a)
            base = np.array(k["center"]) - a * k["length"] / 2
            return cq.Solid.makeCylinder(k["r"], k["length"], cq.Vector(*base), cq.Vector(*a))
        if self.kind == "tube":
            p = np.array(k["points"], float)
            sol = None
            for a, b in zip(p[:-1], p[1:]):
                d = b - a
                c = cq.Solid.makeCylinder(k["r"], float(np.linalg.norm(d)), cq.Vector(*a), cq.Vector(*(d / np.linalg.norm(d))))
                sol = c if sol is None else sol.fuse(c)
            for q in p:
                sol = sol.fuse(cq.Solid.makeSphere(k["r"], cq.Vector(*q)))
            return sol.clean()


COL = {
    "sensor": (0.85, 0.25, 0.25), "lens": (0.95, 0.55, 0.2), "board": (0.2, 0.55, 0.3), "switch": (0.25, 0.35, 0.85),
    "wheel": (0.45, 0.45, 0.5), "battery": (0.9, 0.75, 0.15), "usb": (0.55, 0.2, 0.7), "wire": (0.95, 0.35, 0.65),
    "antenna": (0.2, 0.75, 0.85), "power": (0.6, 0.4, 0.2),
}


def frame_from(u, n):
    u = np.array(u, float)
    n = np.array(n, float)
    n /= np.linalg.norm(n)
    u = u - u.dot(n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.array([u, v, n])


def layout(g, battery="301230"):
    """Place estimated envelopes in mouse coordinates using geometry from the built trial grip."""
    P = g.P
    E = P["electronics"]
    val = lambda k: E[k][0]
    src = lambda k: E[k][1]
    L = P["size"]["length"]
    sen = P["bottom"]["sensor"]
    sx, sy = sen["x"], sen["y"]
    z_surf = -val("skate_thickness")
    z_lens = z_surf + val("lens_to_surface")
    z_pb = z_lens + val("lens_ref_to_pcb_bottom")
    z_pt = z_pb + val("pcb_thickness")
    chain = {"mousepad_surface_z": z_surf, "lens_reference_plane_z": z_lens, "pcb_bottom_z": z_pb, "pcb_top_z": z_pt}
    envs = []
    add = lambda *a, **k: envs.append(Env(*a, **k))

    # sensor + lens (optical axis vertical, through (sx, sy))
    cx0, cx1, cy0, cy1 = val("lens_courtyard")
    add("lens_courtyard", "box", "sensor", src("lens_courtyard"), COL["lens"],
        center=(sx + (cx0 + cx1) / 2, sy + (cy0 + cy1) / 2, (z_lens + z_pb) / 2), size=(cx1 - cx0, cy1 - cy0, z_pb - z_lens))
    sb = val("sensor_body")
    add("sensor_PAW3950_body", "box", "sensor", src("sensor_body"), COL["sensor"],
        center=(sx, sy, z_pt + sb[2] / 2), size=sb)
    bb = val("breakout_board")
    add("ref_paw3395_breakout_pcb", "box", "sensor", src("breakout_board"), COL["board"],
        center=(sx, sy + bb[3], (z_pb + z_pt) / 2), size=(bb[0], bb[1], val("pcb_thickness")))
    add("sensor_optical_axis", "cyl", "sensor", "axis marker (vertical, perpendicular to the pad)", COL["sensor"],
        center=(sx, sy, (z_surf + 12) / 2), axis=(0, 0, 1), r=0.3, length=12 - z_surf)

    # main switches under the fingertip dishes
    ms = val("main_switch")
    for key in ("lmb", "rmb"):
        c = P["contact"][key]
        add(f"switch_{key.upper()}", "box", "switch", src("main_switch"), COL["switch"],
            center=(c["x"], c["y"], z_pt + ms[2] / 2), size=ms)

    # wheel, axle, encoder, middle click
    wx, wy, wz = g.wheel_center
    wh = P["wheel"]
    add("wheel", "cyl", "wheel", "E: same as V0.1 dummy wheel (D%.0f x %.0f)" % (2 * wh["radius"], wh["width"]), COL["wheel"],
        center=(wx, wy, wz), axis=(1, 0, 0), r=wh["radius"], length=wh["width"])
    ad, al = val("wheel_axle")
    add("wheel_axle", "cyl", "wheel", src("wheel_axle"), COL["wheel"], center=(wx - 2.0, wy, wz), axis=(1, 0, 0), r=ad / 2, length=al)
    en = val("encoder")
    ex1 = wx - wh["width"] / 2 - 1.0
    top = wz + en[2] / 2
    add("encoder", "box", "wheel", src("encoder"), COL["wheel"],
        center=(ex1 - en[0] / 2, wy, (z_pt + top) / 2), size=(en[0], en[1], top - z_pt))
    mm = val("mmb_switch")
    add("switch_MMB", "box", "switch", src("mmb_switch"), COL["switch"],
        center=(wx, wy - wh["radius"] - 1.5, z_pt + mm[2] / 2), size=mm)

    # side buttons: switches + small PCB parallel to the thumb-side wall
    t = P["shell"]["wall"]
    ss, sbd = val("side_switch"), val("side_board")
    fr = g.boss_frames
    nmean = np.mean([f["n"] for f in fr], axis=0)
    u = fr[0]["p0"] - fr[-1]["p0"]
    R = frame_from(u, nmean)
    n = R[2]
    gap = 0.5   # actuator gap to the inner face of the boss
    for f in fr:
        Rl = frame_from(u, f["n"])
        add(f"switch_{f['name']}", "box", "switch", src("side_switch"), COL["switch"],
            center=tuple(f["p0"] - f["n"] * (t + gap + ss[2] / 2)), size=(ss[0], ss[1], ss[2]), R=Rl)
    pm = np.mean([f["p0"] for f in fr], axis=0)
    pb = pm - n * (t + gap + ss[2] + sbd[2] / 2 + 0.6)   # +0.6: curvature between the two switch normals
    add("side_button_pcb", "box", "board", src("side_board"), COL["board"], center=tuple(pb), size=sbd, R=R)

    # battery + plug (rear, flat on the skeleton)
    bat = E["battery"][battery]
    bl, bw, bt = bat[0]
    by = 22.0 if bt <= 3.5 else 21.0
    zb = P["bottom"]["rib_height"] + 0.5      # on the grid with ~0.5 mm foam tape
    add(f"battery_{battery}", "box", "battery", bat[1], COL["battery"], center=(0.0, by, zb + bt / 2), size=(bl, bw, bt))
    pl = val("battery_plug")
    add("battery_plug", "box", "battery", src("battery_plug"), COL["battery"],
        center=(9.0, by + bw / 2 + 0.5 + pl[1] / 2, z_pt + pl[2] / 2), size=pl)

    # USB-C at the front centre, receptacle mid-mounted in the main board
    uc = val("usb_c")
    uy1 = L - 0.3
    add("usb_c_receptacle", "box", "usb", src("usb_c"), COL["usb"],
        center=(0.0, uy1 - uc[1] / 2, z_pt - 1.0 + uc[2] / 2), size=uc, needs_cutout=True)
    om = val("usb_c_plug_overmold")
    add("usb_c_plug_overmold_clearance", "box", "usb", src("usb_c_plug_overmold"), COL["usb"],
        center=(0.0, L + 0.6 + om[1] / 2, z_pt - 1.0 + uc[2] / 2), size=om)

    # power switch under the main board, operated through the open bottom
    ps = val("power_switch")
    add("power_switch", "box", "power", src("power_switch"), COL["power"],
        center=(-17.0, 36.0, z_pb - ps[2] / 2), size=ps)

    # 2.4 GHz antenna keep-out on the main board (front-right corner, away from battery and encoder)
    ak = val("antenna_keepout")
    add("antenna_keepout_2G4", "box", "antenna", src("antenna_keepout"), COL["antenna"],
        center=(18.0, 62.0, z_pb + ak[2] / 2), size=ak, plastic_ok=True)

    # wiring
    wr = val("wire_bundle_d") / 2
    p_side = pb - n * 1.5 - R[1] * (sbd[1] / 2)
    add("wire_side_buttons", "tube", "wire", src("wire_bundle_d"), COL["wire"],
        points=[tuple(p_side), (p_side[0] + 4.0, p_side[1], z_pt + 4.0), (-12.0, p_side[1], z_pt + wr + 0.2)], r=wr)
    add("wire_battery", "tube", "wire", src("wire_bundle_d"), COL["wire"],
        points=[(bl / 2 - 3.0, by + bw / 2, zb + bt / 2), (9.0, by + bw / 2 + 0.5, z_pt + pl[2] / 2)], r=wr)
    return envs, chain


def board_keepin(g, F, z0, z1, clearance=0.5):
    """Largest XY region free for a main board occupying z0..z1 (placeholders removed).

    The board rests on the grid top (z0); everything solid between z0+0.25 and z1 is an obstacle,
    and the clearance is applied laterally (2D distance from the projected obstacles)."""
    from scipy import ndimage
    from skimage import measure
    kz = (g.zs >= z0 + 0.25) & (g.zs <= z1)
    obst = (F[:, :, kz] < 0).any(axis=2)
    inside = (g.s[:, :, kz] < 0).all(axis=2)
    dist = ndimage.distance_transform_edt(~obst, sampling=g.h)
    ok = inside & (dist > clearance)
    lab = measure.label(ok)
    if lab.max() == 0:
        return None, ok
    big = np.argmax(np.bincount(lab.ravel())[1:]) + 1
    reg = lab == big
    cs = measure.find_contours(reg.astype(float), 0.5)
    polys = []
    for c in cs:
        x = np.interp(c[:, 0], np.arange(len(g.xs)), g.xs)
        y = np.interp(c[:, 1], np.arange(len(g.ys)), g.ys)
        polys.append(np.column_stack([x, y]))
    polys.sort(key=lambda p: -len(p))
    return polys, reg


def analyze(g, envs, F, chain, clearance=0.5):
    """Overlap / clearance of each envelope vs the skeleton (placeholders removed) and between envelopes."""
    h = g.h
    res, masks = {}, {}
    for e in envs:
        lo, hi = e.bbox()
        ix = np.nonzero((g.xs >= lo[0] - 1) & (g.xs <= hi[0] + 1))[0]
        iy = np.nonzero((g.ys >= lo[1] - 1) & (g.ys <= hi[1] + 1))[0]
        iz = np.nonzero((g.zs >= lo[2] - 1) & (g.zs <= hi[2] + 1))[0]
        r = {"group": e.group, "source": e.src, "bbox_min": np.round(lo, 2).tolist(), "bbox_max": np.round(hi, 2).tolist()}
        if len(ix) and len(iy) and len(iz):
            X, Y, Z = np.meshgrid(g.xs[ix], g.ys[iy], g.zs[iz], indexing="ij")
            d = e.sdf(X, Y, Z)
            m = d < 0
            sub = F[np.ix_(ix, iy, iz)]
            subs = g.s[np.ix_(ix, iy, iz)]
            ov = m & (sub < 0)
            r["overlap_with_skeleton_mm3"] = float(ov.sum() * h ** 3)
            r["max_penetration_mm"] = float(-sub[ov].min()) if ov.any() else 0.0
            r["min_clearance_to_skeleton_mm"] = float(sub[m].min()) if m.any() else None
            r["volume_outside_skin_mm3"] = float((m & (subs > 0)).sum() * h ** 3)
            if ov.any():
                pts = np.column_stack([X[ov], Y[ov], Z[ov]])
                r["overlap_region_bbox"] = [np.round(pts.min(0), 1).tolist(), np.round(pts.max(0), 1).tolist()]
            masks[e.name] = (ix, iy, iz, m)
        r["min_z"] = float(lo[2])
        r["below_ground_clearance"] = bool(lo[2] < 0.5 and e.group not in ("sensor",) and not e.name.startswith("usb_c_plug"))
        st = "ok"
        if r.get("overlap_with_skeleton_mm3", 0) > h ** 3 * 2:
            if e.kw.get("needs_cutout"):
                st = "NEEDS_SHELL_CUTOUT"
            elif r["max_penetration_mm"] <= 0.3:
                st = "CONTACT(<=0.3mm, resting/fillet)"
            else:
                st = "INTERFERENCE"
        elif r.get("min_clearance_to_skeleton_mm") is not None and r["min_clearance_to_skeleton_mm"] < clearance:
            st = "TIGHT"
        if e.kw.get("plastic_ok") and st != "ok":
            st = "ok (only plastic inside; keep metal, battery and wires out)"
        if r["below_ground_clearance"]:
            st += "+BELOW_0.5mm"
        r["status"] = st
        res[e.name] = r
    # pairwise
    intended = {frozenset(p) for p in [("wheel", "wheel_axle"), ("wheel_axle", "encoder"), ("sensor_PAW3950_body", "ref_paw3395_breakout_pcb"),
                                       ("lens_courtyard", "sensor_optical_axis"), ("sensor_PAW3950_body", "sensor_optical_axis"),
                                       ("ref_paw3395_breakout_pcb", "sensor_optical_axis"), ("wire_battery", "battery_plug"),
                                       ("wire_battery", "battery_301230"), ("wire_battery", "battery_602030"),
                                       ("battery_301230", "battery_602030"),   # alternatives, never both
                                       ("wire_side_buttons", "side_button_pcb"), ("usb_c_receptacle", "usb_c_plug_overmold_clearance")]}
    pairs = []
    names = list(masks)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if frozenset((a, b)) in intended:
                continue
            ia, ja, ka, ma = masks[a]
            ib, jb, kb, mb = masks[b]
            cx, xa, xb = np.intersect1d(ia, ib, return_indices=True)
            cy, ya, yb = np.intersect1d(ja, jb, return_indices=True)
            cz, za, zb = np.intersect1d(ka, kb, return_indices=True)
            if len(cx) == 0 or len(cy) == 0 or len(cz) == 0:
                continue
            ov = ma[np.ix_(xa, ya, za)] & mb[np.ix_(xb, yb, zb)]
            if ov.sum() > 2:
                pairs.append({"a": a, "b": b, "overlap_mm3": float(ov.sum() * h ** 3)})
    return {"z_chain": chain, "envelopes": res, "envelope_pair_overlaps": pairs}


def export_step(envs, path):
    import cadquery as cq
    asm = cq.Assembly(name="fingertip_v01_electronics_envelopes")
    for e in envs:
        asm.add(e.cadquery(), name=e.name, color=cq.Color(*e.color))
    asm.save(path)  # STEP inferred from extension
    return path
