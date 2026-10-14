"""Where does the slicer put support material?  Classifies support extrusions as inside the shell
(removed through the open bottom / windows) or outside the outer skin (would leave marks on grip surfaces).

    python3 cad/support_check.py out/v0.1/fingertip_v01_trial_grip.stl out/v0.1/slice_with_supports.gcode
"""
import json
import os
import re
import sys

import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely import contains_xy


def support_points(gcode):
    pts, z, typ = [], 0.0, ""
    x = y = 0.0
    for line in open(gcode, errors="ignore"):
        if line.startswith(";TYPE:"):
            typ = line[6:].strip()
            continue
        if line.startswith(";Z:"):
            z = float(line[3:])
            continue
        if line.startswith("G1"):
            mx = re.search(r"X([-\d.]+)", line)
            my = re.search(r"Y([-\d.]+)", line)
            me = re.search(r"E([-\d.]+)", line)
            if mx:
                x = float(mx.group(1))
            if my:
                y = float(my.group(1))
            if me and float(me.group(1)) > 0 and typ.startswith("Support material") and (mx or my):
                pts.append((x, y, z, typ == "Support material interface"))
    return np.array(pts)


def skin_mesh(step):
    import cadquery as cq
    sh = cq.importers.importStep(step).val()
    v, f = sh.tessellate(0.02, 0.1)
    return trimesh.Trimesh(np.array([[q.x, q.y, q.z] for q in v]), np.array(f), process=True)


def main(stl, gcode, skin_step=None, center=(125.0, 105.0)):
    m = trimesh.load(stl, process=False)
    b = m.bounds
    # containment test against the closed outer body (skin + flat bottom, no windows);
    # side-button bosses lie outside it, so support under them counts as "outside"
    skin_step = skin_step or os.path.join(os.path.dirname(stl), "outer_skin_nurbs_ref_v01.step")
    sk = skin_mesh(skin_step)
    off = np.array(center) - (b[0, :2] + b[1, :2]) / 2
    sp = support_points(gcode)
    sp[:, :2] -= off
    res = {"support_extrusion_points": int(len(sp))}
    outside = np.zeros(len(sp), bool)
    for z in np.unique(sp[:, 2]):
        sec = sk.section(plane_origin=[0, 0, z - 0.1], plane_normal=[0, 0, 1])
        if sec is None:
            continue
        p2, T = sec.to_2D()
        polys = [Polygon(pl.exterior) for pl in p2.polygons_full]
        # skin sections are closed loops; the NURBS body is ~0.25 mm short at the tips -> small buffer
        if not polys:
            continue
        outer = unary_union(polys)
        k = sp[:, 2] == z
        xy = sp[k, :2] - T[:2, 3]
        outside[np.nonzero(k)[0]] = ~contains_xy(outer.buffer(0.4), xy[:, 0], xy[:, 1])
    res["outside_outer_skin_points_all"] = int(outside.sum())
    # contact = support extrusions outside the outer skin that come within 0.6 mm of the part surface
    # (tree trunks standing on the bed next to the part are not contacts)
    contact = np.zeros(len(sp), bool)
    idx = np.nonzero(outside & (sp[:, 2] > 0.6))[0]
    if len(idx):
        _, dist, _ = trimesh.proximity.closest_point(m, sp[idx, :3])
        contact[idx[dist < 0.6]] = True
    o = sp[contact]
    res["outside_skin_contact_points"] = int(contact.sum())
    if len(o):
        res["outside_bbox"] = [np.round(o[:, :3].min(0), 1).tolist(), np.round(o[:, :3].max(0), 1).tolist()]
        # group by region
        reg = {}
        for q in o:
            key = f"x{int(q[0] // 10 * 10)} y{int(q[1] // 10 * 10)} z{int(q[2] // 5 * 5)}"
            reg[key] = reg.get(key, 0) + 1
        res["outside_by_region"] = dict(sorted(reg.items(), key=lambda kv: -kv[1])[:15])
    res["inside_points"] = int((~outside).sum())
    res["interface_points_total"] = int((sp[:, 3] > 0).sum())
    out = os.path.join(os.path.dirname(stl), "support_check.json")
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
