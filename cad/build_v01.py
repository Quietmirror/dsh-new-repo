"""Build the V0.1 fingertip trial grip: calibrate -> SDF -> STL -> checks -> envelopes -> renders -> report.

Usage (from repo root, headless):
    xvfb-run -a python3 cad/build_v01.py --h 0.2 --out out/v0.1
    python3 cad/build_v01.py --h 0.4 --out /tmp/preview --no-render        # quick preview
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params_v01 import P  # noqa: E402
import fingertip_mouse as FM  # noqa: E402
import checks as C  # noqa: E402
import envelopes as EN  # noqa: E402


def decimate(mesh, target_faces, log):
    """Quadric edge-collapse decimation preserving topology, normals and planar regions (MeshLab)."""
    import pymeshlab
    import trimesh
    if len(mesh.faces) <= target_faces:
        return mesh
    ms = pymeshlab.MeshSet()
    ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(mesh.vertices, np.float64), face_matrix=np.asarray(mesh.faces, np.int32)))
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=int(target_faces), qualitythr=0.7, preserveboundary=True,
                                                preservenormal=True, preservetopology=True, optimalplacement=True,
                                                planarquadric=True, autoclean=True)
    cm = ms.current_mesh()
    m = trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=False)
    if m.volume < 0:
        m.invert()
    log(f"decimated {len(mesh.faces)} -> {len(m.faces)} faces")
    return m


def repair_self_intersections(mesh, log, iters=10):
    """Delete self-intersecting faces (+2 rings) and close the small holes; repeat until clean.
    Coordinates are rounded to float32 every pass, so the result is also clean after saving as STL."""
    import pymeshlab
    import trimesh
    V = np.asarray(mesh.vertices, np.float32).astype(np.float64)
    Fc = np.asarray(mesh.faces, np.int32)
    hist, where = [], []
    for _ in range(iters):
        ms = pymeshlab.MeshSet()
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=V, face_matrix=Fc))
        ms.compute_selection_by_self_intersections_per_face()
        cm = ms.current_mesh()
        n = cm.selected_face_number()
        hist.append(int(n))
        if n == 0:
            break
        sel = cm.face_selection_array()
        c = cm.vertex_matrix()[cm.face_matrix()[sel]].mean(axis=1)
        where += np.round(c, 1).tolist()
        ms.apply_selection_dilatation()
        ms.apply_selection_dilatation()
        ms.meshing_remove_selected_faces()
        ms.meshing_remove_unreferenced_vertices()
        ms.meshing_close_holes(maxholesize=300, newfaceselected=False, selfintersection=False)
        ms.meshing_repair_non_manifold_edges()
        cm = ms.current_mesh()
        m = fix_collinear_faces(trimesh.Trimesh(cm.vertex_matrix(), cm.face_matrix(), process=False), log)
        V = np.asarray(m.vertices, np.float32).astype(np.float64)
        Fc = np.asarray(m.faces, np.int32)
    m = fix_collinear_faces(trimesh.Trimesh(V, Fc, process=False), log)
    if m.volume < 0:
        m.invert()
    # cluster locations for the report
    locs = []
    if where:
        w = np.array(where)
        from scipy.cluster.hierarchy import fcluster, linkage
        lab = fcluster(linkage(w, "single"), 3.0, "distance") if len(w) > 1 else np.ones(1, int)
        locs = [np.round(w[lab == k].mean(0), 1).tolist() for k in np.unique(lab)]
    return snap_bottom(m), {"self_intersections_per_pass_float32": hist, "repaired_locations_xyz": locs}


def fix_collinear_faces(m, log, tol=1e-12):
    """Zero-area (collinear) triangles from hole filling: split the neighbour across the long edge at the
    middle vertex and drop the sliver. Keeps the mesh closed and consistently oriented."""
    import trimesh
    F = m.faces.copy()
    V = m.vertices
    fixed = 0
    for _ in range(10):
        area = trimesh.triangles.area(V[F])
        bad = np.nonzero(area < tol)[0]
        if len(bad) == 0:
            break
        f0 = bad[0]
        tri = F[f0]
        L = [np.linalg.norm(V[tri[(k + 1) % 3]] - V[tri[k]]) for k in range(3)]
        k = int(np.argmax(L))
        a, c, b = tri[k], tri[(k + 1) % 3], tri[(k + 2) % 3]      # long edge a->c, b in the middle
        nb = [i for i in np.nonzero((F == a).any(1) & (F == c).any(1))[0] if i != f0]
        if len(nb) != 1:
            break
        f1 = nb[0]
        t1 = list(F[f1])
        d = [v for v in t1 if v not in (a, c)][0]
        ia = t1.index(a)
        if t1[(ia + 1) % 3] == c:          # f1 = (a, c, d)
            new = [[a, b, d], [b, c, d]]
        else:                               # f1 = (c, a, d)
            new = [[c, b, d], [b, a, d]]
        keep = np.ones(len(F), bool)
        keep[[f0, f1]] = False
        F = np.vstack([F[keep], new])
        fixed += 1
    if fixed:
        log(f"  fixed {fixed} collinear (zero-area) triangle(s)")
    return trimesh.Trimesh(V, F, process=False)


def hausdorff(a, b):
    import pymeshlab
    ms = pymeshlab.MeshSet()
    for m in (a, b):
        ms.add_mesh(pymeshlab.Mesh(vertex_matrix=np.asarray(m.vertices, np.float64), face_matrix=np.asarray(m.faces, np.int32)))
    r1 = ms.get_hausdorff_distance(sampledmesh=0, targetmesh=1, samplenum=400000)
    r2 = ms.get_hausdorff_distance(sampledmesh=1, targetmesh=0, samplenum=400000)
    return {"max_mm": float(max(r1["max"], r2["max"])), "mean_mm": float((r1["mean"] + r2["mean"]) / 2),
            "rms_mm": float(max(r1["RMS"], r2["RMS"]))}


def drop_specks(mesh, log, max_faces=50):
    """Remove zero-volume marching-cubes specks (a few faces around a node where F ~ 0)."""
    import trimesh
    parts = mesh.split(only_watertight=False)
    keep = [p for p in parts if len(p.faces) > max_faces or abs(p.volume) > 1e-2]
    if len(keep) < len(parts):
        log(f"removed {len(parts) - len(keep)} degenerate speck(s): " +
            ", ".join(f"{len(p.faces)}f@{np.round(p.bounds.mean(0), 1).tolist()}" for p in parts if p not in keep))
    m = trimesh.util.concatenate(keep) if len(keep) > 1 else keep[0]
    return m


def snap_bottom(mesh, tol=1e-3):
    """Marching cubes puts the pad plane at Z=0 analytically; remove float noise."""
    v = mesh.vertices.copy()
    v[np.abs(v[:, 2]) < tol, 2] = 0.0
    mesh.vertices = v
    return mesh


def nurbs_skin(g, h, out, log):
    if h > 0.4:
        log("skipped (coarse voxel)")
        return {"skipped": "voxel > 0.4 mm gives too coarse sections; build with --h 0.2"}
    try:
        import skin_step
        r = skin_step.export(g, os.path.join(out, "outer_skin_nurbs_ref_v01.step"))
        log(json.dumps(r))
        return r
    except Exception as ex:
        log("NURBS skin failed:", ex)
        return {"error": str(ex)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=float, default=0.2, help="voxel size (mm)")
    ap.add_argument("--h-cal", type=float, default=0.4)
    ap.add_argument("--out", default="out/v0.1")
    ap.add_argument("--faces", type=int, default=320000, help="target faces of the delivered STL")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-thickness", action="store_true")
    args = ap.parse_args()
    out = args.out
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.path.join(out, "img"), exist_ok=True)
    logf = open(os.path.join(out, "build.log"), "w")

    def log(*a):
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        logf.write(s + "\n")
        logf.flush()

    t0 = time.time()
    log("== calibrate size targets")
    Q, hist = FM.calibrate(P, h=args.h_cal, iters=4, log=log)
    with open(os.path.join(out, "params_resolved.json"), "w") as f:
        json.dump(Q, f, indent=1, ensure_ascii=False, default=str)

    log("== build field h=%.3f" % args.h)
    g = FM.TrialGrip(Q, h=args.h, log=log)
    g.build(placeholders=True)
    raw = snap_bottom(drop_specks(g.mesh(), log))
    log(f"raw mesh {len(raw.faces)} faces, watertight={raw.is_watertight}  t={time.time()-t0:.0f}s")
    meas = g.measure(raw)
    log("measured:", json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in meas.items() if not isinstance(v, list)}))

    # decimate, then repair any self-intersections the edge collapses created (delete + local hole fill)
    mesh = snap_bottom(decimate(raw, args.faces, log))
    mesh, rep = repair_self_intersections(mesh, log)
    rep["hausdorff_to_raw_marching_cubes"] = hausdorff(mesh, raw)
    log("  decimation/repair:", json.dumps(rep))
    stl = os.path.join(out, "fingertip_v01_trial_grip.stl")
    mesh.export(stl)
    log("wrote", stl, os.path.getsize(stl) // 1024, "KB")
    try:
        mesh.export(os.path.join(out, "fingertip_v01_trial_grip.3mf"))
    except Exception as ex:
        log("3mf export skipped:", ex)

    # reference skeleton without V0.1 placeholders (for the assembly study)
    import trimesh  # noqa: F401
    noph = snap_bottom(drop_specks(g.mesh(g.skeleton_no_placeholders), log))
    noph_d, _ = repair_self_intersections(snap_bottom(decimate(noph, 160000, log)), log)
    noph_d.export(os.path.join(out, "ref_skeleton_without_placeholders.stl"))

    report = {"params_file": "params_resolved.json", "calibration": hist, "measured": meas, "decimation": rep}
    log("== mesh checks (delivered STL, re-read from disk)")
    disk = trimesh.load(stl, process=False)
    disk.merge_vertices()
    report["mesh_checks"] = C.mesh_checks(disk)
    d3 = trimesh.load(os.path.join(out, "fingertip_v01_trial_grip.3mf"), force="mesh", process=False)
    d3.merge_vertices()
    report["mesh_checks_3mf"] = {"watertight": bool(d3.is_watertight), "bodies": len(d3.split(only_watertight=False)),
                                 "self_intersecting_faces": C.self_intersections(d3), "faces": int(len(d3.faces))}
    report["mesh_checks_raw_marching_cubes"] = {k: v for k, v in C.mesh_checks(raw).items() if k in
                                                ("faces", "watertight", "bodies", "self_intersecting_faces", "volume_mm3", "non_manifold_edges")}
    log(json.dumps(report["mesh_checks"], indent=1))
    if report["mesh_checks"]["bodies"] != 1 or report["mesh_checks"]["self_intersecting_faces"] != 0:
        log("WARNING: delivered mesh is not a single clean body - inspect before printing")
    report["skeleton_without_placeholders"] = {"volume_mm3": float(noph.volume), "watertight": bool(noph_d.is_watertight),
                                               "bodies": len(noph_d.split(only_watertight=False))}

    log("== pads")
    report["pads"] = C.pad_checks(g, mesh)
    log(json.dumps(report["pads"], indent=1))

    report["wheel_placeholder"] = {"center": list(g.wheel_center), "gap_to_shell_mm": C.wheel_gap(g),
                                   "top_surface_z_at_wheel": g.wheel_top_surface_z,
                                   "wheel_top_z": g.wheel_center[2] + Q["wheel"]["radius"]}
    report["keepout"] = C.keepout_checks(g, Q["keepout_margin"], Q["keepout_margin_wheel"])
    log("keepout:", json.dumps({k: (v.get("ok"), round(v.get("min_dist_to_window_mm", 0), 2), round(v.get("min_dist_to_wheel_opening_mm", 0), 2)) for k, v in report["keepout"].items()}))
    over, report["overhang"] = C.overhang_report(mesh)
    report["side_buttons"] = [{"name": b["name"], "surface_point": b["p0"].tolist(), "normal": b["n"].tolist()} for b in g.boss_frames]
    report["pad_positions"] = g.pads
    report["rib_segments"] = [[list(a), list(b)] for a, b in g.rib_segments]

    # volume breakdown by feature (voxel count, each voxel assigned to the first matching group)
    vox = g.h ** 3
    Zp = g.Z >= 0
    a = (g.shell_final < 0) & Zp
    b = (g.ring < 0) & Zp & ~a
    c = ((g.inner < 0) | (g.pad_field < 0)) & Zp & ~a & ~b
    d = (g.F < 0) & ~((g.skeleton_no_placeholders < 0))
    report["volume_breakdown_mm3_voxel_estimate"] = {
        "shell_with_windows_and_side_bosses": float(a.sum() * vox), "bottom_perimeter_flange": float(b.sum() * vox),
        "ribs_sensor_frame_pads": float(c.sum() * vox), "v01_placeholders_wheel_and_posts": float(d.sum() * vox)}
    del a, b, c, d
    density = {"PLA": 1.24, "PETG": 1.27}
    vol_cm3 = report["mesh_checks"]["volume_mm3"] / 1000
    report["mass_solid_g"] = {k: round(vol_cm3 * d, 2) for k, d in density.items()}

    if not args.no_thickness:
        log("== wall thickness")
        vt = C.ray_thickness(g, mesh)
        report["thickness"] = C.thickness_report(g, mesh, vt)
        log(json.dumps(report["thickness"], indent=1))
    else:
        vt = None

    log("== electronics envelopes")
    envs, chain = EN.layout(g, battery="301230")
    envs_b, _ = EN.layout(g, battery="602030")
    bat_alt = [e for e in envs_b if e.name.startswith("battery_6")]
    allenv = envs + bat_alt
    E = Q["electronics"]
    zpb = chain["pcb_bottom_z"]
    zpt_comp = chain["pcb_top_z"] + E["pcb_top_components"][0]
    # board bottom rests on the grid top; lateral clearance from the walls / other solids
    polys, reg = EN.board_keepin(g, g.skeleton_no_placeholders, zpb, zpt_comp, E["clearance"])
    keep = None
    if polys:
        pmain = polys[0]
        keep = {"z_range": [zpb, zpt_comp], "outline_bbox": [np.round(pmain.min(0), 2).tolist(), np.round(pmain.max(0), 2).tolist()],
                "area_mm2": float(reg.sum() * g.h ** 2)}
        np.savetxt(os.path.join(out, "main_board_keepin_outline.csv"), pmain, fmt="%.3f", delimiter=",", header="x_mm,y_mm", comments="")
        try:
            import ezdxf
            doc = ezdxf.new()
            doc.modelspace().add_lwpolyline(pmain.tolist(), close=True)
            doc.saveas(os.path.join(out, "main_board_keepin_outline.dxf"))
        except Exception as ex:
            log("dxf skipped", ex)
    report["main_board_keepin"] = keep
    report["electronics"] = EN.analyze(g, allenv, g.skeleton_no_placeholders, chain, E["clearance"])
    log(json.dumps({k: v["status"] for k, v in report["electronics"]["envelopes"].items()}, indent=1))
    log("pairs:", report["electronics"]["envelope_pair_overlaps"])
    try:
        EN.export_step(allenv, os.path.join(out, "electronics_envelopes_v01.step"))
        log("wrote envelopes STEP")
    except Exception as ex:
        log("STEP export failed:", ex)
    import trimesh as tm
    env_mesh = tm.util.concatenate([e.trimesh() for e in envs])
    env_mesh.export(os.path.join(out, "electronics_envelopes_v01.stl"))

    log("== NURBS reference skin STEP")
    report["nurbs_skin"] = nurbs_skin(g, args.h, out, log)

    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    log("report written, t=%.0fs" % (time.time() - t0))

    if not args.no_render:
        import render_sheets
        render_sheets.all_figures(g, Q, mesh, noph_d, envs, bat_alt, over, vt, polys, report, os.path.join(out, "img"), log)
    import resource
    log("done t=%.0fs peak_rss=%.1f GB" % (time.time() - t0, resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6))


if __name__ == "__main__":
    main()
