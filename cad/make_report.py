"""Render out/<ver>/report.json (+ slice_report.json) as a Chinese Markdown check report.

    python3 cad/make_report.py out/v0.1
"""
import json
import os
import sys


def f(x, n=2):
    return "—" if x is None else (f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))


def main(d):
    r = json.load(open(os.path.join(d, "report.json")))
    sp = os.path.join(d, "slice_report.json")
    s = json.load(open(sp)) if os.path.exists(sp) else None
    m, mc, pads = r["measured"], r["mesh_checks"], r["pads"]
    L = []
    w = L.append
    w("# V0.1 试握件自动检查报告\n")
    w("由 `cad/build_v01.py` 生成的 `report.json` 转写；数值均为本次构建实测，单位 mm。\n")
    w("## 1 外形尺寸（网格包围盒，Z 自脚贴平台平面起算）\n")
    w("| 项目 | 目标 | 实测 |\n|---|---|---|")
    w(f"| 长 Y | 90.0 | {f(m['length_Y'])} |")
    w(f"| 最大外宽 X（含右托边、侧键凸台） | 62.0 | {f(m['width_X_max'])} |")
    w(f"| 外壳宽（不含侧键凸台） | — | {f(m['skin_width_without_side_buttons'])} |")
    w(f"| 高 Z（贴脚贴后约 +0.65） | 31.0 | {f(m['height_Z_max'])} |")
    w(f"| 有效夹持宽（拇指点→无名指点，X 向） | ≈54 | {f(m['grip_width_thumb_to_ring'])} |")
    th, rg = m["thumb_contact_point"], m["ring_contact_point"]
    w(f"\n拇指测点 ({f(th[0],1)}, {f(th[1],1)}, {f(th[2],1)})，无名指测点 ({f(rg[0],1)}, {f(rg[1],1)}, {f(rg[2],1)})。\n")
    w("## 2 网格检查（交付 STL）\n")
    w("| 项目 | 结果 |\n|---|---|")
    w(f"| 面数 / 顶点数 | {mc['faces']} / {mc['vertices']} |")
    w(f"| 封闭（watertight） | {mc['watertight']} |")
    w(f"| 实体数（连通分量） | {mc['bodies']} |")
    w(f"| 边界边 / 非流形边 / 退化面 | {mc['boundary_edges']} / {mc['non_manifold_edges']} / {mc['degenerate_faces']} |")
    w(f"| 自相交面（MeshLab） | {mc['self_intersecting_faces']} |")
    w(f"| manifold3d 校验 | {mc['manifold3d_status']} |")
    w(f"| 体积 | {f(mc['volume_mm3']/1000, 2)} cm³ |")
    raw = r.get("mesh_checks_raw_marching_cubes", {})
    dec = r.get("decimation", {})
    w(f"| 原始 marching-cubes 网格 | {raw.get('faces')} 面，封闭={raw.get('watertight')}，实体={raw.get('bodies')}，自相交={raw.get('self_intersecting_faces')} |")
    if dec:
        hd = dec.get("hausdorff_to_raw_marching_cubes", {})
        w(f"| 抽稀后自相交修复（每轮检出面数，按 float32 精度） | {dec.get('self_intersections_per_pass_float32', dec.get('self_intersections_per_pass'))}，修复位置 {dec.get('repaired_locations_xyz')} |")
        w(f"| 交付网格 vs 原始网格 Hausdorff | 最大 {f(hd.get('max_mm'),3)}，平均 {f(hd.get('mean_mm'),4)} |")
    w("\n## 3 五个脚贴平台\n")
    w("| 平台 | 中心 X, Y | 平台内底面面积 mm² | 底面 Z 范围 | 7.5 mm 圆点脚贴完全落在平面上 |\n|---|---|---|---|---|")
    for k in ("RC", "RL", "RR", "FL", "FR"):
        p = pads[k]
        w(f"| {k} | {p['center_xy'][0]}, {p['center_xy'][1]} | {f(p['bottom_face_area_mm2_within_pad'],1)} | {f(p['z_min'],4)} … {f(p['z_max'],4)} | {p['skate_7.5mm_disc_fully_supported']} |")
    w(f"\n五平台底面顶点 Z 最大偏差 **{f(pads['coplanarity_max_dev_mm'],4)} mm**；全模型最低点 Z = {f(pads['global_min_z'],4)}。\n")
    t = r.get("thickness")
    if t:
        w("## 4 壁厚（沿内法向射线穿过实体场，亚体素插值）\n")
        a, e = t["shell_wall_excluding_window_edges"], t["rounded_window_and_wheel_edges"]
        w("| 区域 | 样本 | 最小 | P1 | P5 | 中位 |\n|---|---|---|---|---|---|")
        w(f"| 外壳（排除窗口/滚轮开口圆角 1.5 mm 带） | {a['n']} | {f(a['min'])} | {f(a['p1'])} | {f(a['p5'])} | {f(a['median'])} |")
        if e:
            w(f"| 窗口/滚轮开口圆角边 | {e['n']} | {f(e['min'])} | {f(e['p1'])} | {f(e['p5'])} | {f(e['median'])} |")
        cl = t[[k for k in t if k.startswith("thin_spots")][0]]
        if cl:
            w("\n薄点（< 1.4 mm，已排除圆角边）：\n")
            w("| 位置 X,Y,Z | 最小 | 顶点数 |\n|---|---|---|")
            for c in cl[:10]:
                w(f"| {c['at_xyz']} | {f(c['min_mm'])} | {c['vertices']} |")
        w("")
    w("## 5 接触区与开孔\n")
    w("| 接触区 | 距最近减重窗口 | 距滚轮开口 | 通过 |\n|---|---|---|---|")
    for k, v in r["keepout"].items():
        w(f"| {k} | {f(v.get('min_dist_to_window_mm'))} | {f(v.get('min_dist_to_wheel_opening_mm'))} | {v.get('ok')} |")
    wp = r["wheel_placeholder"]
    w(f"\n固定滚轮占位件与外壳最小间隙 {f(wp['gap_to_shell_mm'])} mm；滚轮顶 Z={f(wp['wheel_top_z'])}，该处外壳顶面 Z={f(wp['top_surface_z_at_wheel'])}。\n")
    w("## 6 体积 / 重量\n")
    vb = r.get("volume_breakdown_mm3_voxel_estimate", {})
    w("| 部分（体素估计） | 体积 cm³ |\n|---|---|")
    for k, v in vb.items():
        w(f"| {k} | {f(v/1000)} |")
    w(f"\n实体体积 {f(mc['volume_mm3']/1000)} cm³ → 100% 实心估重 PLA {r['mass_solid_g']['PLA']} g / PETG {r['mass_solid_g']['PETG']} g。\n")
    if s:
        w("切片（PrusaSlicer CLI，`cad/prusa_v01.ini`：0.4 喷嘴、0.2 层高、3 圈壁、20% gyroid、PLA 1.24 g/cm³）：\n")
        w("| 方案 | 耗材 g | 耗材 m | 预计时间 | 层数 |\n|---|---|---|---|---|")
        for k, lab in (("no_supports", "零件本体（不含支撑）"), ("with_supports", "实际打印（含树状支撑）")):
            q = s.get(k, {})
            w(f"| {lab} | {q.get('total filament used [g]')} | {f(float(q.get('filament used [mm]', 0))/1000, 2)} | {q.get('estimated printing time (normal mode)')} | {q.get('layers')} |")
        inf = s.get("info", {})
        w(f"\n切片器读入尺寸：{inf.get('size_x')} × {inf.get('size_y')} × {inf.get('size_z')}，manifold={inf.get('manifold')}，parts={inf.get('number_of_parts')}，volume={inf.get('volume')}。\n")
    sc = os.path.join(d, "support_check.json")
    if os.path.exists(sc):
        q = json.load(open(sc))
        w(f"支撑位置检查（`cad/support_check.py`，以 NURBS 外皮为界）：支撑挤出点 {q['support_extrusion_points']}，其中接触面（interface）{q['interface_points_total']}；"
          f"**外皮以外、距零件表面 < 0.6 mm 的支撑点 {q['outside_skin_contact_points']}**；外皮以外的全部支撑点 {q['outside_outer_skin_points_all']}（多为树状支撑在热床上的底座）。\n")
        if q.get("outside_by_region"):
            w(f"外皮以外的支撑分布（10 mm 网格）：{q['outside_by_region']}\n")
    o = r["overhang"]
    w("## 7 悬垂（打印朝向：底面贴床）\n")
    w(f"悬垂 > 45° 面积 {f(o['overhang_area_mm2'],0)} mm²，贴床面积 {f(o['bed_contact_area_mm2'],0)} mm²；按高度分布：{ {k: round(v) for k, v in o['overhang_area_by_z_band_mm2'].items()} }。\n")
    e = r["electronics"]
    w("## 8 电子参考包络干涉（骨架已删除 V0.1 占位件）\n")
    w("尺寸链（Z）：" + "，".join(f"{k} = {f(v)}" for k, v in e["z_chain"].items()) + "\n")
    w("| 包络 | 来源 | 状态 | 与骨架重叠 mm³ | 最大侵入 mm | 最小间隙 mm |\n|---|---|---|---|---|---|")
    for k, v in e["envelopes"].items():
        w(f"| {k} | {v['source'][:60]} | **{v['status']}** | {f(v.get('overlap_with_skeleton_mm3'),1)} | {f(v.get('max_penetration_mm'))} | {f(v.get('min_clearance_to_skeleton_mm'))} |")
    if e["envelope_pair_overlaps"]:
        w("\n包络之间的重叠：\n")
        for p in e["envelope_pair_overlaps"]:
            w(f"- {p['a']} × {p['b']}：{f(p['overlap_mm3'],1)} mm³")
    kb = r.get("main_board_keepin")
    if kb:
        w(f"\n主板可用最大外形（Z {f(kb['z_range'][0])}–{f(kb['z_range'][1])}，0.5 mm 间隙）：包围盒 {kb['outline_bbox']}，面积 {f(kb['area_mm2'],0)} mm²；轮廓见 `main_board_keepin_outline.dxf/.csv`。\n")
    ns = r.get("nurbs_skin")
    if ns and "deviation_vs_skin" in ns:
        dv = ns["deviation_vs_skin"]
        w("## 9 NURBS 外形参考 STEP\n")
        w(f"`{os.path.basename(ns['file'])}`：{ns['sections']} 个截面放样，B-rep 有效={ns['valid_brep']}，与 SDF 外形偏差 最大 {f(dv['max_mm'],3)} / P99 {f(dv['p99_mm'],3)} / 平均 {f(dv['mean_mm'],4)} mm（{dv['note']}）。\n")
    open(os.path.join(d, "检查报告.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main(sys.argv[1])
