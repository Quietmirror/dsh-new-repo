"""All documentation figures for the V0.1 build (views, contact zones, thickness, overhang, envelopes, sections)."""
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Patch

import render as R

GREY = (0.80, 0.82, 0.86)


def _rgb(c):
    return (np.array(c) * 255).astype(np.uint8)


def contact_vertex_colors(g, Q, mesh):
    v = mesh.vertices
    n = mesh.vertex_normals
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    outer = (g.s_interp(v) > -0.35)          # vertex lies on the outer skin (not inner wall / ribs)
    col = np.tile(_rgb(GREY), (len(v), 1))
    c = Q["contact"]
    L = Q["size"]["length"]
    lp = Q["lip"]
    zones = []
    for key in ("lmb", "rmb"):
        d = c[key]
        zones.append(((np.hypot((x - d["x"]) / d["rx"], (y - d["y"]) / d["ry"]) < 1.0) & (n[:, 2] > 0.3), (0.25, 0.45, 0.95)))
    t = c["thumb"]
    zones.append(((x < -10) & (np.abs(y - t["y"]) < t["ay"]) & (np.abs(z - t["z"]) < t["az"]), (0.2, 0.7, 0.3)))
    r = c["ring"]
    zones.append(((x > 10) & (np.abs(y - r["y"]) < r["ay"]) & (np.abs(z - r["z"]) < r["az"]), (0.95, 0.55, 0.15)))
    zones.append(((x > 10) & (y > lp["right_y0"] * L) & (y < lp["right_y1"] * L) & (z < lp["right_z"]), (0.6, 0.3, 0.75)))
    for bf, sb in zip(g.boss_frames, Q["side_buttons"]):
        d = v - bf["p0"]
        zones.append(((np.abs(d @ bf["u"]) < sb["length"] / 2 + 0.5) & (np.abs(d @ bf["v"]) < sb["height"] / 2 + 0.5) & (np.abs(d @ bf["n"]) < 4), (0.9, 0.2, 0.2)))
    for m, cc in zones:
        col[m & outer] = _rgb(cc)
    return col


LEGEND_CONTACT = [Patch(color=(0.25, 0.45, 0.95), label="主键指肚区 LMB/RMB"), Patch(color=(0.2, 0.7, 0.3), label="拇指浅凹"),
                  Patch(color=(0.95, 0.55, 0.15), label="无名指支撑"), Patch(color=(0.6, 0.3, 0.75), label="小指/防拖地托边"),
                  Patch(color=(0.9, 0.2, 0.2), label="侧键触感凸台")]


def sheet(paths, out, cols=2, title=None):
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in paths]
    w = max(i.size[0] for i in ims)
    h = max(i.size[1] for i in ims)
    rows = (len(ims) + cols - 1) // cols
    top = 60 if title else 0
    S = Image.new("RGB", (cols * w, rows * h + top), "white")
    for k, im in enumerate(ims):
        S.paste(im, ((k % cols) * w, (k // cols) * h + top))
    if title:
        from PIL import ImageDraw, ImageFont
        d = ImageDraw.Draw(S)
        try:
            f = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 34)
        except Exception:
            f = None
        d.text((20, 12), title, fill=(0, 0, 0), font=f)
    S.save(out)
    return out


def all_figures(g, Q, mesh, noph, envs, bat_alt, over, vt, polys, report, outdir, log):
    os.makedirs(outdir, exist_ok=True)
    meas = report["measured"]
    sen = Q["bottom"]["sensor"]
    rp = Q["bottom"]["pad_d"] / 2

    def ann_top(ax, right, upv):
        P2 = lambda p: (np.dot(p, right), np.dot(p, upv))
        for name, (px, py) in g.pads.items():
            a, b = P2((px, py, 0))
            ax.add_patch(Circle((a, b), rp, fill=False, ls="--", lw=1.0, color="tab:red"))
            ax.add_patch(Circle((a, b), 3.75, fill=False, lw=0.8, color="tab:orange"))
            ax.text(a, b + rp + 1.2, name, ha="center", fontsize=7, color="tab:red")
        a, b = P2((sen["x"], sen["y"], 0))
        ax.plot([a - 4, a + 4], [b, b], color="tab:red", lw=1)
        ax.plot([a, a], [b - 4, b + 4], color="tab:red", lw=1)
        ax.text(a + 1, b - 5.5, "传感器轴线", fontsize=7, color="tab:red")
        bnd = np.array(meas["bounds"])
        x0, y0 = P2(bnd[0])
        x1, y1 = P2(bnd[1])
        xa, xb = sorted([x0, x1])
        ya, yb = sorted([y0, y1])
        ax.annotate("", (xa, ya - 3), (xb, ya - 3), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text((xa + xb) / 2, ya - 6, f"最大外宽 {meas['width_X_max']:.1f}", ha="center", fontsize=8)
        ax.annotate("", (xb + 3, ya), (xb + 3, yb), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text(xb + 4, (ya + yb) / 2, f"长 {meas['length_Y']:.1f}", rotation=90, va="center", fontsize=8)
        th, rg = np.array(meas["thumb_contact_point"]), np.array(meas["ring_contact_point"])
        ta, tb = P2(th)
        ra, rb = P2(rg)
        ax.plot([ta, ra], [tb, rb], "o", color="tab:green", ms=4)
        yy = (tb + rb) / 2
        ax.annotate("", (ta, yy), (ra, yy), arrowprops=dict(arrowstyle="<->", lw=0.8, color="tab:green"))
        ax.text((ta + ra) / 2, yy + 1.2, f"有效夹持宽 {meas['grip_width_thumb_to_ring']:.1f}", ha="center", fontsize=8, color="tab:green")

    def ann_side(ax, right, upv):
        P2 = lambda p: (np.dot(p, right), np.dot(p, upv))
        bnd = np.array(meas["bounds"])
        top = mesh.vertices[np.argmax(mesh.vertices[:, 2])]
        a, b = P2(top)
        ax.annotate("", (a, 0), (a, b), arrowprops=dict(arrowstyle="<->", lw=0.8))
        ax.text(a + 1, b / 2, f"高 {meas['height_Z_max']:.1f}\n(脚贴平面起算)", fontsize=8)
        ax.axhline(0, color="tab:red", lw=0.6, ls="--")
        ax.text(P2((0, bnd[0][1], 0))[0], -2.5, "Z=0 脚贴平台平面", fontsize=7, color="tab:red")

    views = ["top", "bottom", "left", "right", "front", "rear", "iso", "iso_fr", "iso_bottom"]
    paths = {}
    for v in views:
        ann = ann_top if v in ("top", "bottom") else (ann_side if v in ("left", "right") else None)
        paths[v] = R.render([{"mesh": mesh}], v, os.path.join(outdir, f"view_{v}.png"), size=(1500, 1100), annotate=ann)
    sheet([paths[v] for v in ("iso", "iso_fr", "top", "bottom", "left", "right")], os.path.join(outdir, "overview_sheet.png"),
          title="Fingertip mouse V0.1 一体式试握件  (单位 mm)")
    log("views rendered")

    # contact zones
    vc = contact_vertex_colors(g, Q, mesh)
    zp = []
    for v in ("iso", "iso_fr", "left", "right"):
        def leg(ax, r, u):
            ax.legend(handles=LEGEND_CONTACT, loc="lower right", fontsize=7, framealpha=0.85)
        zp.append(R.render([{"mesh": mesh, "vcolors": vc}], v, os.path.join(outdir, f"contact_{v}.png"), size=(1300, 950), annotate=leg,
                           title=R.VIEWS[v]["title"] + "  接触区（无开孔）"))
    sheet(zp, os.path.join(outdir, "contact_zones_sheet.png"), title="接触区与减重窗口：窗口距接触区 ≥ 2 mm（见 report.json keepout）")

    # thickness
    if vt is not None:
        cmap = plt.get_cmap("turbo_r")
        lo, hi = 0.6, 3.0
        cols = (np.array(cmap(np.clip((vt - lo) / (hi - lo), 0, 1)))[:, :3] * 255).astype(np.uint8)
        cols[vt <= 0] = (255, 0, 255)
        tp = []
        for v in ("iso", "iso_bottom", "top", "bottom"):
            tp.append(R.render([{"mesh": mesh, "vcolors": cols}], v, os.path.join(outdir, f"thickness_{v}.png"), size=(1300, 950),
                               colorbar={"cmap": cmap, "vmin": lo, "vmax": hi, "label": "壁厚 mm"},
                               title=R.VIEWS[v]["title"] + "  壁厚"))
        sheet(tp, os.path.join(outdir, "thickness_sheet.png"),
              title=f"壁厚（沿内法向射线穿过实体场，亚体素插值）设计壁厚 {Q['shell']['wall']} mm；深紫 = 平台/筋/凸台等实心部位")

    # overhang / support
    vcol = np.tile(_rgb(GREY), (len(mesh.vertices), 1))
    fv = mesh.faces[over].ravel()
    vcol[fv] = _rgb((0.9, 0.2, 0.2))
    op = []
    for v in ("iso_bottom", "bottom", "front", "left"):
        op.append(R.render([{"mesh": mesh, "vcolors": vcol}], v, os.path.join(outdir, f"overhang_{v}.png"), size=(1300, 950),
                           title=R.VIEWS[v]["title"] + "  红色=悬垂>45°需支撑"))
    sheet(op, os.path.join(outdir, "overhang_sheet.png"), title="打印朝向：底面（五脚平台）贴床；红色 = 需支撑（壳内顶面、窗口上沿、滚轮/受力柱），外表面接触区无支撑")

    # envelopes in a translucent shell
    items = [{"mesh": noph, "color": (0.75, 0.78, 0.82), "opacity": 0.22}]
    for e in envs:
        items.append({"mesh": e.trimesh(), "color": e.color, "opacity": 0.95})
    groups = {}
    for e in envs:
        groups.setdefault(e.group, e.color)
    gname = {"sensor": "传感器/镜头", "switch": "微动", "wheel": "滚轮/编码器", "board": "PCB", "battery": "电池/插头",
             "usb": "USB-C", "wire": "走线", "antenna": "天线净空", "power": "电源开关"}
    legend = [Patch(color=c, label=gname.get(k, k)) for k, c in groups.items()]
    ep = []
    for v in ("iso", "top", "left", "front"):
        def leg(ax, r, u):
            ax.legend(handles=legend, loc="upper right", fontsize=7, framealpha=0.85)
        ep.append(R.render(items, v, os.path.join(outdir, f"envelopes_{v}.png"), size=(1400, 1000), annotate=leg,
                           title=R.VIEWS[v]["title"] + "  电子参考包络（多为估计）"))
    sheet(ep, os.path.join(outdir, "envelopes_sheet.png"), title="装配版参考包络（占位件已删除的骨架）；E=估计值，不是装配尺寸")

    sections(g, Q, envs + bat_alt, polys, report, outdir)
    log("figures done")


def _slice(g, F, axis, value):
    if axis == "y":
        i = int(np.argmin(np.abs(g.ys - value)))
        return F[:, i, :], g.xs, g.zs, ("X", "Z")
    if axis == "x":
        i = int(np.argmin(np.abs(g.xs - value)))
        return F[i, :, :], g.ys, g.zs, ("Y", "Z")
    i = int(np.argmin(np.abs(g.zs - value)))
    return F[:, :, i], g.xs, g.ys, ("X", "Y")


def sections(g, Q, envs, polys, report, outdir):
    chain = report["electronics"]["z_chain"]
    wy = g.wheel_center[1]
    secs = [("y", Q["bottom"]["sensor"]["y"], "Y=%.1f 传感器轴线截面"), ("y", wy, "Y=%.1f 滚轮轴截面"),
            ("y", 22.0, "Y=%.1f 电池截面"), ("y", 52.0, "Y=%.1f 侧键截面"), ("x", 0.0, "X=%.1f 中纵截面"),
            ("z", chain["pcb_bottom_z"] + 0.8, "Z=%.1f 主板平面截面")]
    paths = []
    for axis, val, title in secs:
        A, u, w, lab = _slice(g, g.skeleton_no_placeholders, axis, val)
        Fp, _, _, _ = _slice(g, g.F, axis, val)
        Sk, _, _, _ = _slice(g, g.s, axis, val)
        U, Wg = np.meshgrid(u, w, indexing="ij")
        fig, ax = plt.subplots(figsize=(9, 6.2) if axis != "z" else (8, 9), dpi=140)
        ax.contourf(U, Wg, Fp, levels=[-100, 0], colors=["#e7c9a9"], alpha=0.6)
        ax.contourf(U, Wg, A, levels=[-100, 0], colors=["#9aa3ad"])
        ax.contour(U, Wg, Sk, levels=[0], colors=["#555"], linewidths=0.6)
        for e in envs:
            if axis == "y":
                d = e.sdf(U, np.full_like(U, val), Wg)
            elif axis == "x":
                d = e.sdf(np.full_like(U, val), U, Wg)
            else:
                d = e.sdf(U, Wg, np.full_like(U, val))
            if (d < 0).any():
                ax.contourf(U, Wg, d, levels=[-100, 0], colors=[e.color], alpha=0.45)
                ax.contour(U, Wg, d, levels=[0], colors=[e.color], linewidths=1.0)
                iy = np.unravel_index(np.argmin(d), d.shape)
                ax.text(U[iy], Wg[iy], e.name, fontsize=6, ha="center", va="center")
        if axis != "z":
            for k, zc in chain.items():
                ax.axhline(zc, lw=0.5, ls=":", color="tab:blue")
                ax.text(u[0] + 0.5, zc + 0.15, f"{k} {zc:.2f}", fontsize=6, color="tab:blue")
            ax.axhline(0, lw=0.6, color="tab:red")
        elif polys:
            p = polys[0]
            ax.plot(p[:, 0], p[:, 1], "g--", lw=1.2, label="主板可用最大外形 (keep-in, 0.5mm 间隙)")
            ax.legend(loc="lower right", fontsize=7)
        ax.set_aspect("equal")
        ax.set_xlabel(lab[0] + " [mm]")
        ax.set_ylabel(lab[1] + " [mm]")
        ax.grid(alpha=0.25)
        ax.set_title(title % val + "（灰=骨架/壳体，米色=V0.1 占位件，彩色=电子包络）", fontsize=9)
        fn = os.path.join(outdir, f"section_{axis}{val:.0f}.png")
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)
        paths.append(fn)
    sheet(paths, os.path.join(outdir, "sections_sheet.png"), cols=2, title="内部截面：骨架 vs 电子参考包络")
