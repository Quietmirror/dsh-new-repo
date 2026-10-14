"""Parametric fingertip mouse trial-grip generator (V0.1).

Geometry is built as signed distance fields on a voxel grid:
  1. outer skin  : plan outline (+ thumb groove / ring rest / right lip) smoothly intersected with a top height field
  2. exact SDF   : true Euclidean distance to the outer skin (KD-tree on the skin mesh) -> uniform wall thickness
  3. shell       : |s| band of `wall` thickness, open bottom, windows with rounded edges
  4. skeleton    : bottom perimeter flange, ribs, sensor frame, five coplanar skate pads
  5. features    : thumb side-button bosses, wheel opening, fixed dummy wheel, button posts (V0.1 placeholders)
The final field is meshed with marching cubes; the grid is offset half a voxel so the pad plane Z=0 is exact.
"""
import numpy as np
from scipy.interpolate import PchipInterpolator, splprep, splev, RegularGridInterpolator
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes

import sdf as S


def nonzero(f):
    """Nudge exact zeros off the iso level so marching cubes never emits degenerate faces."""
    return np.where(np.abs(f) < 1e-3, np.float32(1e-3), f)


def _pchip(pairs, xscale=1.0, yscale=1.0):
    a = np.array(pairs, dtype=float)
    return PchipInterpolator(a[:, 0] * xscale, a[:, 1] * yscale, extrapolate=True)


class TrialGrip:
    def __init__(self, P, h=0.25, log=print):
        self.P, self.h, self.log = P, h, log
        L = P["size"]["length"]
        hw = P["plan"]["half_width"]
        flare = P["lip"]["right_flare"]
        H = P["size"]["height"]
        x0, x1 = -hw * P["plan"]["left_scale"] - 5.0, hw + flare + 5.0
        y0, y1 = -3.0, L + 3.0
        nzb = int(np.ceil(1.2 / h))
        self.xs = np.arange(x0, x1 + h, h)
        self.ys = np.arange(y0, y1 + h, h)
        self.zs = (np.arange(int(np.ceil((H + 4.0) / h)) + nzb) - nzb + 0.5) * h  # Z=0 between nodes
        self.X2, self.Y2 = np.meshgrid(self.xs, self.ys, indexing="ij")
        self.X = self.X2[:, :, None]
        self.Y = self.Y2[:, :, None]
        self.Z = self.zs[None, None, :]
        self.shape = (len(self.xs), len(self.ys), len(self.zs))
        self.log(f"grid {self.shape} = {np.prod(self.shape)/1e6:.1f} M voxels at h={h}")

    # ------------------------------------------------------------------ outline & skin
    def plan_polygon(self, n=720):
        P = self.P
        L, hw, ls = P["size"]["length"], P["plan"]["half_width"], P["plan"]["left_scale"]
        c = np.array(P["plan"]["ctrl"], dtype=float)
        y0, y1 = P["plan"].get("y0", 0.0), P["plan"].get("y1", L)   # calibrated outline Y range
        right = [(w * hw, y0 + yn * (y1 - y0)) for yn, w in c]
        left = [(-w * hw * ls, y0 + yn * (y1 - y0)) for yn, w in c[::-1]][1:-1]
        pts = np.array(right + left)
        tck, _ = splprep([pts[:, 0], pts[:, 1]], s=0, per=1)
        u = np.linspace(0, 1, n, endpoint=False)
        x, y = splev(u, tck)
        return np.column_stack([x, y])

    def crest(self):
        P = self.P
        return _pchip(P["top"]["crest"], P["size"]["length"], P["size"]["height"])

    def top_height(self, x, y):
        P = self.P
        L, hw = P["size"]["length"], P["plan"]["half_width"]
        # no clipping at the ends: a slope kink there makes the smooth blend fold (thin crack at the tip)
        Hc = self.crest()(y)
        crown = _pchip(P["top"]["crown_drop"], L)(y)
        T = Hc - crown * ((x - P["top"]["crest_x"]) / hw) ** 2 - np.tan(np.radians(P["top"]["tilt_deg"])) * x
        for key in ("lmb", "rmb"):
            d = P["contact"][key]
            r = np.hypot((x - d["x"]) / d["rx"], (y - d["y"]) / d["ry"])
            T = T - d["depth"] * S.bump(r)
        return T

    def side_offset(self):
        """delta(x,y,z) added to the plan SDF: + moves the wall inward."""
        P, Y, Z = self.P, self.Y, self.Z
        L = P["size"]["length"]
        sL = S.smoothstep(3.0, -3.0, self.X2)[:, :, None]
        sR = 1.0 - sL
        c = P["contact"]
        t = c["thumb"]
        d = t["depth"] * sL * S.bump((Y - t["y"]) / t["ay"]) * S.bump((Z - t["z"]) / t["az"])
        r = c["ring"]
        d = d + r["depth"] * sR * S.bump((Y - r["y"]) / r["ay"]) * S.bump((Z - r["z"]) / r["az"])
        lp = P["lip"]
        zc = np.maximum(Z, 0.0)
        wy = S.plateau(Y, lp["right_y0"] * L, lp["right_y1"] * L, lp["right_ramp"] * L)
        d = d - lp["right_flare"] * sR * wy * (1.0 - S.smoothstep(0.0, lp["right_z"], zc)) ** 2
        if lp.get("left_flare", 0) > 0:
            d = d - lp["left_flare"] * sL * wy * (1.0 - S.smoothstep(0.0, lp["left_z"], zc)) ** 2
        return d

    def outer_field(self):
        P = self.P
        L = P["size"]["length"]
        self.poly = self.plan_polygon()
        self.log("plan SDF ...")
        sd2 = S.sd_polygon(self.X2, self.Y2, self.poly).astype(np.float32)
        self.sd_plan2d = sd2
        T = self.top_height(self.X2, self.Y2)
        gx, gy = np.gradient(T, self.h, self.h)
        g = np.sqrt(1.0 + gx * gx + gy * gy)
        self.T = T
        f_top = ((self.Z - T[:, :, None]) / g[:, :, None]).astype(np.float32)
        f_plan = (sd2[:, :, None] + self.side_offset()).astype(np.float32)
        k = _pchip(P["top"]["blend"], L)(self.Y2)[:, :, None].astype(np.float32)
        f = S.smax(f_plan, f_top, k).astype(np.float32)
        return f

    def exact_sdf(self, f, band=4.5):
        """Replace the approximate field by the true distance to its zero set inside a band."""
        h = self.h
        origin = np.array([self.xs[0], self.ys[0], self.zs[0]])
        verts, faces, _, _ = marching_cubes(nonzero(f), 0.0, spacing=(h, h, h))
        verts += origin
        cen = verts[faces].mean(axis=1)
        pts = np.vstack([verts, cen])
        self.log(f"skin samples {len(pts)/1e3:.0f} k, KD-tree ...")
        tree = cKDTree(pts)
        s = f.copy()
        mask = (f > -band) & (f < band)
        idx = np.nonzero(mask)
        q = np.column_stack([self.xs[idx[0]], self.ys[idx[1]], self.zs[idx[2]]])
        D, _ = tree.query(q, workers=-1, distance_upper_bound=band * 2)
        D = np.where(np.isfinite(D), D, band * 2)
        s[idx] = (np.sign(f[idx]) * D).astype(np.float32)
        self.skin_verts, self.skin_faces = verts, faces
        return s

    # ------------------------------------------------------------------ helpers
    def interp(self, field):
        return RegularGridInterpolator((self.xs, self.ys, self.zs), field, bounds_error=False, fill_value=None)

    def surface_point(self, s_interp, y, z, side):
        """Point on the outer skin at height z / position y on the left (-1) or right (+1) side."""
        xs = np.linspace(0.0, side * (self.P["plan"]["half_width"] + self.P["lip"]["right_flare"] + 4), 4000)
        v = s_interp(np.column_stack([xs, np.full_like(xs, y), np.full_like(xs, z)]))
        i = np.nonzero((v[:-1] < 0) & (v[1:] >= 0))[0][0]
        t = v[i] / (v[i] - v[i + 1])
        return np.array([xs[i] + t * (xs[i + 1] - xs[i]), y, z])

    def grad(self, s_interp, p, e=0.15):
        g = np.array([(s_interp([p + d])[0] - s_interp([p - d])[0]) / (2 * e) for d in np.eye(3) * e])
        return g / np.linalg.norm(g)

    def pad_positions(self):
        P = self.P
        L = P["size"]["length"]
        b = P["bottom"]
        inset = b["pad_inset"]
        out = {}
        for name, yn, side in b["pads"]:
            if side == "C":
                # rear centre pad: on X=0, `inset` in front of the rear outline at Z=0
                ys = np.linspace(0, 30, 3001)
                v = self.s_interp(np.column_stack([np.zeros_like(ys), ys, np.full_like(ys, 0.05)]))
                y = ys[np.argmin(np.abs(v + inset))]
                out[name] = (0.0, float(y))
            else:
                y = yn * L
                sgn = -1 if side == "L" else 1
                xs = np.linspace(0, sgn * 40, 4001)
                v = self.s_interp(np.column_stack([xs, np.full_like(xs, y), np.full_like(xs, 0.05)]))
                ok = np.nonzero(v < -inset)[0]
                x = xs[ok[-1]]
                out[name] = (float(x), float(y))
        return out

    # ------------------------------------------------------------------ build
    def build(self, placeholders=True):
        P, X, Y, Z = self.P, self.X, self.Y, self.Z
        t = P["shell"]["wall"]
        b = P["bottom"]

        f = self.outer_field()
        self.log("exact distance ...")
        s = self.exact_sdf(f)
        self.s = s
        self.s_interp = self.interp(s)

        # ---- shell with open bottom
        shell = np.maximum(s, -s - t)

        # ---- windows
        W = np.full(self.shape, 1e3, dtype=np.float32)
        for w in P["windows"]:
            W = np.minimum(W, self.window_field(w))
        self.W = W
        kwin = P["shell"]["window_edge_round"]
        shell = S.smax(shell, -W, kwin)

        # ---- wheel opening (continuous rounded edge around it)
        wh = P["wheel"]
        Tw = float(self.top_height(np.array(0.0), np.array(wh["y"])))
        self.wheel_top_surface_z = Tw
        dW = S.sd_roundbox2d(self.X2, self.Y2 - wh["y"], wh["opening_w"] / 2, wh["opening_len"] / 2, wh["opening_corner_r"])
        Wwheel = np.maximum(dW[:, :, None], (Tw - 7.0) - Z).astype(np.float32)
        self.Wwheel = Wwheel
        shell = S.smax(shell, -Wwheel, wh["edge_round"])

        # ---- side-button bosses
        bosses = np.full(self.shape, 1e3, dtype=np.float32)
        self.boss_frames = []
        for sb in P["side_buttons"]:
            bosses = np.minimum(bosses, self.boss_field(sb, s, t))
        shell = np.minimum(shell, bosses)

        # ---- bottom skeleton
        pads = self.pad_positions()
        self.pads = pads
        ring = np.maximum(np.maximum(s, -s - (t + b["ring_width"])), Z - b["ring_height"])
        ribs2d = self.rib_field2d(pads)
        ribs = np.maximum(ribs2d[:, :, None], S.slab(Z, -2.0, b["rib_height"]))
        clip = max(b["rib_blend"], 0.6) + 0.2      # keep blended parts inside the skin (no bulge outward)
        ribs = np.maximum(ribs, s + clip)
        padf = np.full(self.shape, 1e3, dtype=np.float32)
        for (px, py) in pads.values():
            padf = np.minimum(padf, S.sd_cylinder_z(X, Y, Z, px, py, b["pad_d"] / 2, -2.0, b["pad_h"]))
        padf = np.maximum(padf, s + clip)
        sen = b["sensor"]
        dsq = S.sd_roundbox2d(self.X2 - sen["x"], self.Y2 - sen["y"], sen["outer_x"] / 2, sen["outer_y"] / 2, sen["corner_r"])
        frame2d = np.maximum(dsq, -(dsq + b["rib_width"]))
        frame = np.maximum(frame2d[:, :, None], S.slab(Z, -2.0, b["rib_height"]))
        kb = b["rib_blend"]
        self.shell_final = shell
        self.ring = ring
        self.inner = S.smin(ribs, frame, kb)
        # shell and perimeter flange share the outer skin -> plain union; ribs/frame/pads blend in with fillets
        body = S.smin(np.minimum(shell, ring), S.smin(ribs, frame, kb), kb)
        relief = b.get("bottom_relief", 0.0)
        if relief > 0:   # optional: lift everything except the pads off the pad plane (needs support when printed)
            body = np.maximum(body, relief - Z)
        solid = S.smin(body, padf, kb)
        self.pad_field = padf
        self.skeleton_no_placeholders = np.maximum(solid, -Z)

        # ---- V0.1 placeholders (fixed wheel + button posts)
        if placeholders:
            if wh["placeholder"]:
                solid = np.minimum(solid, self.wheel_placeholder(Tw))
            if b["button_posts"]:
                posts = np.full(self.shape, 1e3, dtype=np.float32)
                yb = pads["FL"][1]
                for key in ("lmb", "rmb"):
                    posts = np.minimum(posts, S.sd_cylinder_z(X, Y, Z, P["contact"][key]["x"], yb, b["post_d"] / 2, -2.0, 60.0))
                posts = np.maximum(posts, s + clip)
                solid = S.smin(solid, posts, kb)

        # 45 deg chamfer on the outer bottom edge (finger-friendly, limits elephant foot); pads sit deeper inside
        c = b.get("bottom_edge_chamfer", 0.0)
        if c > 0:
            solid = np.maximum(solid, (s - (Z - c)) / np.sqrt(2.0))
            self.skeleton_no_placeholders = np.maximum(self.skeleton_no_placeholders, (s - (Z - c)) / np.sqrt(2.0))
        F = np.maximum(solid, -Z).astype(np.float32)   # cut at the foot-pad plane
        self.F = F
        return F

    def window_field(self, w):
        X2, Y2, X, Y, Z = self.X2, self.Y2, self.X, self.Y, self.Z
        c1, c2 = w["c"]
        kind = w["kind"]
        if kind == "top":
            u, v = S.rot2(X2 - c1, Y2 - c2, w["ang"])
            d = S.sd_roundbox2d(u, v, w["len"] / 2, w["wid"] / 2, w["r"])
            return np.maximum(d[:, :, None], w["z_min"] - Z).astype(np.float32)
        if kind in ("side_L", "side_R"):
            Yz, Zy = np.meshgrid(self.ys, self.zs, indexing="ij")
            u, v = S.rot2(Yz - c1, Zy - c2, w["ang"])
            d = S.sd_roundbox2d(u, v, w["len"] / 2, w["wid"] / 2, w["r"])[None, :, :]
            lim = (X - w["lim"]) if kind == "side_L" else (w["lim"] - X)
            return np.maximum(d, lim).astype(np.float32)
        if kind in ("front", "rear"):
            Xz, Zx = np.meshgrid(self.xs, self.zs, indexing="ij")
            u, v = S.rot2(Xz - c1, Zx - c2, w["ang"])
            d = S.sd_roundbox2d(u, v, w["len"] / 2, w["wid"] / 2, w["r"])[:, None, :]
            lim = (w["lim"] - Y) if kind == "front" else (Y - w["lim"])
            return np.maximum(d, lim).astype(np.float32)
        raise ValueError(kind)

    def boss_field(self, sb, s, t):
        p0 = self.surface_point(self.s_interp, sb["y"], sb["z"], -1)
        n = self.grad(self.s_interp, p0)
        ey = np.array([0.0, 1.0, 0.0])
        u0 = ey - ey.dot(n) * n
        u0 /= np.linalg.norm(u0)
        w0 = np.cross(n, u0)
        if w0[2] < 0:
            w0 = -w0                         # tangent pointing up
        a = np.radians(sb["tilt_deg"])      # positive: front end of the boss lower
        u = np.cos(a) * u0 - np.sin(a) * w0
        u /= np.linalg.norm(u)
        v = np.cross(n, u)
        if v[2] < 0:
            v = -v
        dx, dy, dz = self.X - p0[0], self.Y - p0[1], self.Z - p0[2]
        du = dx * u[0] + dy * u[1] + dz * u[2]
        dv = dx * v[0] + dy * v[1] + dz * v[2]
        dn = dx * n[0] + dy * n[1] + dz * n[2]
        foot = S.sd_roundbox2d(du, dv, sb["length"] / 2, sb["height"] / 2, min(sb["height"] / 2, 3.0))
        k = self.P["side_button_edge_r"]
        f = S.smax(s - sb["protrude"], foot, k)
        f = np.maximum(f, -(s + t))
        f = np.maximum(f, np.abs(dn) - 6.0)
        ang = self.P.get("side_button_underside_deg")
        if ang:
            # self-supporting underside: plane through the lower footprint edge, face inclined `ang` from horizontal
            nh = np.array([n[0], n[1], 0.0])
            nh /= np.linalg.norm(nh)
            m = nh * np.sin(np.radians(ang)) - np.array([0, 0, 1.0]) * np.cos(np.radians(ang))
            q = p0 - v * (sb["height"] / 2)
            # the footprint edge lies on the tangent plane; move the hinge onto the curved skin (slightly inside)
            q = self.surface_point(self.s_interp, q[1], q[2], -1) - nh * 0.2
            f = np.maximum(f, (self.X - q[0]) * m[0] + (self.Y - q[1]) * m[1] + (self.Z - q[2]) * m[2])
        self.boss_frames.append({"name": sb["name"], "p0": p0, "n": n, "u": u, "v": v})
        return f.astype(np.float32)

    def rib_field2d(self, pads):
        b = self.P["bottom"]
        sen = b["sensor"]
        sx, sy, ox, oy = sen["x"], sen["y"], sen["outer_x"] / 2, sen["outer_y"] / 2
        yf = pads["FL"][1]
        segs = [
            (pads["RC"], (sx, sy - oy)),
            ((sx, sy + oy), (0.0, yf)),
            (pads["RL"], (sx - ox + 1, sy - oy + 1)),
            (pads["RR"], (sx + ox - 1, sy - oy + 1)),
            (pads["FL"], (sx - ox + 1, sy + oy - 1)),
            (pads["FR"], (sx + ox - 1, sy + oy - 1)),
            (pads["FL"], pads["FR"]),
            ((sx - ox, sy), (sx - 45.0, sy)),
            ((sx + ox, sy), (sx + 45.0, sy)),
        ]
        self.rib_segments = segs
        d = np.full(self.X2.shape, 1e3)
        for (a, c) in segs:
            d = np.minimum(d, S.sd_segment2d(self.X2, self.Y2, a[0], a[1], c[0], c[1]))
        return (d - b["rib_width"] / 2).astype(np.float32)

    def wheel_placeholder(self, Tw):
        wh = self.P["wheel"]
        X, Y, Z = self.X, self.Y, self.Z
        R = wh["radius"]
        zc = Tw + wh["protrude"] - R
        self.wheel_center = (0.0, wh["y"], zc)
        th = np.arctan2(Z - zc, Y - wh["y"])
        groove = wh["groove_depth"] * np.maximum(np.cos(wh["grooves"] * th), 0.0) ** 6
        disc = S.sd_rounded_cylinder_x(X, Y, Z, 0.0, wh["y"], zc, R, wh["width"] / 2, wh["rim_r"]) + groove
        ys = R * np.sin(np.radians(wh["pedestal_angle_deg"]))
        ped = np.maximum(np.maximum(np.abs(X) - wh["pedestal_half_w"], np.abs(Y - wh["y"]) - ys), S.slab(Z, -2.0, zc))
        self.wheel_disc = disc.astype(np.float32)
        return S.smin(disc, ped, 0.6).astype(np.float32)

    # ------------------------------------------------------------------ measurement
    def measure(self, mesh):
        P = self.P
        c = P["contact"]
        sk = self.skin_verts[self.skin_verts[:, 2] >= 0]
        th = self.surface_point(self.s_interp, c["thumb"]["y"], c["thumb"]["z"], -1)
        rg = self.surface_point(self.s_interp, c["ring"]["y"], c["ring"]["z"], +1)
        b = mesh.bounds
        return {
            "length_Y": float(b[1, 1] - b[0, 1]),
            "width_X_max": float(b[1, 0] - b[0, 0]),
            "height_Z_max": float(b[1, 2] - b[0, 2]),
            "bounds": b.tolist(),
            "skin_width_without_side_buttons": float(sk[:, 0].max() - sk[:, 0].min()),
            "skin_x_min": float(sk[:, 0].min()), "skin_x_max": float(sk[:, 0].max()),
            "skin_y_min": float(sk[:, 1].min()), "skin_y_max": float(sk[:, 1].max()),
            "thumb_contact_point": th.tolist(), "ring_contact_point": rg.tolist(),
            "grip_width_thumb_to_ring": float(rg[0] - th[0]),
        }

    # ------------------------------------------------------------------ meshing
    def mesh(self, F=None):
        import trimesh
        F = self.F if F is None else F
        h = self.h
        verts, faces, _, _ = marching_cubes(nonzero(F), 0.0, spacing=(h, h, h), gradient_direction="ascent")
        verts += np.array([self.xs[0], self.ys[0], self.zs[0]])
        # keep marching-cubes topology as is (vertex merging by position can create pinches)
        m = trimesh.Trimesh(verts, faces, process=False)
        if m.volume < 0:
            m.invert()
        return m


def calibrate(P, h=0.4, iters=3, log=print):
    """Adjust outline Y range, nominal half width and right lip so the built part meets the size targets.

    Targets: length (Y), grip width (thumb contact -> ring contact), max width (incl. side buttons).
    Height is set directly by the crest profile. Returns (calibrated P copy, history).
    """
    import copy
    Q = copy.deepcopy(P)
    L = Q["size"]["length"]
    Q["plan"].setdefault("y0", 0.0)
    Q["plan"].setdefault("y1", L)
    hist = []
    for it in range(iters):
        g = TrialGrip(Q, h=h, log=lambda *a: None)
        g.build(placeholders=False)
        m = g.measure(g.mesh())
        hist.append({"iter": it, "y0": Q["plan"]["y0"], "y1": Q["plan"]["y1"], "half_width": Q["plan"]["half_width"],
                     "right_flare": Q["lip"]["right_flare"], **{k: m[k] for k in ("length_Y", "width_X_max", "height_Z_max", "grip_width_thumb_to_ring")}})
        log(f"  calib {it}: L={m['length_Y']:.2f} W={m['width_X_max']:.2f} H={m['height_Z_max']:.2f} grip={m['grip_width_thumb_to_ring']:.2f}")
        b = m["bounds"]
        Q["plan"]["y0"] -= b[0][1] - 0.0
        Q["plan"]["y1"] += L - b[1][1]
        Q["plan"]["half_width"] += 0.5 * (Q["size"]["grip_width"] - m["grip_width_thumb_to_ring"])
        Q["lip"]["right_flare"] += Q["size"]["width_max"] - m["width_X_max"] - 0.5 * (Q["size"]["grip_width"] - m["grip_width_thumb_to_ring"]) * 2
    return Q, hist
