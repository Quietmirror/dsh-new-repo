"""Orthographic VTK renders with millimetre axes (run under xvfb-run on headless machines)."""
import numpy as np
import vtk
from vtk.util import numpy_support as ns
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# camera direction (from target to eye), view-up, horizontal/vertical world axes shown in the image
VIEWS = {
    "top":    {"dir": (0, 0, 1), "up": (0, 1, 0), "axes": ("X", "Y"), "title": "俯视 Top (+Z)"},
    "bottom": {"dir": (0, 0, -1), "up": (0, 1, 0), "axes": ("-X", "Y"), "title": "底视 Bottom (-Z, 左右镜像)"},
    "left":   {"dir": (-1, 0, 0), "up": (0, 0, 1), "axes": ("-Y", "Z"), "title": "左侧视 Left / thumb side (-X)"},
    "right":  {"dir": (1, 0, 0), "up": (0, 0, 1), "axes": ("Y", "Z"), "title": "右侧视 Right / ring-pinky side (+X)"},
    "front":  {"dir": (0, 1, 0), "up": (0, 0, 1), "axes": ("-X", "Z"), "title": "前视 Front (+Y)"},
    "rear":   {"dir": (0, -1, 0), "up": (0, 0, 1), "axes": ("X", "Z"), "title": "后视 Rear (-Y)"},
    "iso":    {"dir": (-0.75, -0.95, 0.85), "up": (0, 0, 1), "axes": None, "title": "斜视 Iso (rear-left-top)"},
    "iso_fr": {"dir": (0.8, 1.0, 0.75), "up": (0, 0, 1), "axes": None, "title": "斜视 Iso (front-right-top)"},
    "iso_bottom": {"dir": (-0.7, -0.6, -0.9), "up": (0, 0, 1), "axes": None, "title": "斜视 Iso (from below)"},
}


def polydata(mesh, vcolors=None):
    pts = vtk.vtkPoints()
    pts.SetData(ns.numpy_to_vtk(np.ascontiguousarray(mesh.vertices, dtype=np.float64), deep=True))
    f = np.asarray(mesh.faces, dtype=np.int64)
    cells = np.hstack([np.full((len(f), 1), 3, dtype=np.int64), f]).ravel()
    ca = vtk.vtkCellArray()
    ca.SetCells(len(f), ns.numpy_to_vtkIdTypeArray(cells, deep=True))
    pd = vtk.vtkPolyData()
    pd.SetPoints(pts)
    pd.SetPolys(ca)
    if vcolors is not None:
        arr = ns.numpy_to_vtk(np.ascontiguousarray(vcolors, dtype=np.uint8), deep=True)
        arr.SetName("rgb")
        pd.GetPointData().SetScalars(arr)
    nf = vtk.vtkPolyDataNormals()
    nf.SetInputData(pd)
    nf.SetFeatureAngle(50)
    nf.SplittingOn()
    nf.Update()
    return nf.GetOutput()


def render(items, view, out_png, size=(1600, 1200), bounds=None, title=None, margin=6.0, annotate=None, colorbar=None):
    """items: list of dicts {mesh, color, opacity}. Writes a PNG with mm axes for orthographic views."""
    v = VIEWS[view]
    ren = vtk.vtkRenderer()
    ren.SetBackground(1, 1, 1)
    allv = []
    for it in items:
        m = it["mesh"]
        allv.append(m.vertices)
        mp = vtk.vtkPolyDataMapper()
        vc = it.get("vcolors")
        mp.SetInputData(polydata(m, vc))
        if vc is None:
            mp.ScalarVisibilityOff()
        else:
            mp.ScalarVisibilityOn()
            mp.SetColorModeToDirectScalars()
        a = vtk.vtkActor()
        a.SetMapper(mp)
        pr = a.GetProperty()
        pr.SetColor(*it.get("color", (0.82, 0.84, 0.88)))
        pr.SetOpacity(it.get("opacity", 1.0))
        pr.SetAmbient(0.22)
        pr.SetDiffuse(0.75)
        pr.SetSpecular(0.18)
        pr.SetSpecularPower(25)
        if it.get("edges"):
            pr.EdgeVisibilityOn()
            pr.SetEdgeColor(0.3, 0.3, 0.3)
        ren.AddActor(a)
    allv = np.vstack(allv)
    if bounds is None:
        lo, hi = allv.min(0) - margin, allv.max(0) + margin
    else:
        lo, hi = np.array(bounds[0]), np.array(bounds[1])
    c = (lo + hi) / 2
    d = np.array(v["dir"], float)
    d /= np.linalg.norm(d)
    up = np.array(v["up"], float)
    cam = ren.GetActiveCamera()
    cam.SetFocalPoint(*c)
    cam.SetPosition(*(c + d * 400))
    cam.SetViewUp(*up)
    cam.ParallelProjectionOn()
    W, H = size
    aspect = W / H
    # extent of the bounding box projected on the image plane
    right = np.cross(-d, up)
    right /= np.linalg.norm(right)
    upv = np.cross(right, -d)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]) - c
    hx = np.abs(corners @ right).max()
    hy = np.abs(corners @ upv).max()
    ps = max(hy, hx / aspect)
    cam.SetParallelScale(ps)
    ren.ResetCameraClippingRange()
    lk = vtk.vtkLightKit()
    lk.SetKeyLightIntensity(0.9)
    lk.AddLightsToRenderer(ren)
    win = vtk.vtkRenderWindow()
    win.SetOffScreenRendering(1)
    win.SetMultiSamples(8)
    win.AddRenderer(ren)
    win.SetSize(W, H)
    win.Render()
    w2i = vtk.vtkWindowToImageFilter()
    w2i.SetInput(win)
    w2i.ReadFrontBufferOff()
    w2i.Update()
    img = ns.vtk_to_numpy(w2i.GetOutput().GetPointData().GetScalars()).reshape(H, W, -1)[::-1]

    fig, ax = plt.subplots(figsize=(W / 160, H / 160), dpi=160)
    hc, vc = c @ right, c @ upv
    ext = [hc - ps * aspect, hc + ps * aspect, vc - ps, vc + ps]
    ax.imshow(img, extent=ext, interpolation="lanczos")
    if v["axes"]:
        ax.set_xlabel(f"{v['axes'][0].lstrip('-')}  [mm]" + ("  (数值为实际坐标, 方向镜像)" if v['axes'][0].startswith('-') else ""))
        ax.set_ylabel(f"{v['axes'][1]}  [mm]")
        ax.grid(alpha=0.25, lw=0.5)
        ax.xaxis.set_major_locator(plt.MultipleLocator(10))
        ax.yaxis.set_major_locator(plt.MultipleLocator(10))
        if v["axes"][0].startswith("-"):
            ax.set_xticks(ax.get_xticks())
            ax.set_xticklabels([f"{-t:g}" for t in ax.get_xticks()])
            ax.set_xlim(ext[0], ext[1])
    else:
        ax.set_axis_off()
    if annotate:
        annotate(ax, right, upv)
    if colorbar:
        sm = plt.cm.ScalarMappable(cmap=colorbar["cmap"], norm=plt.Normalize(colorbar["vmin"], colorbar["vmax"]))
        cb = fig.colorbar(sm, ax=ax, fraction=0.035, pad=0.02)
        cb.set_label(colorbar["label"])
    ax.set_title(title or v["title"], fontsize=11)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return out_png
