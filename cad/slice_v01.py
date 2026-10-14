"""Slice the trial grip with PrusaSlicer CLI and record dimensions / filament / time.

    python3 cad/slice_v01.py out/v0.1/fingertip_v01_trial_grip.stl
Runs two jobs: with organic build-plate-only supports (real print) and without supports (part-only weight).
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + p.stderr


def gcode_stats(path):
    s = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = re.match(r"; (filament used \[(?:mm|cm3|g)\]|total filament used \[g\]|estimated printing time \(normal mode\)) = (.*)", line)
            if m:
                s[m.group(1)] = m.group(2).strip()
    layers = 0
    with open(path, errors="ignore") as f:
        for line in f:
            if line.startswith(";LAYER_CHANGE"):
                layers += 1
    s["layers"] = layers
    return s


def main(stl):
    out = os.path.dirname(os.path.abspath(stl))
    ini = os.path.join(HERE, "prusa_v01.ini")
    info = run(["prusa-slicer", "--info", stl])
    inf = dict(re.findall(r"^(\w+) = (.*)$", info, re.M))
    res = {"slicer": run(["prusa-slicer", "--help"]).splitlines()[0], "profile": "cad/prusa_v01.ini", "info": inf}
    base = ["prusa-slicer", "--export-gcode", "--load", ini, "--center", "125,105"]
    jobs = {"with_supports": [], "no_supports": ["--support-material=0", "--support-material-auto=0"]}
    for name, extra in jobs.items():
        g = os.path.join(out, f"slice_{name}.gcode")
        log = run(base + extra + ["--output", g, stl])
        res[name] = gcode_stats(g) if os.path.exists(g) else {"error": log[-2000:]}
    with open(os.path.join(out, "slice_report.json"), "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(sys.argv[1])
