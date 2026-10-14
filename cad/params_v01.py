"""V0.1 trial-grip parameters (millimetres / degrees).

Coordinate system used everywhere in this project:
  X  + to the right (ring/pinky side), - to the left (thumb side), X=0 is the main-button centreline
  Y  0 at the rearmost point, + toward the front (buttons)
  Z  0 = foot-pad plane (bottom faces of the five skate platforms), + up

Every value here is a TRIAL ASSUMPTION for the first grip print, not a validated optimum.
Hand data from the user (approximate AR photo measurement, NOT used to scale the shell):
  wrist crease -> middle fingertip ~230 mm, thumb tip -> pinky tip spread ~190 mm.
Palm width, phalanx lengths and the real grip posture are not measured yet.
"""

P = {
    "meta": {
        "name": "fingertip_mouse_v01_trial_grip",
        "version": "0.1",
        "hand_length_mm_approx": 230,
        "hand_span_thumb_pinky_mm_approx": 190,
        "unmeasured": ["palm width", "phalanx lengths", "fingertip contact positions in real grip"],
    },
    # ---------------- overall size targets ----------------
    "size": {
        "length": 90.0,          # Y extent of the shell
        "width_max": 62.0,       # max outer X extent (incl. right lip)
        "grip_width": 54.0,      # thumb-groove contact to ring-finger contact, measured horizontally
        "height": 31.0,          # max Z above the foot-pad plane (skates add ~0.6-0.8 mm on top of this)
    },
    # ---------------- plan view (top) outline, nominal side walls ----------------
    # normalised control points (y/L, half-width) for the right half, rear centre -> front centre.
    # Left half mirrors it with `left_scale`.
    "plan": {
        "half_width": 28.0,      # nominal max half width of the side walls (before groove/lip)
        "left_scale": 1.0,
        "ctrl": [  # (y/L, w/half_width)
            (0.000, 0.00), (0.012, 0.36), (0.055, 0.62), (0.125, 0.79), (0.23, 0.905),
            (0.37, 0.975), (0.52, 1.00), (0.66, 1.00), (0.77, 0.985), (0.86, 0.94),
            (0.925, 0.84), (0.972, 0.62), (0.994, 0.30), (1.000, 0.00),
        ],
    },
    # ---------------- side profile of the crest line (top centre) ----------------
    "top": {
        # (y/L, z/height) control points, PCHIP interpolated; rear is short and low (avoid palm contact)
        "crest": [
            (0.00, 0.36), (0.04, 0.48), (0.11, 0.615), (0.22, 0.765), (0.34, 0.90), (0.46, 0.98), (0.54, 1.00),
            (0.63, 0.98), (0.72, 0.925), (0.81, 0.85), (0.89, 0.775), (0.96, 0.72), (1.00, 0.69),
        ],
        "crown_drop": [(0.0, 7.0), (0.35, 5.5), (0.65, 3.6), (1.0, 3.0)],  # z drop at |x|=half_width
        "crest_x": 0.0,          # lateral position of the crest
        "tilt_deg": 0.0,         # + lowers the right side of the top (pronation), 0 = symmetric
        "blend": [(0.0, 11.0), (0.40, 9.5), (0.80, 7.5), (1.0, 5.5)],  # side/top rounding blend (mm)
    },
    # ---------------- finger contact areas (all continuous, no holes) ----------------
    "contact": {
        "lmb": {"x": -14.0, "y": 77.0, "rx": 7.5, "ry": 10.5, "depth": 0.9},   # index fingertip dish
        "rmb": {"x": 14.0, "y": 78.5, "rx": 7.5, "ry": 10.5, "depth": 0.9},    # middle fingertip dish
        "thumb": {"y": 54.0, "z": 10.5, "ay": 18.0, "az": 7.5, "depth": 1.6},  # left shallow groove
        "ring": {"y": 66.0, "z": 12.0, "ay": 14.0, "az": 8.0, "depth": 0.5},   # right ring-finger rest
    },
    # right-side anti-drag lip (pinky/ring support) and a small left bottom bulge
    "lip": {
        "right_flare": 4.5, "right_z": 9.5, "right_y0": 0.30, "right_y1": 0.72, "right_ramp": 0.14,
        "left_flare": 0.0, "left_z": 6.0,
    },
    # ---------------- thumb side buttons: fixed tactile bosses in V0.1 ----------------
    "side_buttons": [
        {"name": "SB_front", "y": 61.5, "z": 20.5, "length": 15.5, "height": 7.0, "protrude": 1.6, "tilt_deg": 4.0},
        {"name": "SB_rear", "y": 43.0, "z": 20.5, "length": 15.5, "height": 7.0, "protrude": 1.6, "tilt_deg": 4.0},
    ],
    "side_button_edge_r": 1.5,   # blend width; gives ~R1.3 edge rounding
    "side_button_underside_deg": 58.0,  # underside ramp angle from horizontal (>45 = printable without support); None = square
    # ---------------- scroll wheel ----------------
    "wheel": {
        "y": 75.5,                 # axle Y
        "opening_len": 22.0, "opening_w": 9.6, "opening_corner_r": 3.0, "edge_round": 1.6,
        "placeholder": True,       # fixed dummy wheel tied to the skeleton (delete in assembly version)
        "radius": 10.0, "width": 7.0, "rim_r": 1.2, "protrude": 2.6,  # protrude above local top surface
        "grooves": 18, "groove_depth": 0.45,
        "pedestal_half_w": 3.0, "pedestal_angle_deg": 50.0,
    },
    # ---------------- shell & skeleton ----------------
    "shell": {
        "wall": 1.7,               # FDM 0.4 nozzle: 1.6-1.8 for the first round
        "window_edge_round": 1.2,  # rounding of window edges
    },
    "bottom": {
        "ring_width": 2.5,         # perimeter flange inside the wall at the bottom
        "ring_height": 3.0,
        "rib_width": 2.5,
        "rib_height": 3.0,
        "rib_blend": 0.8,
        "pad_d": 8.5,              # skate platform diameter (for 7-7.5 mm dot skates)
        "pad_h": 3.0,              # = rib height -> flat grid top for the future main board
        "pad_inset": 6.0,          # pad centre distance from the outer bottom outline
        "bottom_edge_chamfer": 0.6,  # 45 deg chamfer on the outer bottom edge
        "bottom_relief": 0.0,      # >0 lifts ring/ribs/shell rim above the pad plane (V0.1: 0 = flat, support-free bottom)
        # (name, y/L or None, side) ; side in {"L","R","C"}; rear-centre pad found from the rear outline
        "pads": [("RC", None, "C"), ("RL", 0.26, "L"), ("RR", 0.26, "R"), ("FL", 0.875, "L"), ("FR", 0.875, "R")],
        # sensor locating frame; inner opening must clear the lens courtyard (paw3395-pcb B.CrtYd 19.0 x 21.75)
        "sensor": {"x": 0.0, "y": 45.0, "outer_x": 26.0, "outer_y": 30.0, "corner_r": 3.0},
        "button_posts": True,      # V0.1 only: solid posts under the main-button dishes
        "post_d": 4.5,
    },
    # ---------------- weight-reduction windows (never in contact areas) ----------------
    # kind: "top" (cut along Z, only above z_min), "side_L"/"side_R" (cut along X), "rear"/"front" (cut along Y)
    # (c1,c2) = centre in the projection plane, len/wid, angle (deg), corner radius
    "windows": [
        {"kind": "top", "c": (-9.5, 33.0), "len": 30.0, "wid": 8.5, "ang": 90.0, "r": 4.25, "z_min": 14.0},
        {"kind": "top", "c": (9.5, 33.0), "len": 30.0, "wid": 8.5, "ang": 90.0, "r": 4.25, "z_min": 14.0},
        {"kind": "top", "c": (0.0, 11.5), "len": 14.0, "wid": 6.5, "ang": 0.0, "r": 3.25, "z_min": 9.0},
        {"kind": "side_L", "c": (21.0, 9.0), "len": 22.0, "wid": 7.0, "ang": 8.0, "r": 3.5, "lim": -8.0},
        {"kind": "side_R", "c": (14.0, 10.0), "len": 13.0, "wid": 6.0, "ang": 12.0, "r": 3.0, "lim": 8.0},
        {"kind": "front", "c": (0.0, 7.5), "len": 22.0, "wid": 5.0, "ang": 0.0, "r": 2.5, "lim": 70.0},
    ],
    # contact keep-out zones used by the checker (windows must stay >= margin away)
    "keepout_margin": 2.0,          # windows vs any contact zone
    "keepout_margin_wheel": 1.5,    # wheel opening vs fingertip dishes / other zones
    # ---------------- electronics reference envelopes (NOT assembly dimensions) ----------------
    # src tags:  M = measured from an open-source file (named),  B = from a reference BOM / README,
    #            S = industry-standard nominal, E = ESTIMATE (placeholder until measured / datasheet)
    "electronics": {
        "skate_thickness": (0.65, "B: olliebella README assumes ~0.65 mm dot skates; measure your dots"),
        "lens_to_surface": (2.40, "B: olliebella README 'lens-to-surface ~2.4 mm (+/-0.2)' (PAW3950/3395 + LOAE-LSI1); confirm with lens datasheet"),
        "lens_ref_to_pcb_bottom": (1.25, "E: unknown; chosen so the PCB bottom lands on the 3.0 mm rib top - MUST be measured"),
        "pcb_thickness": (1.6, "M: paw3395-pcb KiCad (1.6 mm); donor board thickness unknown"),
        "pcb_top_components": (3.0, "E: allowance for components on the main-board top side"),
        "lens_courtyard": ((-9.5, 9.5, -11.02, 10.73), "M: paw3395-pcb PAW3395.kicad_mod B.CrtYd, relative to optical centre"),
        "pcb_aperture": ((8.6, 17.26), "M: paw3395-pcb footprint Edge.Cuts sensor aperture"),
        "sensor_body": ((12.5, 18.0, 4.5), "E: PAW3950 package envelope, replace with datasheet outline"),
        "breakout_board": ((15.45, 28.0, 1.6, 4.6), "M: paw3395-pcb Edge.Cuts 15.45x28.0, optical centre 4.6 mm off board centre"),
        "main_switch": ((6.0, 6.0, 7.3), "B: olliebella BOM Kailh CMI627301D07 6x6x7.3"),
        "side_switch": ((6.0, 6.0, 4.3), "E: generic 6x6 tact; replace with chosen side switch"),
        "side_board": ((34.0, 10.0, 1.0), "E: two-switch side PCB"),
        "encoder": ((6.5, 12.5, 12.5), "E: vertical encoder body; olliebella uses ALPS EC05E1220401 - check datasheet"),
        "wheel_axle": ((2.0, 19.0), "E: axle diameter / length"),
        "mmb_switch": ((6.0, 6.0, 5.0), "E: middle-click switch"),
        "battery": {"301230": ((30.0, 12.0, 3.0), "B: olliebella BOM 301230 LiPo (TxWxL naming)"),
                    "602030": ((30.0, 20.0, 6.0), "E: larger 602030 option (TxWxL naming)")},
        "battery_plug": ((6.0, 4.5, 6.0), "E: PH2.0 2P (olliebella CAD lists PH2.0 2P socket)"),
        "usb_c": ((8.94, 7.35, 3.26), "S/E: mid-mount Type-C receptacle typical body"),
        "usb_c_plug_overmold": ((12.4, 15.0, 6.6), "E: cable overmold clearance in front of the face"),
        "power_switch": ((9.0, 4.0, 3.5), "E: MSK-12C02 / MSK-1153 slide switch incl. lever travel"),
        "antenna_keepout": ((8.0, 18.0, 9.0), "E: 2.4 GHz PCB antenna + clearance, position depends on board design"),
        "wire_bundle_d": (2.0, "E: 2-3 x 28-30 AWG silicone"),
        "clearance": 0.5,
    },
}

