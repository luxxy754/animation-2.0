"""SIRF TEST KE LIYE: PART A ka asli toon/world.py na ho to prototype (room_demo_code.py) ke room/renderer se render_scene ka stub.
Chalane ke liye: STUB_WORLD=1 python render_story.py"""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import room_demo_code as rd  # noqa: E402
from toon.characters import CHARS, draw_char  # noqa: E402

SLOT_POS = {"left": (-1.3, 6.4), "center": (0.0, 7.0), "right": (1.3, 6.4), "far": (0.4, 10.2), "seat": (-0.2, 6.6)}
ROOMS = {k: {s: {"pos": SLOT_POS[s]} for s in SLOT_POS} for k in ("office", "corridor", "break_room", "night_exterior")}
P = lambda *a: np.array(a, float)


def render_scene(frame_idx, fps, scene, talk_lookup):
    t = frame_idx / fps
    u = min(1.0, t / max(0.1, scene["duration"]))
    night = scene["room"] in ("night_exterior", "corridor")
    mv = scene["camera_move"]
    z = 3.4 + (1.0 * u if mv != "pull_out" else 1.0 * (1 - u))
    x = {"pan": -0.8 + 1.6 * u, "orbit": -1.0 + 2 * u}.get(mv, 0.0)
    cam = rd.Cam(P(x, 1.5 if mv != "low_angle" else 0.6, z), math.atan2(-x * .3, 1), -0.03 + (0.1 * u if mv == "tilt" else 0), 0, 78)
    fr = np.zeros((rd.H, rd.W, 3), np.uint8)
    r = rd.R(fr, cam, frame_idx, (1, 1, 1))
    r.lights = [{"pos": (-4, 2.5, 6), "color": (70, 225, 255), "intensity": 1.0, "kind": "sun"},
                {"pos": (0, 3.1, 6.5), "color": (255, 255, 255), "intensity": 0.8, "kind": "panel"}] if not night else \
               [{"pos": (-1.3, 2.1, 11.9), "color": (130, 225, 255), "intensity": 1.2, "kind": "lamp"}]
    rd.room(r, night)
    objs = list(rd.room_objs(night))
    for c in scene["characters"]:
        px, pz = SLOT_POS[c["slot"]]
        st = dict(t=t, expr=c["expr"], look=(math.sin(t * 2 + len(c["name"])) * .6, 0), talk=talk_lookup(c["name"], t), sit=c["slot"] == "seat",
                  props=["coffee"] if scene["room"] == "break_room" else [])
        objs.append((P(px, 0, pz), (lambda rr, p=P(px, 0, pz), ch=CHARS[c["name"]], st=st: draw_char(rr, p, ch, st))))
    for _, fn in sorted(objs, key=lambda o: -cam.cam(o[0])[0, 2]):
        fn(r)
    return fr
