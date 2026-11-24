"""toon/world.py — render_scene(frame_idx, fps, scene, talk_lookup) -> BGR frame.
scene = ek scene dict {"room","camera_move","duration","characters":[{name,slot,expr,line,(walk_to),(props)}]};
frame_idx us scene ke andar ka index. Agar scene mein "scenes" list ho to frame_idx global maana jata hai.
talk_lookup(name, t): t = t_offset + scene-local seconds (single scene mein t_offset=0)."""
import math
import numpy as np
from .core import R, W, H, clamp, ease, lerp
from .lighting import LightFX
from .shadows import Shadows
from .rooms import ROOMS, draw_obj
from .cameras import CAMERA_MOVES

try:
    from toon.characters import CHARS, draw_char          # PART B
except Exception:                                          # stub: prototype ka draw_char
    from ._stub_char import CHARS, draw_char


class Renderer(R, LightFX, Shadows):
    pass


def _pick(scene, fi, fps):
    """Multi-scene JSON ho to global frame se scene dhoondo."""
    if "scenes" not in scene: return scene, fi, 0.0
    acc = 0
    for sc in scene["scenes"]:
        n = int(round(sc.get("duration", 3.0) * fps))
        if fi < acc + n or sc is scene["scenes"][-1]: return sc, fi - acc, acc / fps
        acc += n


def _char(name):
    if name in CHARS: return CHARS[name]
    ks = sorted(CHARS); return CHARS[ks[sum(map(ord, str(name))) % len(ks)]]


def render_scene(frame_idx, fps, scene, talk_lookup=None, t_offset=0.0):
    scene, fi, off = _pick(scene, frame_idx, fps); t_offset += off
    room = ROOMS[scene.get("room", "office")]; dur = max(0.1, scene.get("duration", 3.0))
    t = fi / fps; u = clamp(t / dur); slots = room["slots"]
    # --- characters ki positions/state
    items = []; cl = scene.get("characters", [])
    for i, c in enumerate(cl):
        sl = c.get("slot", "center"); sit = sl == "seat" and "seat" in slots
        a = slots.get(sl, slots["center"])["pos"]; walk = False; ph = 0.0
        if c.get("walk_to") in slots:
            b = slots[c["walk_to"]]["pos"]; e = ease(u); a = (lerp(a[0], b[0], e), lerp(a[1], b[1], e)); walk = 0.02 < e < 0.98; sit = False
            ph = t * 8
        name = c.get("name", "guy")
        tk = float(talk_lookup(name, t_offset + t)) if talk_lookup else (abs(math.sin(t * 12)) * .6 if c.get("line") else 0.0)
        others = [slots.get(o.get("slot", "center"), slots["center"])["pos"][0] for j, o in enumerate(cl) if j != i]
        lx = (-1 if others and others[0] < a[0] else 1) * .7 if others else math.sin(t * 1.3 + i) * .5
        props = c.get("props", []) or []
        sw = math.sin(ph) * .08 if walk else 0.0
        if sit: arms = {"l": (-.26, .98), "r": (.26, .98 + (.15 * tk * math.sin(t * 7 + i) if tk > .1 else .0))}
        else: arms = {"l": (-.30, .80 + sw), "r": (.30 if tk < .1 else .34, .80 - sw + (.2 * tk * (1 + math.sin(t * 7 + i)) / 2 if tk > .1 else 0))}
        st = dict(t=t, walk=walk, ph=ph, sit=sit, expr=c.get("expr", "smile"), look=(lx, 0), talk=clamp(tk), arms=arms,
                  props=props, phone="phone" in props, key="key" in props)
        items.append((np.array([a[0], 0.0, a[1]], float), _char(name), st))
    pos = [p for p, _, _ in items]
    txz = (float(np.mean([p[0] for p in pos])), float(np.mean([p[2] for p in pos]))) if pos else (0.0, room["dims"][2] * .4)
    cam = CAMERA_MOVES.get(scene.get("camera_move", "push_in"), CAMERA_MOVES["push_in"])(u, room, txz)
    # --- render: shell -> shadows -> objects/characters (door se paas tak) -> light fx
    fr = np.empty((H, W, 3), np.uint8); r = Renderer(fr, cam, frame_idx, room)
    room["shell_draw"](r)
    for o in room["objs"]:
        for b in o["boxes"]: r.shadow_box(*b)
    for p, ch, st in items: r.shadow_char(p, sit=st["sit"])      # draw_char se pehle
    r.apply_shadows()
    order = [(cam.cam(o["c"])[0, 2], 0, o) for o in room["objs"]] + [(cam.cam(p)[0, 2], 1, (p, ch, st)) for p, ch, st in items]
    base_amb = r.amb.copy()
    for _, kind, o in sorted(order, key=lambda x: -x[0]):
        if kind == 0: draw_obj(r, o)
        else:
            p, ch, st = o; r.amb = base_amb * r.lk(p + np.array([0, 1.1, 0]))   # character par room ki roshni ka rang
            draw_char(r, p, ch, st); r.amb = base_amb
    r.run_fx(room["fx"]); r.finish(room["tone"])
    return fr
