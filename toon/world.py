"""Scene renderer (REFERENCE-STYLE v2).
videoplayback.mp4 jaisa look: rangeen textured office rooms (khaali neela aasman/dabbe nahi), anda-sar safed characters mota kaala outline,
tirchhe (dutch) camera angles, saturated colour grade. Gemini ka dynamic world_spec ab fixed rich rooms par map hota hai.
"""
import math
import os
import numpy as np
import cv2

from .core import R, W, H, clamp, ease, lerp
from .lighting import LightFX
from .shadows import Shadows
from .rooms import ROOMS, draw_obj
from .cameras import CAMERA_MOVES
from .world_builder import DEFAULT_SPEC, validate_world_spec, nearest_fixed_room, draw_world
from .rooms import Wall, _wallks, _text_tex

try:
    from .characters import CHARS, draw_char
except Exception:
    from ._stub_char import CHARS, draw_char


# ======================================================================= REFERENCE STYLE PATCH
USE_DYNAMIC_WORLDS = os.environ.get("USE_DYNAMIC_WORLDS", "0") == "1"   # 1 = purana khaali outdoor world_spec wapas
OUTLINE_W = .034                                                       # mota cartoon outline (pehle .018)
SAT, CONTRAST = 1.22, 1.07                                             # colour grade (reference jaisa saturated)
FACE = (232, 242, 250)                                                 # BGR cream-safed chehra
_STYLE_DONE = False


def _apply_reference_style():
    """rig.py ka outline mota karo aur tamam characters ko anda-sar / patli taangon wala banao (sirf ek baar)."""
    global _STYLE_DONE
    if _STYLE_DONE:
        return
    _STYLE_DONE = True
    try:
        from . import rig
        path = rig.__file__
        text = open(path, encoding="utf-8").read()
        text = text.replace("width(.018)", "width(%s)" % OUTLINE_W)
        text = text.replace("OUT = (18, 15, 22)", "OUT = (8, 6, 10)")
        exec(compile(text, path, "exec"), rig.__dict__)                 # _draw / draw_char naye outline ke saath dobara bane
    except Exception as e:
        print("[style] rig patch skip:", e)
    for ch in CHARS.values():
        ch["skin"] = FACE
        ch["head_w"], ch["head_h"] = .255, .275                         # bada gol anda sar
        ch["body_w"], ch["body_h"] = .15, .27                           # patla jism
        ch["limb_w"] = .032                                             # mote stick haath-pair
        ch["shoe"] = (20, 18, 24)
    if "boss" in CHARS:
        CHARS["boss"]["hair"] = (150, 148, 150)
    if "uncle" in CHARS:
        CHARS["uncle"]["hair"] = (135, 135, 140)


_apply_reference_style()


_FIRE = _text_tex(["FIRE", "PULL DOWN"], [58, 22], [(255, 255, 255), (255, 255, 255)], (35, 35, 200), (220, 170), (20, 20, 120))
_LAW = _text_tex(["LABOR", "LAW"], [34, 34], [(30, 30, 40), (30, 30, 40)], (60, 200, 245), (150, 110), (30, 30, 40))
_BOARD = _text_tex(["NOTICE", "TO DO", "MEETING"], [30, 26, 26], [(40, 40, 200), (30, 30, 40), (30, 30, 40)], (180, 215, 240), (260, 190), (60, 90, 140))


def _extra_decor(r, room):
    """Office/corridor ki khaali deewaron par reference jaisi cheezein: fire alarm, clock, notice board, safe-red button."""
    if room.get("name") not in ("office", "corridor", "break_room"):
        return
    try:
        dims = room["dims"]
        k = _wallks(dims, room["lights"], room.get("amb0", .74))
        B, Rt = Wall("back", dims, k["back"]), Wall("right", dims, k["right"])
        if room["name"] == "office":
            B.fill(r, 2.05, 2.75, 1.0, 1.85, (60, 60, 70), 3)           # fire alarm box
            B.tex(r, _FIRE, 2.1, 2.7, 1.05, 1.8, .03)
            B.circ(r, 2.4, 2.1, .13, (40, 40, 205), 3, .03)             # laal button
            B.circ(r, 3.3, 2.6, .30, (245, 245, 245), 3)                # wall clock
            B.line(r, 3.3, 2.6, 3.3, 2.82, OUT_COL, 3, .02)
            B.line(r, 3.3, 2.6, 3.45, 2.52, OUT_COL, 3, .02)
            B.tex(r, _LAW, 3.0, 3.6, 1.2, 1.7, .03)
        else:
            B.tex(r, _BOARD, 1.0, 2.4, 1.4, 2.3, .03)
            B.circ(r, -1.6, 2.6, .30, (245, 245, 245), 3)
            B.line(r, -1.6, 2.6, -1.6, 2.82, OUT_COL, 3, .02)
            B.line(r, -1.6, 2.6, -1.45, 2.52, OUT_COL, 3, .02)
    except Exception as e:
        print("[decor] skip:", e)


OUT_COL = (22, 18, 26)


def _grade(frame):
    """Saturation + halka contrast: flat-cartoon look (reference jaisa)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] = np.clip(hsv[..., 1] * SAT, 0, 255)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32)
    out = (out - 128.0) * CONTRAST + 128.0
    frame[:] = np.clip(out, 0, 255).astype(np.uint8)


def _dutch_cam(cam, scene, u):
    """Kuch scenes mein halka tirchha camera (roll) + fov variation, taake har shot alag lage."""
    from .core import Cam
    h = sum(map(ord, str(scene.get("location", "")) + str(scene.get("camera_move", "")))) + int(float(scene.get("duration", 3)) * 10)
    roll = (-1, 0, 1, 0, 1, -1)[h % 6] * (.05 + .03 * math.sin(u * math.pi))
    if not roll or getattr(cam, "roll", 0):
        return cam
    f = cam.f
    hfov = math.degrees(2 * math.atan((W / 2) / f))
    return Cam(cam.pos, cam.yaw, cam.pitch, roll, hfov)


class Renderer(R, LightFX, Shadows):
    pass


def _pick(scene, fi, fps):
    if "scenes" not in scene:
        return scene, fi, 0.0
    acc = 0
    for i, sc in enumerate(scene["scenes"]):
        n = int(round(float(sc.get("duration", 3.0)) * fps))
        if fi < acc + n or i == len(scene["scenes"]) - 1:
            return sc, fi - acc, acc / fps
        acc += n
    return scene["scenes"][-1], 0, 0.0


def _char(name):
    if name in CHARS:
        return CHARS[name]
    keys = sorted(CHARS)
    return CHARS[keys[sum(map(ord, str(name))) % len(keys)]]


def _dynamic_room(spec):
    """Validated world_spec ko renderer ke room contract mein badlo."""
    size = spec["size"]
    w, d, h = size["w"], size["d"], size["h"]
    lights = list(spec.get("lights") or [])
    if not lights:
        lights = [{
            "kind": "sun" if spec["env"] == "outdoor" else "panel",
            "pos": [0.0, h + 1.0, d * .45],
            "color": (255, 245, 225) if spec["time"] in ("day", "sunset") else (180, 195, 255),
            "intensity": 1.0
        }]
    slots = {}
    xmap = {
        "left": -w * .25, "center": 0.0, "right": w * .25,
        "far": 0.0, "seat": -w * .12
    }
    zmap = {"left": d * .42, "center": d * .32, "right": d * .42,
            "far": d * .70, "seat": d * .32}
    for name in ("left", "center", "right", "far", "seat"):
        slots[name] = {"pos": (xmap[name], zmap[name])}
    return {
        "dims": (-w / 2, w / 2, d, h),
        "slots": slots,
        "lights": lights,
        "key": 0 if lights else None,
        "door": None,
        "objs": [],
        "fx": [],
        "tone": (1.0, 1.0, 1.0),
        "amb0": .74,
        "dynamic": True
    }


def _location(scene):
    """Scene ke fixed room ya dynamic spec ko normalize karo."""
    raw = scene.get("world_spec")
    loc = scene.get("location", scene.get("room", "office"))
    if isinstance(loc, dict) and raw is None:
        raw = loc
    if raw is not None:
        clean = validate_world_spec(raw)
        if clean is not None and USE_DYNAMIC_WORLDS:
            return None, clean
        if clean is not None:                                   # dynamic world -> sabse qareeb rich fixed room
            return ROOMS.get(nearest_fixed_room(clean), ROOMS["office"]), None
        fallback = nearest_fixed_room(raw)
        print("[world] invalid world_spec -> fixed room fallback:", fallback)
        return ROOMS.get(fallback, ROOMS["office"]), None
    if isinstance(loc, str) and loc in ROOMS:
        return ROOMS[loc], None
    if isinstance(scene.get("room"), str) and scene["room"] in ROOMS:
        return ROOMS[scene["room"]], None
    return ROOMS["office"], None


def _draw_fixed_shell(r, room):
    room["shell_draw"](r)
    _extra_decor(r, room)
    for obj in room["objs"]:
        for bounds in obj.get("boxes", []):
            r.shadow_box(*bounds)
    r.ao_room()


def _character_items(scene, room, t, talk_lookup, t_offset):
    items = []
    chars = scene.get("characters", []) or []
    slots = room["slots"]
    for i, c in enumerate(chars):
        slot = c.get("slot", "center")
        sit = slot == "seat" and "seat" in slots
        anchor = slots.get(slot, slots["center"])["pos"]
        walk = False
        phase = 0.0
        if c.get("walk_to") in slots:
            dest = slots[c["walk_to"]]["pos"]
            q = ease(clamp(t / max(.15, float(scene.get("duration", 3.0)))))
            anchor = (lerp(anchor[0], dest[0], q), lerp(anchor[1], dest[1], q))
            walk = .02 < q < .98
            sit = False
            phase = t * 8.0
        name = str(c.get("name", "arjun")).lower()
        talk = float(talk_lookup(name, t_offset + t)) if talk_lookup else (
            abs(math.sin(t * 12.0)) * .6 if c.get("line") else 0.0
        )
        others = [
            slots.get(other.get("slot", "center"), slots["center"])["pos"][0]
            for j, other in enumerate(chars) if j != i
        ]
        look_x = (-1 if others and others[0] < anchor[0] else 1) * .7 if others else 0.0
        hand_props = c.get("props", []) or []
        state = {
            "t": float(t), "walk": walk, "ph": phase, "sit": sit,
            "expr": c.get("expr", "smile"), "look": c.get("look", (look_x, 0.0)),
            "talk": clamp(talk), "arms": c.get("arms", {}),
            "props": hand_props, "phone": "phone" in hand_props,
            "key": "key" in hand_props, "gesture": c.get("gesture", "idle"),
            "facing": c.get("facing", "front"), "speed": c.get("speed", 1.0),
            "seat": slot == "seat"
        }
        pos = np.array([float(anchor[0]), 0.0, float(anchor[1])], dtype=float)
        items.append((pos, _char(name), state))
    return items


def render_scene(frame_idx, fps, scene, talk_lookup=None, t_offset=0.0):
    """Global frame index -> BGR frame. talk_lookup(name, scene-local/global offset time)."""
    scene, local_fi, offset = _pick(scene, frame_idx, fps)
    t_offset += offset
    fps = max(1, int(fps))
    duration = max(.1, float(scene.get("duration", 3.0)))
    t = local_fi / float(fps)
    u = clamp(t / duration)

    fixed_room, spec = _location(scene)
    if spec is not None:
        room = _dynamic_room(spec)
    else:
        room = fixed_room

    items = _character_items(scene, room, t, talk_lookup, t_offset)
    if items:
        txz = (
            float(np.mean([p[0] for p, _, _ in items])),
            float(np.mean([p[2] for p, _, _ in items]))
        )
    else:
        txz = (0.0, room["dims"][2] * .4)

    move = scene.get("camera_move", "push_in")
    if move == "through_door" and (spec is not None or not room.get("door")):
        move = "walk_through"
    camera_fn = CAMERA_MOVES.get(move, CAMERA_MOVES["push_in"])
    cam = camera_fn(u, room, txz)
    cam = _dutch_cam(cam, scene, u)

    frame = np.empty((H, W, 3), dtype=np.uint8)
    r = Renderer(frame, cam, frame_idx, room)

    if spec is not None:
        # Dynamic background, props aur unke cast shadows.
        if not draw_world(
            r, spec, t,
            {k: v["pos"] for k, v in room["slots"].items()},
            (float(cam.pos[0]), float(cam.pos[2]))
        ):
            fallback_name = nearest_fixed_room(spec)
            room = ROOMS.get(fallback_name, ROOMS["office"])
            r.room = room
            r.lights = room["lights"]
            _draw_fixed_shell(r, room)
    else:
        _draw_fixed_shell(r, room)

    for pos, _, state in items:
        r.shadow_char(pos, sit=state["sit"])
    r.apply_shadows()

    base_amb = r.amb.copy()
    draw_order = []
    if spec is None:
        draw_order.extend(
            (cam.cam(obj["c"])[0, 2], 0, obj) for obj in room["objs"]
        )
    draw_order.extend(
        (cam.cam(pos)[0, 2], 1, (pos, ch, state))
        for pos, ch, state in items
    )

    for _, kind, obj in sorted(draw_order, key=lambda entry: -entry[0]):
        if kind == 0:
            draw_obj(r, obj)
        else:
            pos, ch, state = obj
            r.amb = base_amb * r.lk(pos + np.array([0.0, 1.1, 0.0]))
            draw_char(r, pos, ch, state)
            r.amb = base_amb

    if spec is None:
        r.run_fx(room.get("fx", []))

    r.finish(room.get("tone", (1.0, 1.0, 1.0)))
    _grade(frame)
    return frame
