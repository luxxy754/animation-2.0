
"""Dynamic indoor/outdoor world builder."""

import math
import numpy as np
import cv2

from .core import W, H, clamp
from .props import CATALOG, draw_prop, draw_custom_prop


FIXED_ROOMS = ("office", "corridor", "break_room", "night_exterior")

ENUMS = {
    "env": ("indoor", "outdoor"),
    "time": ("day", "night", "sunset", "overcast"),
    "weather": ("clear", "rain", "snow", "fog"),
    "ground": ("carpet", "tile", "wood", "grass", "road", "sand", "snow", "water", "concrete"),
    "walls": ("plain", "brick", "tile", "wood"),
    "light": ("sun", "panel", "lamp", "street_lamp", "fire")
}

DEFAULT_SPEC = {
    "env": "indoor",
    "size": {"w": 12, "d": 14, "h": 4},
    "time": "day",
    "weather": "clear",
    "ground": {"type": "tile", "color": [185, 185, 180]},
    "walls": {"pattern": "plain", "color": [215, 220, 225]},
    "sky": {"gradient": [[95, 155, 220], [220, 230, 245]]},
    "lights": [{"kind": "panel", "pos": [0, 3.5, 5]}],
    "props": [],
    "custom_props": [],
    "slots": ["left", "center", "right", "far", "seat"]
}

GROUND_COLORS = {
    "carpet": (95, 90, 115),
    "tile": (185, 185, 180),
    "wood": (125, 90, 60),
    "grass": (75, 145, 70),
    "road": (65, 70, 75),
    "sand": (185, 165, 125),
    "snow": (235, 240, 245),
    "water": (145, 105, 65),
    "concrete": (145, 145, 145)
}


def _rgb(value, fallback):
    """Gemini ke RGB rang ko BGR renderer rang mein badlo."""
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return tuple(fallback)
    try:
        vals = [int(v) for v in value]
    except (ValueError, TypeError, OverflowError):
        return tuple(fallback)
    if any(v < 0 or v > 255 for v in vals):
        return tuple(fallback)
    return (vals[2], vals[1], vals[0])


def _number(value, default, lo, hi):
    try:
        x = float(value)
        if not math.isfinite(x):
            return default
        return max(lo, min(hi, x))
    except (TypeError, ValueError, OverflowError):
        return default


def validate_world_spec(spec):
    """World JSON validate karo; ghalat data par None return karo."""
    if not isinstance(spec, dict):
        return None

    env = spec.get("env")
    if env not in ENUMS["env"]:
        return None

    size = spec.get("size")
    if not isinstance(size, dict):
        return None

    w = _number(size.get("w"), 12, 4, 40)
    d = _number(size.get("d"), 14, 4, 40)
    h = _number(size.get("h"), 4, 2.5, 12)

    time = spec.get("time", "day")
    weather = spec.get("weather", "clear")
    if time not in ENUMS["time"] or weather not in ("clear","rain","snow","fog"):
        return None

    ground = spec.get("ground", {})
    walls = spec.get("walls", {})
    sky = spec.get("sky", {})

    if not isinstance(ground, dict) or not isinstance(walls, dict):
        return None
    if not isinstance(sky, dict):
        return None

    gt = ground.get("type", "concrete")
    wp = walls.get("pattern", "plain")
    if gt not in ENUMS["ground"] or wp not in ENUMS["walls"]:
        return None

    props = spec.get("props", [])
    custom = spec.get("custom_props", [])
    lights = spec.get("lights", [])
    slots = spec.get("slots", ["left","center","right","far","seat"])

    if not isinstance(props, list) or len(props) > 45:
        return None
    if not isinstance(custom, list) or len(custom) > 12:
        return None
    if not isinstance(lights, list) or len(lights) > 12:
        return None
    if not isinstance(slots, list):
        return None

    clean_props = []
    for item in props:
        if not isinstance(item, dict):
            return None
        name = item.get("prop")
        pos = item.get("pos")
        if name not in CATALOG or not isinstance(pos, (list, tuple)) or len(pos) != 2:
            return None
        try:
            x, z = float(pos[0]), float(pos[1])
            rot = float(item.get("rot", 0))
            scale = float(item.get("scale", 1))
        except (ValueError, TypeError, OverflowError):
            return None
        if not all(math.isfinite(v) for v in (x,z,rot,scale)):
            return None
        if not .1 <= scale <= 4:
            return None
        if abs(x) > w/2 or z < 0 or z > d:
            return None

        clean_props.append({
            "prop": name,
            "pos": [x,z],
            "rot": rot,
            "scale": scale,
            "color": item.get("color")
        })

    clean_lights = []
    for light in lights:
        if not isinstance(light, dict):
            return None
        kind = light.get("kind", "panel")
        pos = light.get("pos")
        if kind not in ENUMS["light"] or not isinstance(pos, (list,tuple)) or len(pos)!=3:
            return None
        try:
            p = [float(v) for v in pos]
        except (TypeError,ValueError,OverflowError):
            return None
        if not all(math.isfinite(v) and abs(v)<=60 for v in p):
            return None
        clean_lights.append({
            "kind": kind,
            "pos": p,
            "color": _rgb(light.get("color"),(255,255,255))[::-1],
            "intensity": _number(light.get("intensity"),1,.05,3)
        })

    clean_custom = []
    for item in custom:
        if not isinstance(item, dict) or not isinstance(item.get("name"),str):
            return None
        parts = item.get("parts")
        if not isinstance(parts,list) or not 1 <= len(parts) <= 25:
            return None
        clean_custom.append(item)

    valid_slots = []
    for slot in slots:
        if slot in ("left","center","right","far","seat"):
            if slot not in valid_slots:
                valid_slots.append(slot)

    if not valid_slots:
        valid_slots = ["left","center","right","far","seat"]

    return {
        "env": env,
        "size": {"w":w,"d":d,"h":h},
        "time": time,
        "weather": weather,
        "ground": {
            "type":gt,
            "color":_rgb(ground.get("color"),GROUND_COLORS[gt][::-1])
        },
        "walls": {
            "pattern":wp,
            "color":_rgb(walls.get("color"),(220,220,220))
        },
        "sky": sky,
        "lights": clean_lights,
        "props": clean_props,
        "custom_props": clean_custom,
        "slots": valid_slots
    }


def nearest_fixed_room(spec):
    """Invalid world par fixed room ka safe fallback do."""
    if not isinstance(spec,dict):
        return "office"
    env = spec.get("env")
    time = spec.get("time")
    if env == "outdoor" and time == "night":
        return "night_exterior"
    if env == "indoor":
        return "break_room" if spec.get("ground",{}).get("type") == "wood" else "office"
    return "corridor"


def _ground(r, w, d, col):
    """Farsh ka stable polygon."""
    r.poly(np.array([
        [-w/2,0,0],
        [w/2,0,0],
        [w/2,0,d],
        [-w/2,0,d]
    ],dtype=np.float64),col,2,tint=True)


def _indoor(r, spec):
    """Indoor room ka floor, deewarein aur chhat."""
    size = spec["size"]
    w,d,h = size["w"],size["d"],size["h"]
    ground = spec["ground"]["color"]
    wall = spec["walls"]["color"]

    _ground(r,w,d,ground)

    r.box(-w/2,w/2,0,h, d-.12,d,wall,lw=2)
    r.box(-w/2,-w/2+.12,0,h,0,d,wall,lw=2)
    r.box(w/2-.12,w/2,0,h,0,d,wall,lw=2)

    ceiling = tuple(int(v*.88) for v in wall)
    r.poly(np.array([
        [-w/2,h,0],[w/2,h,0],[w/2,h,d],[-w/2,h,d]
    ],dtype=np.float64),ceiling,1)

    # Baseboard deewaron ke saath lagao.
    for x in (-w/2+.06,w/2-.06):
        r.box(x-.06,x+.06,0,.16,0,d,(125,125,130),lw=1)
    r.box(-w/2,w/2,0,.16,d-.16,d,(125,125,130),lw=1)

    # Wall pattern ko fixed geometry se banao.
    if spec["walls"]["pattern"] == "brick":
        for y in np.arange(.35,h-.15,.42):
            points_a = np.array([[-w/2,y,.015],[w/2,y,.015]])
            r.segs(points_a[:1],points_a[1:],(150,145,140),1)
        for y in np.arange(.2,h-.2,.42):
            offset = .5 if int(y*10)%2 else 0
            xs = np.arange(-w/2+offset,w/2,.95)
            for x in xs:
                r.segs(np.array([[x,y,.015]]),np.array([[x,y+.42,.015]]),(155,150,145),1)

    elif spec["walls"]["pattern"] == "tile":
        for y in np.arange(.3,h,.55):
            r.segs(np.array([[-w/2,y,.02]]),np.array([[w/2,y,.02]]),(165,165,165),1)
        for x in np.arange(-w/2,w/2,.65):
            r.segs(np.array([[x,0,.02]]),np.array([[x,h,.02]]),(165,165,165),1)

    elif spec["walls"]["pattern"] == "wood":
        for x in np.arange(-w/2+.2,w/2,.45):
            r.segs(np.array([[x,0,.02]]),np.array([[x,h,.02]]),(145,115,85),1)


def _outdoor(r, spec):
    """Outdoor aasman, ground aur distant skyline."""
    size = spec["size"]
    w,d = size["w"],size["d"]
    time = spec["time"]

    if time == "night":
        sky_top, sky_bottom = (32,22,18),(95,65,40)
    elif time == "sunset":
        sky_top, sky_bottom = (105,95,185),(65,135,235)
    elif time == "overcast":
        sky_top, sky_bottom = (125,130,135),(185,185,185)
    else:
        sky_top, sky_bottom = (210,145,85),(245,205,145)

    stops = [
        (-1.2,sky_top),
        (0.0,sky_bottom),
        (1.2,sky_bottom)
    ]
    r.sky(stops)

    _ground(r,w,d,spec["ground"]["color"])

    # Door ke baghair distant buildings ki silhouette.
    if spec["ground"]["type"] != "water":
        for i in range(9):
            x = -w*.48 + i*w*.12
            bh = 1.0 + ((i*7)%5)*.38
            bw = w*.085
            col = (75,80,95) if time=="night" else (135,135,145)
            r.box(x-bw/2,x+bw/2,0,bh,d*.90,d*.94,col,lw=1)

            # Building windows.
            if time == "night":
                for yy in np.arange(.3,bh-.1,.35):
                    r.box(x-bw*.25,x+bw*.25,yy,yy+.10,d*.895,d*.89,(190,165,95),lw=0)


def _weather(r, spec, t):
    """Rain, snow aur fog ko deterministic frame effects se banao."""
    kind = spec["weather"]
    if kind == "clear":
        return

    frame = r.fr
    if kind == "fog":
        overlay = np.full_like(frame,(205,205,205))
        cv2.addWeighted(frame,.88,overlay,.12,0,dst=frame)
        return

    fi = int(getattr(r,"fi",0))
    if kind == "rain":
        # Har frame ka pattern deterministic hai; koi random jitter nahi.
        for i in range(65):
            x = (i*137 + fi*7) % W
            y = (i*83 + fi*17) % H
            cv2.line(frame,(int(x),int(y)),(int(x-5),int(y+15)),
                     (190,175,145),1,cv2.LINE_AA)

    elif kind == "snow":
        for i in range(45):
            x = (i*151 + fi*2) % W
            y = (i*97 + fi*5) % H
            cv2.circle(frame,(int(x),int(y)),2,(245,245,245),-1,cv2.LINE_AA)


def _collides(box, other):
    """Do props ke horizontal bounds overlap check karo."""
    return not (
        box[1] <= other[0] or box[0] >= other[1] or
        box[5] <= other[4] or box[4] >= other[5]
    )


def auto_place_props(spec, character_slots=None, camera_xz=None):
    """Overlapping props ko safe jagah shift karo."""
    slots = character_slots or {}
    occupied = []
    result = []

    w = spec["size"]["w"]
    d = spec["size"]["d"]

    for item in spec["props"]:
        name = item["prop"]
        from .props import CATALOG
        _, dims, _ = CATALOG[name]
        sx, sy, sz = [float(v)*item["scale"] for v in dims]
        x,z = item["pos"]

        candidates = [(x,z)]
        for step in range(1,7):
            candidates.extend([
                (x+step*.65,z),
                (x-step*.65,z),
                (x,z+step*.65),
                (x,z-step*.65)
            ])

        chosen = None
        for cx,cz in candidates:
            if cx-sx/2 < -w/2 or cx+sx/2 > w/2 or cz-sz/2 < 0 or cz+sz/2 > d:
                continue

            box = (cx-sx/2,cx+sx/2,0,sy,cz-sz/2,cz+sz/2)
            if any(_collides(box,o) for o in occupied):
                continue

            blocked = False
            for p in slots.values():
                if math.hypot(cx-p[0],cz-p[1]) < max(sx,sz)*.55+0.45:
                    blocked = True
                    break

            if blocked:
                continue

            if camera_xz is not None:
                if math.hypot(cx-camera_xz[0],cz-camera_xz[1]) < .65:
                    continue

            chosen = (cx,cz)
            occupied.append(box)
            break

        if chosen is not None:
            new_item = dict(item)
            new_item["pos"] = list(chosen)
            result.append(new_item)

    return result


def draw_world(r, spec, t=0.0, character_slots=None, camera_xz=None):
    """Validated world ka background aur props render karo."""
    clean = validate_world_spec(spec)
    if clean is None:
        return False

    if clean["env"] == "indoor":
        _indoor(r,clean)
    else:
        _outdoor(r,clean)

    props = auto_place_props(clean,character_slots,camera_xz)

    for item in props:
        result = draw_prop(
            r,
            item["prop"],
            item["pos"],
            item["rot"],
            item["scale"],
            item["color"]
        )
        if result and hasattr(r,"shadow_box"):
            r.shadow_box(*result["bounds"])

    for custom in clean["custom_props"]:
        draw_custom_prop(r,custom)

    _weather(r,clean,t)
    return True
