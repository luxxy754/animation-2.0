"""toon/scenery.py - detailed painted backgrounds (parallax layers), ground detail, set pieces, ambient NPC extras aur animals.
Sab kuch deterministic hai (seed = theme naam), koi random flicker nahi. Colours BGR mein hain. Har theme = ek chhota recipe."""
import math
import numpy as np
import cv2

from .core import W, H, clamp

OUT = (22, 18, 26)


def _rng(seed):
    return np.random.RandomState(abs(hash(seed)) % (2 ** 31) if isinstance(seed, str) else seed)


def _stable_seed(s):
    return sum((i + 1) * ord(c) for i, c in enumerate(str(s))) % 100003


def mix(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def dark(c, k):
    return tuple(int(clamp(v * k, 0, 255)) for v in c)


# --------------------------------------------------------------------------------------------- THEMES
# sky=(top,bottom) BGR, hills=[(color, height, freq, z_frac)], ground_detail=kind, skyline=kind, sun=(x_off, y, color) ya None
THEMES = {
    "village_market": dict(sky=((235, 175, 95), (250, 225, 175)), hills=[((190, 160, 120), 2.6, .5, .97), ((120, 175, 120), 1.6, .8, .93)],
                           detail="dirt", skyline="village", sun=(3.5, 6.5, (200, 245, 255)), clouds=5, birds=True, npc="market"),
    "island": dict(sky=((240, 190, 100), (255, 235, 190)), hills=[], detail="sand", skyline="sea", sun=(-3, 6.0, (200, 245, 255)), clouds=4, birds=True, npc="beach"),
    "farm": dict(sky=((235, 180, 100), (250, 230, 180)), hills=[((170, 190, 140), 2.8, .35, .97), ((90, 165, 95), 1.5, .7, .93)],
                 detail="grass", skyline="farm", sun=(4, 6.5, (200, 245, 255)), clouds=6, birds=True, npc="farm"),
    "city_street": dict(sky=((230, 170, 100), (250, 225, 190)), hills=[], detail="road", skyline="city", sun=None, clouds=3, birds=True, npc="city"),
    "park": dict(sky=((235, 175, 95), (250, 228, 180)), hills=[((140, 190, 130), 2.2, .45, .97), ((95, 175, 95), 1.3, .9, .93)],
                 detail="grass", skyline="trees", sun=(-4, 6.5, (200, 245, 255)), clouds=6, birds=True, npc="park"),
    "school_yard": dict(sky=((235, 180, 100), (250, 230, 185)), hills=[((150, 185, 140), 1.8, .5, .97)], detail="concrete", skyline="school",
                        sun=(4, 6.5, (200, 245, 255)), clouds=5, birds=True, npc="school"),
    "campfire_night": dict(sky=((60, 30, 18), (120, 70, 45)), hills=[((60, 45, 40), 3.0, .4, .97), ((45, 60, 40), 1.8, .8, .93)], detail="grass_night",
                           skyline="trees_night", sun=(3, 7.5, (225, 235, 245)), clouds=0, birds=False, npc="camp", stars=True),
    "mountain_town": dict(sky=((190, 150, 120), (235, 220, 205)), hills=[((200, 195, 190), 4.5, .3, .98), ((170, 160, 150), 3.2, .5, .95)],
                          detail="snow", skyline="chalets", sun=None, clouds=4, birds=False, npc="town"),
    "harbor": dict(sky=((110, 90, 185), (90, 160, 250)), hills=[], detail="concrete", skyline="harbor", sun=(5, 2.2, (180, 235, 255)), clouds=4, birds=True, npc="port"),
}
DEFAULT_THEME = "park"


def theme_for(spec):
    """Spec se theme naam: preset ka 'theme' ya ground/time se andaza."""
    t = spec.get("theme")
    if t in THEMES:
        return t
    g = (spec.get("ground") or {}).get("type", "grass")
    return {"sand": "island", "road": "city_street", "snow": "mountain_town", "water": "harbor", "concrete": "school_yard"}.get(g, DEFAULT_THEME)


# --------------------------------------------------------------------------------------------- drawing helpers
def _P(r, pts3, fill, lw=2, tint=True):
    r.poly(np.array(pts3, float), fill, lw, tint=tint)


def _quad_v(r, x0, x1, y0, y1, z, fill, lw=2):
    """Khadi (vertical) billboard-rectangle z par."""
    _P(r, [[x0, y0, z], [x1, y0, z], [x1, y1, z], [x0, y1, z]], fill, lw)


def _circle3(r, cx, cy, z, rad, fill, lw=0, n=24):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.stack([cx + rad * np.cos(a), cy + rad * np.sin(a), np.full(n, z)], 1)
    r.poly(pts, fill, lw, tint=False)


def _blob3(r, cx, cy, z, rx, ry, fill, lw=0, n=22):
    a = np.linspace(0, 2 * math.pi, n, endpoint=False)
    pts = np.stack([cx + rx * np.cos(a), cy + ry * np.sin(a), np.full(n, z)], 1)
    r.poly(pts, fill, lw, tint=False)


def _grad_sky(r, top, bot, time):
    r.sky([(-1.2, bot), (0.0, bot), (0.30, mix(bot, top, .55)), (0.75, top), (1.4, top)])


# --------------------------------------------------------------------------------------------- sky layer
def _sb(r, cx, cy, z, rx, ry, fill, lw=0, n=0):
    """Screen-space ellipse: world point door (z par) project karke gol hi rakhta hai (camera pitch/roll se sooraj kheenchta nahi)."""
    pc = r.c.cam(np.array([cx, cy, z], float))
    if pc[0, 2] < .1:
        return
    u, v = r.c.proj(pc)[0]
    sc = r.c.f / pc[0, 2]
    if not (-3000 < u < 4000 and -3000 < v < 5000):
        return
    cv2.ellipse(r.fr, (int(u), int(v)), (max(1, int(rx * sc)), max(1, int(ry * sc))), 0, 0, 360, tuple(int(c) for c in fill), -1, cv2.LINE_AA)


def _sc(r, cx, cy, z, rad, fill, lw=0, n=0):
    _sb(r, cx, cy, z, rad, rad, fill)


def draw_sky(r, spec, theme, t):
    th = THEMES[theme]
    top, bot = th["sky"]
    if spec.get("time") == "night" and theme != "campfire_night":
        top, bot = (60, 30, 18), (120, 70, 45)
    if spec.get("time") == "overcast":
        top, bot = (150, 150, 155), (205, 205, 205)
    _grad_sky(r, top, bot, spec.get("time"))
    w, d = spec["size"]["w"], spec["size"]["d"]
    zf = d * 1.15
    rng = np.random.RandomState(_stable_seed(theme))
    # taare
    if th.get("stars"):
        for _ in range(60):
            x, y = rng.uniform(-w * 1.2, w * 1.2), rng.uniform(5, 14)
            tw = .6 + .4 * math.sin(t * 2 + x * 3)
            _sc(r, x, y, zf, .05 + .04 * rng.rand(), (int(235 * tw), int(240 * tw), int(250 * tw)))
    # sooraj / chand + glow
    if th["sun"]:
        sx, sy, sc = th["sun"]
        for k, a in ((1.9, .10), (1.4, .18), (1.1, .30)):
            col = mix(bot, sc, a * 2.2)
            _sc(r, sx * 2.2, sy * 2.0 + 2, zf, 1.4 * k, col)
        _sc(r, sx * 2.2, sy * 2.0 + 2, zf, .8, sc if theme != "campfire_night" else (235, 240, 250), 0)
    # badal (halke drift)
    for i in range(th["clouds"] + (4 if th["clouds"] else 0)):
        cx = ((rng.uniform(-w * 1.5, w * 1.5) + t * (.12 + .05 * (i % 3)) + w * 2) % (w * 3)) - w * 1.5
        cy = rng.uniform(5.5, 16.0)
        s = rng.uniform(.9, 1.7)
        for dx, dy, rx, ry in ((0, 0, 1.6, .55), (-1.1, -.08, 1.0, .42), (1.1, -.1, 1.1, .42), (.1, .35, 1.1, .5)):
            _sb(r, cx + dx * s, cy + dy * s, zf, rx * s, ry * s, (252, 250, 248))
        _sb(r, cx, cy - .32 * s, zf, 1.5 * s, .22 * s, (232, 228, 224))
    # door ke parinde
    if th["birds"]:
        for i in range(5):
            bx = ((i * 2.3 + t * (.9 + .1 * i) + w) % (w * 2.4)) - w * 1.2
            by = 6.0 + i * .5 + .25 * math.sin(t * 3 + i)
            flap = .18 * math.sin(t * 9 + i * 1.7)
            r.segs(np.array([[bx - .35, by + flap, zf * .8], [bx, by, zf * .8]]), np.array([[bx, by, zf * .8], [bx + .35, by + flap, zf * .8]]), (40, 35, 45), 2)


# --------------------------------------------------------------------------------------------- far hills / skyline
def _ridge(r, w, color, height, freq, z, phase, base=0.0, lw=2):
    xs = np.linspace(-w * 1.4, w * 1.4, 36)
    ys = base + height * (.55 + .30 * np.sin(xs * freq + phase) + .15 * np.sin(xs * freq * 2.3 + phase * 1.7))
    pts = [[xs[0], 0, z]] + [[x, y, z] for x, y in zip(xs, ys)] + [[xs[-1], 0, z]]
    _P(r, pts, color, lw)


def draw_backdrop(r, spec, theme, t):
    th = THEMES[theme]
    w, d = spec["size"]["w"], spec["size"]["d"]
    night = spec.get("time") == "night" or theme == "campfire_night"
    for i, (col, hgt, fr, zf) in enumerate(th["hills"]):
        _ridge(r, w, col, hgt, fr, d * zf + i * .4, i * 1.9 + _stable_seed(theme) % 7, lw=2)
    sk = th["skyline"]
    z0 = d * .93
    rng = np.random.RandomState(_stable_seed(theme + "sk"))
    xs = np.linspace(-w * .75, w * .75, 9)
    if sk == "village":
        for i, x in enumerate(xs):
            hw, hh = rng.uniform(1.0, 1.7), rng.uniform(1.2, 2.0)
            wall = [(205, 225, 245), (170, 205, 235), (190, 220, 200), (205, 195, 235)][i % 4]
            roof = [(70, 80, 190), (60, 110, 200), (90, 70, 160)][i % 3]
            _quad_v(r, x - hw / 2, x + hw / 2, 0, hh, z0, wall, 2)
            _P(r, [[x - hw / 2 - .15, hh, z0], [x + hw / 2 + .15, hh, z0], [x, hh + hw * .45, z0]], roof, 2)
            _quad_v(r, x - .18, x + .18, .1, .75, z0 - .02, (60, 90, 130), 2)
            _quad_v(r, x + hw * .25, x + hw * .42, hh * .55, hh * .85, z0 - .02, (230, 200, 130) if night else (255, 225, 140), 1)
            _quad_v(r, x - hw * .42, x - hw * .25, hh * .55, hh * .85, z0 - .02, (230, 200, 130) if night else (255, 225, 140), 1)
    elif sk == "city":
        for i, x in enumerate(np.linspace(-w * .8, w * .8, 8)):
            bw, bh = rng.uniform(1.4, 2.2), rng.uniform(3.5, 7.5)
            col = [(185, 170, 160), (160, 150, 150), (200, 190, 175), (150, 160, 175)][i % 4]
            _quad_v(r, x - bw / 2, x + bw / 2, 0, bh, z0, col, 2)
            for yy in np.arange(.7, bh - .4, .8):
                for xx in (-bw * .28, 0, bw * .28):
                    _quad_v(r, x + xx - .17, x + xx + .17, yy, yy + .42, z0 - .02, (250, 235, 150) if (i + int(yy * 3)) % 3 else (180, 200, 215), 0)
    elif sk == "farm":
        # laal gola ghar (barn) + silo + windmill
        bx = -w * .28
        _quad_v(r, bx - 1.5, bx + 1.5, 0, 1.9, z0, (70, 70, 195), 2)
        _P(r, [[bx - 1.7, 1.9, z0], [bx + 1.7, 1.9, z0], [bx + 1.1, 2.9, z0], [bx - 1.1, 2.9, z0]], (60, 60, 170), 2)
        _quad_v(r, bx - .5, bx + .5, 0, 1.2, z0 - .02, (235, 240, 245), 2)
        r.segs(np.array([[bx - .5, 0, z0 - .03], [bx + .5, 0, z0 - .03]]), np.array([[bx + .5, 1.2, z0 - .03], [bx - .5, 1.2, z0 - .03]]), (235, 240, 245), 2)
        sx = bx + 2.7
        _quad_v(r, sx - .5, sx + .5, 0, 3.0, z0, (200, 205, 210), 2)
        _blob3(r, sx, 3.0, z0, .5, .35, (185, 190, 195), 2)
        mx = w * .32
        _P(r, [[mx - .35, 0, z0], [mx + .35, 0, z0], [mx + .12, 3.3, z0], [mx - .12, 3.3, z0]], (210, 215, 220), 2)
        ang = t * .8
        for k in range(4):
            a = ang + k * math.pi / 2
            ex, ey = mx + 1.5 * math.cos(a), 3.3 + 1.5 * math.sin(a)
            nx, ny = -math.sin(a) * .22, math.cos(a) * .22
            _P(r, [[mx, 3.3, z0 - .02], [ex + nx, ey + ny, z0 - .02], [ex - nx, ey - ny, z0 - .02]], (240, 245, 250), 2)
    elif sk == "school":
        _quad_v(r, -w * .45, w * .45, 0, 3.2, z0, (130, 160, 225), 2)
        _P(r, [[-w * .47, 3.2, z0], [w * .47, 3.2, z0], [w * .40, 3.8, z0], [-w * .40, 3.8, z0]], (90, 100, 175), 2)
        for x in np.linspace(-w * .38, w * .38, 6):
            _quad_v(r, x - .35, x + .35, 1.6, 2.7, z0 - .02, (245, 230, 170), 2)
        _quad_v(r, -.5, .5, 0, 1.5, z0 - .02, (90, 70, 60), 2)
        _P(r, [[-.7, 3.8, z0], [.7, 3.8, z0], [0, 5.0, z0]], (60, 70, 200), 2)
    elif sk == "chalets":
        for i, x in enumerate(np.linspace(-w * .7, w * .7, 6)):
            hw, hh = rng.uniform(1.4, 2.0), rng.uniform(1.1, 1.6)
            _quad_v(r, x - hw / 2, x + hw / 2, 0, hh, z0, (110, 150, 205), 2)
            _P(r, [[x - hw / 2 - .2, hh, z0], [x + hw / 2 + .2, hh, z0], [x, hh + 1.0, z0]], (245, 245, 250), 2)
            _quad_v(r, x - .2, x + .2, .35, 1.0, z0 - .02, (250, 225, 150), 1)
        for x in np.linspace(-w * .7, w * .7, 7):
            _P(r, [[x - .5, 0, z0 - .8], [x + .5, 0, z0 - .8], [x, 2.4 + (x * 7) % 1, z0 - .8]], (95, 130, 70), 2)
    elif sk in ("trees", "trees_night"):
        for i, x in enumerate(np.linspace(-w * .85, w * .85, 12)):
            h = rng.uniform(2.4, 3.6)
            lg = (60, 120, 50) if sk == "trees_night" else (80, 175, 70)
            _quad_v(r, x - .12, x + .12, 0, h * .45, z0 - .3, (55, 80, 120), 2)
            _blob3(r, x, h * .65, z0 - .3, .9 + rng.rand() * .3, h * .42, dark(lg, .8 + .3 * (i % 2)), 2)
            _blob3(r, x - .25, h * .7, z0 - .32, .45, h * .22, dark(lg, 1.18), 0)
    elif sk == "sea":
        # samundar: horizon band + lehren + door ka jazeera
        _P(r, [[-w * 1.5, 0, d * .98], [w * 1.5, 0, d * .98], [w * 1.5, .9, d * .98], [-w * 1.5, .9, d * .98]], (200, 160, 70), 0)
        for k in range(7):
            y = .08 + k * .12
            col = mix((215, 175, 85), (190, 130, 40), k / 6)
            _P(r, [[-w * 1.5, y, d * .975 - k * .05], [w * 1.5, y, d * .975 - k * .05], [w * 1.5, y + .13, d * .975 - k * .05], [-w * 1.5, y + .13, d * .975 - k * .05]], col, 0)
        for i in range(10):
            x = ((i * 2.9 + t * .5) % (w * 2)) - w
            y = .2 + (i % 4) * .17
            r.segs(np.array([[x - .35, y, d * .96]]), np.array([[x + .35, y, d * .96]]), (255, 250, 245), 2)
        _P(r, [[-w * .5, 0, d * .97], [-w * .1, 0, d * .97], [-w * .28, 1.0, d * .97]], (110, 165, 110), 2)
        # sail boat
        bx = w * .25 + .25 * math.sin(t * .8)
        _P(r, [[bx - .5, .55, d * .96], [bx + .5, .55, d * .96], [bx + .3, .38, d * .96], [bx - .3, .38, d * .96]], (60, 80, 160), 2)
        _P(r, [[bx, .6, d * .96], [bx, 1.6, d * .96], [bx + .5, .65, d * .96]], (250, 250, 250), 2)
    elif sk == "harbor":
        _P(r, [[-w * 1.5, 0, d * .98], [w * 1.5, 0, d * .98], [w * 1.5, 1.0, d * .98], [-w * 1.5, 1.0, d * .98]], (150, 95, 60), 0)
        for k in range(8):
            y = .05 + k * .11
            _P(r, [[-w * 1.5, y, d * .975], [w * 1.5, y, d * .975], [w * 1.5, y + .08, d * .975], [-w * 1.5, y + .08, d * .975]],
               mix((120, 150, 245), (150, 90, 70), k / 7), 0)
        for i, x in enumerate(np.linspace(-w * .7, w * .7, 7)):
            bw, bh = rng.uniform(1.0, 1.6), rng.uniform(1.6, 3.2)
            _quad_v(r, x - bw / 2, x + bw / 2, 0, bh, d * .95, [(120, 110, 150), (110, 120, 160), (130, 105, 140)][i % 3], 2)
            for yy in np.arange(.5, bh - .3, .6):
                _quad_v(r, x - .15, x + .15, yy, yy + .25, d * .95 - .02, (170, 235, 255), 0)
        # kranes
        cx = w * .35
        r.segs(np.array([[cx, 0, d * .94], [cx, 3.8, d * .94]]), np.array([[cx, 3.8, d * .94], [cx - 2.2, 3.8, d * .94]]), (60, 60, 190), 5)


# --------------------------------------------------------------------------------------------- ground detail (grass tufts, flowers, dirt specks...)
def draw_ground_detail(r, spec, theme, t):
    th = THEMES[theme]
    kind = th["detail"]
    w, d = spec["size"]["w"], spec["size"]["d"]
    rng = np.random.RandomState(_stable_seed(theme + "gd"))
    base = spec["ground"]["color"]
    # gehri/halki patti (mowing stripes / kaccha raasta)
    n = 9
    for i in range(n):
        z0, z1 = d * i / n, d * (i + 1) / n
        if i % 2 == 0:
            col = dark(base, .93) if kind not in ("sand",) else dark(base, .96)
            _P(r, [[-w / 2, 0, z0], [w / 2, 0, z0], [w / 2, 0, z1], [-w / 2, 0, z1]], col, 0, tint=True)
    if kind in ("dirt", "road", "concrete"):
        # beech ka raasta + dhariyan
        pc = {"dirt": dark(base, 1.14), "road": dark(base, .85), "concrete": dark(base, 1.07)}[kind]
        _P(r, [[-1.9, 0, 0], [1.9, 0, 0], [1.4, 0, d * .96], [-1.4, 0, d * .96]], pc, 0)
        if kind == "road":
            for z in np.arange(1.0, d * .9, 2.0):
                _P(r, [[-.09, .002, z], [.09, .002, z], [.09, .002, z + .9], [-.09, .002, z + .9]], (235, 240, 245), 0, tint=False)
            for sd in (-1, 1):
                _P(r, [[sd * 3.0, .002, 0], [sd * 3.15, .002, 0], [sd * 3.15, .002, d * .95], [sd * 3.0, .002, d * .95]], (230, 235, 240), 0, tint=False)
        if kind == "concrete":
            for z in np.arange(1.5, d, 2.6):
                r.segs(np.array([[-w / 2, 0.003, z]]), np.array([[w / 2, 0.003, z]]), dark(base, .8), 1)
    if kind in ("grass", "grass_night"):
        # raasta (pagdandi)
        _P(r, [[-.8, 0, 0], [.8, 0, 0], [.35, 0, d * .9], [-.35, 0, d * .9]], mix(base, (140, 170, 200), .55) if kind == "grass" else dark(base, 1.4), 0)
    # tufts / pebbles / ripples
    cnt = {"grass": 260, "grass_night": 200, "dirt": 120, "sand": 120, "snow": 90, "road": 40, "concrete": 30}.get(kind, 100)
    A, B = [], []
    for _ in range(cnt):
        x, z = rng.uniform(-w / 2, w / 2), rng.uniform(.5, d * .95)
        if kind.startswith("grass") and abs(x) < .6 * (1 - z / d) + .3:
            continue
        sway = .05 * math.sin(t * 2.2 + x * 1.7 + z) if kind.startswith("grass") else 0
        h = rng.uniform(.10, .26) if kind.startswith("grass") else rng.uniform(.02, .06)
        for dx in (-.05, 0, .05):
            A.append([x + dx * .6, 0.0, z]); B.append([x + dx + sway, h * (1.0 - abs(dx) * 3), z + .01])
    if A:
        col = {"grass": dark(base, .62), "grass_night": dark(base, .55), "dirt": dark(base, .75), "sand": dark(base, .85), "snow": (230, 215, 205),
               "road": dark(base, 1.15), "concrete": dark(base, .85)}[kind]
        r.segs(np.array(A), np.array(B), col, 1)
    if kind == "grass":
        cols = [(80, 90, 240), (60, 200, 250), (250, 245, 245), (200, 110, 235)]
        for i in range(34):
            x, z = rng.uniform(-w / 2.2, w / 2.2), rng.uniform(1.5, d * .9)
            if abs(x) < .5:
                continue
            c = cols[i % 4]
            r.segs(np.array([[x, 0, z]]), np.array([[x + .03 * math.sin(t * 2 + i), .22, z]]), (60, 130, 60), 2)
            _circle3(r, x + .03 * math.sin(t * 2 + i), .26, z, .07, c, 1, 10)
            _circle3(r, x + .03 * math.sin(t * 2 + i), .26, z, .025, (60, 210, 250), 0, 8)
    if kind == "sand":
        for i in range(14):
            x, z = rng.uniform(-w / 2.3, w / 2.3), rng.uniform(1.0, d * .9)
            _blob3(r, x, .01, z, .12, .05, (190, 215, 245), 0, 8)  # seashell
            # ripple arcs
        for z in np.arange(2.0, d * .9, 1.7):
            r.segs(np.array([[-w * .35, .004, z]]), np.array([[w * .35, .004, z + .15]]), dark(base, .9), 1)
    if kind == "dirt":
        for i in range(12):
            x, z = rng.uniform(-w / 2.3, w / 2.3), rng.uniform(1.0, d * .9)
            _blob3(r, x, .01, z, .13, .06, dark(base, .7), 1, 8)  # patthar
    if kind == "snow":
        for i in range(10):
            x, z = rng.uniform(-w / 2.3, w / 2.3), rng.uniform(1.0, d * .9)
            _blob3(r, x, .02, z, .35, .12, (250, 245, 242), 0, 12)


# --------------------------------------------------------------------------------------------- set pieces (3D-ish cartoon objects, bade aur outline wale)
def _tree(r, x, z, s, night=False, kind="round"):
    trunk = (50, 85, 125)
    r.box(x - .12 * s, x + .12 * s, 0, 1.2 * s, z - .12 * s, z + .12 * s, trunk, lw=2)
    g = (60, 130, 55) if night else (85, 185, 75)
    if kind == "pine":
        for k in range(4):
            wd = (1.0 - k * .2) * s
            _P(r, [[x - wd, .7 * s + k * .55 * s, z], [x + wd, .7 * s + k * .55 * s, z], [x, 1.5 * s + k * .55 * s, z]], dark(g, .85 + k * .08), 2)
    else:
        _blob3(r, x, 1.9 * s, z, .95 * s, .85 * s, dark(g, .86), 2)
        _blob3(r, x - .45 * s, 1.55 * s, z - .02, .6 * s, .55 * s, dark(g, .95), 2)
        _blob3(r, x + .45 * s, 1.6 * s, z - .02, .6 * s, .55 * s, dark(g, .95), 2)
        _blob3(r, x - .25 * s, 2.15 * s, z - .04, .4 * s, .28 * s, dark(g, 1.2), 0)


def _stall(r, x, z, s, c1=(60, 70, 215), t=0.0):
    """Bazaar ka stall: striped awning + counter + saaman."""
    r.box(x - 1.0 * s, x + 1.0 * s, 0, .8 * s, z - .4 * s, z + .4 * s, (70, 110, 160), lw=2)
    for px in (-1.0, 1.0):
        r.box(x + px * s - .05, x + px * s + .05, 0, 2.1 * s, z - .4 * s, z - .35 * s, (60, 90, 130), lw=1)
    n = 6
    for i in range(n):
        xa, xb = x - 1.15 * s + i * 2.3 * s / n, x - 1.15 * s + (i + 1) * 2.3 * s / n
        _P(r, [[xa, 2.1 * s, z - .55 * s], [xb, 2.1 * s, z - .55 * s], [xb, 1.75 * s, z + .35 * s], [xa, 1.75 * s, z + .35 * s]],
           c1 if i % 2 == 0 else (245, 245, 245), 2)
    for i, col in enumerate([(60, 80, 220), (70, 190, 245), (80, 200, 90), (60, 140, 250), (190, 90, 210)]):
        _blob3(r, x - .75 * s + i * .38 * s, .93 * s, z - .42 * s, .15 * s, .13 * s, col, 1, 10)


def _house(r, x, z, s, wall=(205, 225, 245), roof=(70, 80, 190)):
    r.box(x - 1.2 * s, x + 1.2 * s, 0, 1.6 * s, z - .8 * s, z + .8 * s, wall, lw=2)
    _P(r, [[x - 1.4 * s, 1.6 * s, z - .85 * s], [x + 1.4 * s, 1.6 * s, z - .85 * s], [x + 1.4 * s, 1.6 * s, z + .85 * s], [x - 1.4 * s, 1.6 * s, z + .85 * s]], roof, 2)
    _P(r, [[x - 1.4 * s, 1.6 * s, z - .86 * s], [x + 1.4 * s, 1.6 * s, z - .86 * s], [x + .3 * s, 2.5 * s, z - .86 * s], [x - .3 * s, 2.5 * s, z - .86 * s]], roof, 2)
    _quad_v(r, x - .25 * s, x + .25 * s, 0, 1.0 * s, z - .82 * s, (60, 95, 140), 2)
    _quad_v(r, x + .55 * s, x + .95 * s, .8 * s, 1.3 * s, z - .82 * s, (250, 225, 150), 2)
    _quad_v(r, x - .95 * s, x - .55 * s, .8 * s, 1.3 * s, z - .82 * s, (250, 225, 150), 2)


def _crate(r, x, z, s, col=(95, 150, 205)):
    r.box(x - .35 * s, x + .35 * s, 0, .55 * s, z - .3 * s, z + .3 * s, col, lw=2)
    r.box(x - .28 * s, x + .28 * s, .55 * s, .6 * s, z - .24 * s, z + .24 * s, dark(col, .8), lw=1)


def _haystack(r, x, z, s):
    _blob3(r, x, .55 * s, z, .8 * s, .55 * s, (110, 205, 240), 2)
    r.segs(np.array([[x - .5 * s, .3 * s, z - .01], [x, .2 * s, z - .01]]), np.array([[x - .3 * s, .75 * s, z - .01], [x + .4 * s, .6 * s, z - .01]]), (70, 160, 200), 1)


def _crop_rows(r, x0, x1, z0, z1):
    for z in np.arange(z0, z1, .55):
        for x in np.arange(x0, x1, .45):
            _P(r, [[x - .09, 0, z], [x + .09, 0, z], [x + .02, .32, z], [x - .02, .32, z]], (60, 170, 90), 1)
        r.segs(np.array([[x0, .003, z - .22]]), np.array([[x1, .003, z - .22]]), (60, 105, 140), 1)


def _wave_water(r, w, z0, z1, t, c1, c2):
    _P(r, [[-w, 0, z0], [w, 0, z0], [w, 0, z1], [-w, 0, z1]], c1, 0)
    for i, z in enumerate(np.arange(z0 + .4, z1, .8)):
        x0 = -w + ((i * 1.3 + t * .6) % 2.0)
        r.segs(np.array([[x, .004, z] for x in np.arange(x0, w, 2.0)]), np.array([[x + .8, .004, z] for x in np.arange(x0, w, 2.0)]), c2, 2)


def draw_set_pieces(r, spec, theme, t, cam_xz=None):
    """Camera ke samne wale hisse mein 'set dressing' (darakht, makaan, stall...) taake background khaali na lage. z = characters ke peeche."""
    w, d = spec["size"]["w"], spec["size"]["d"]
    night = spec.get("time") == "night" or theme == "campfire_night"
    zc = d * .32                                            # characters ki z
    items = []                                              # (z, fn) peeche se aage draw
    rng = np.random.RandomState(_stable_seed(theme + "sp"))

    if theme == "village_market":
        items += [(zc + 7.5, lambda: _house(r, -4.2, zc + 7.5, 1.5, (205, 225, 245), (70, 80, 190))),
                  (zc + 8.2, lambda: _house(r, 4.0, zc + 8.2, 1.4, (190, 220, 200), (60, 110, 200))),
                  (zc + 4.0, lambda: _stall(r, -3.0, zc + 4.0, 1.1, (60, 70, 215), t)),
                  (zc + 4.6, lambda: _stall(r, 3.1, zc + 4.6, 1.1, (80, 190, 80), t)),
                  (zc + 2.6, lambda: _crate(r, -1.9, zc + 2.6, 1.0)), (zc + 2.8, lambda: _crate(r, 2.0, zc + 2.8, 1.1, (80, 170, 110))),
                  (zc + 6.0, lambda: _tree(r, 0.0, zc + 6.0, 1.0))]
    elif theme == "island":
        items += [(zc + 6.0, lambda: _wave_water(r, w * .8, zc + 9.0, d * .96, t, (205, 170, 75), (255, 240, 220))),
                  (zc + 4.5, lambda: _tree_palm(r, -3.2, zc + 4.5, 1.2, t)), (zc + 5.0, lambda: _tree_palm(r, 3.4, zc + 5.0, 1.0, t)),
                  (zc + 2.0, lambda: _crate(r, 2.2, zc + 2.0, 1.0, (70, 120, 180))), (zc + 3.0, lambda: _blob3(r, -2.4, .4, zc + 3.0, .6, .4, (120, 125, 135), 2))]
    elif theme == "farm":
        items += [(zc + 3.0, lambda: _crop_rows(r, -3.8, -1.2, zc + 2.5, zc + 6.5)), (zc + 3.0, lambda: _crop_rows(r, 1.2, 3.8, zc + 2.5, zc + 6.5)),
                  (zc + 7.0, lambda: _tree(r, -4.4, zc + 7.0, 1.3)), (zc + 7.0, lambda: _tree(r, 4.6, zc + 7.0, 1.2)),
                  (zc + 3.0, lambda: _haystack(r, 2.4, zc + 2.0, 1.1)), (zc + 2.0, lambda: _fence_run(r, -4.5, 4.5, zc + 9.0))]
    elif theme == "city_street":
        items += [(zc + 7.5, lambda: _shopfront(r, -3.6, zc + 7.0, 1.2, (80, 90, 215), "BANK")), (zc + 7.5, lambda: _shopfront(r, 3.6, zc + 7.2, 1.2, (80, 180, 235), "CAFE")),
                  (zc + 3.0, lambda: _lamp(r, -2.3, zc + 3.0, 1.0, night)), (zc + 3.0, lambda: _lamp(r, 2.6, zc + 4.5, 1.0, night)),
                  (zc + 5.0, lambda: _crate(r, 1.2, zc + 5.0, .9, (90, 110, 130)))]
    elif theme == "park":
        items += [(zc + 7.0, lambda: _tree(r, -3.8, zc + 7.0, 1.4)), (zc + 7.5, lambda: _tree(r, 3.6, zc + 7.5, 1.3)),
                  (zc + 4.5, lambda: _bench(r, -2.6, zc + 4.5, 1.1)), (zc + 3.5, lambda: _lamp(r, 2.5, zc + 3.5, 1.0, night)),
                  (zc + 2.5, lambda: _flowerbed(r, 2.0, zc + 2.4, t))]
    elif theme == "school_yard":
        items += [(zc + 5.0, lambda: _tree(r, 3.8, zc + 5.0, 1.2)), (zc + 4.0, lambda: _bench(r, -3.0, zc + 4.0, 1.1)),
                  (zc + 3.2, lambda: _hoop(r, 3.0, zc + 3.2)), (zc + 6.0, lambda: _fence_run(r, -5, 5, zc + 8.0))]
    elif theme == "campfire_night":
        items += [(zc + 7.0, lambda: _tree(r, -3.8, zc + 7.0, 1.5, True, "pine")), (zc + 7.5, lambda: _tree(r, 3.9, zc + 7.6, 1.4, True, "pine")),
                  (zc + 5.0, lambda: _tree(r, 0.0, zc + 9.0, 1.3, True, "pine")), (zc + 3.2, lambda: _log(r, -1.8, zc + 2.5)), (zc + 3.2, lambda: _log(r, 1.8, zc + 2.6))]
    elif theme == "mountain_town":
        items += [(zc + 7.0, lambda: _tree(r, -3.6, zc + 7.0, 1.4, False, "pine")), (zc + 6.0, lambda: _tree(r, 3.8, zc + 6.0, 1.2, False, "pine")),
                  (zc + 4.0, lambda: _snowman(r, 2.4, zc + 3.4)), (zc + 7.5, lambda: _house(r, -2.0, zc + 8.5, 1.3, (110, 150, 205), (245, 245, 250)))]
    elif theme == "harbor":
        items += [(zc + 6.0, lambda: _wave_water(r, w * .8, zc + 5.5, d * .96, t, (150, 100, 75), (250, 210, 190))),
                  (zc + 3.0, lambda: _crate(r, -2.4, zc + 3.0, 1.1, (80, 140, 200))), (zc + 3.4, lambda: _crate(r, -1.9, zc + 3.4, .9, (90, 110, 210))),
                  (zc + 4.4, lambda: _bollard(r, 2.5, zc + 4.4)), (zc + 4.0, lambda: _lamp(r, -3.2, zc + 5.0, 1.0, night))]
    for _, fn in sorted(items, key=lambda it: -it[0]):
        try:
            fn()
        except Exception:
            pass


def _tree_palm(r, x, z, s, t):
    for k in range(6):
        y0, y1 = k * .6 * s, (k + 1) * .6 * s
        lean = .18 * s
        _P(r, [[x - .14 * s + lean * k / 6, y0, z], [x + .14 * s + lean * k / 6, y0, z], [x + .12 * s + lean * (k + 1) / 6, y1, z], [x - .12 * s + lean * (k + 1) / 6, y1, z]],
           (70, 120, 170) if k % 2 else (85, 140, 190), 2)
    tx, ty = x + .18 * s, 3.6 * s
    for a in (-2.8, -2.2, -1.6, -.9, -.35, .2):
        ex, ey = tx + 1.5 * s * math.cos(a + .08 * math.sin(t * 1.5 + x)), ty + .7 * s * math.sin(a) - .5 * s
        nx, ny = -(ey - ty) * .16, (ex - tx) * .16
        _P(r, [[tx, ty, z - .02], [(tx + ex) / 2 + nx, (ty + ey) / 2 + ny + .25 * s, z - .02], [ex, ey, z - .02], [(tx + ex) / 2 - nx, (ty + ey) / 2 - ny + .1, z - .02]], (80, 175, 70), 2)
    _blob3(r, tx, ty - .1, z - .03, .12 * s, .12 * s, (50, 90, 130), 1, 10)


def _fence_run(r, x0, x1, z):
    for x in np.arange(x0, x1 + .01, .8):
        r.box(x - .05, x + .05, 0, 1.0, z - .05, z + .05, (90, 135, 190), lw=1)
    for y in (.35, .75):
        r.box(x0, x1, y, y + .08, z - .03, z + .03, (110, 160, 215), lw=1)


def _shopfront(r, x, z, s, col, label):
    r.box(x - 1.5 * s, x + 1.5 * s, 0, 3.2 * s, z - .5 * s, z + .5 * s, col, lw=2)
    _quad_v(r, x - 1.1 * s, x + 1.1 * s, .3 * s, 1.9 * s, z - .52 * s, (240, 225, 175), 2)
    _quad_v(r, x - .45 * s, x + .45 * s, 2.35 * s, 2.85 * s, z - .52 * s, (250, 250, 250), 2)
    r.segs(np.array([[x - .35 * s, 2.6 * s, z - .54 * s]]), np.array([[x + .35 * s, 2.6 * s, z - .54 * s]]), OUT, 3)
    for i in range(7):
        xa = x - 1.2 * s + i * .34 * s
        _P(r, [[xa, 2.0 * s, z - .7 * s], [xa + .34 * s, 2.0 * s, z - .7 * s], [xa + .34 * s, 1.75 * s, z - .5 * s], [xa, 1.75 * s, z - .5 * s]],
           (245, 245, 245) if i % 2 else (60, 70, 215), 1)


def _lamp(r, x, z, s, night):
    r.box(x - .05 * s, x + .05 * s, 0, 3.0 * s, z - .05 * s, z + .05 * s, (80, 80, 95), lw=2)
    r.box(x - .05 * s, x + .5 * s, 2.95 * s, 3.02 * s, z - .05 * s, z + .05 * s, (80, 80, 95), lw=1)
    glow = (150, 235, 255) if night else (200, 245, 255)
    if night:
        _circle3(r, x + .45 * s, 2.85 * s, z - .05, .55 * s, (70, 110, 140))
    _blob3(r, x + .45 * s, 2.85 * s, z - .06, .16 * s, .12 * s, glow, 2)


def _bench(r, x, z, s):
    r.box(x - .8 * s, x + .8 * s, .42 * s, .5 * s, z - .25 * s, z + .25 * s, (80, 130, 190), lw=2)
    r.box(x - .8 * s, x + .8 * s, .6 * s, .95 * s, z + .22 * s, z + .27 * s, (70, 115, 175), lw=2)
    for dx in (-.7, .7):
        r.box(x + dx * s - .04, x + dx * s + .04, 0, .42 * s, z - .2 * s, z + .2 * s, (70, 70, 85), lw=1)


def _flowerbed(r, x, z, t):
    r.box(x - .9, x + .9, 0, .18, z - .4, z + .4, (90, 140, 200), lw=2)
    for i, col in enumerate([(80, 90, 240), (60, 210, 250), (250, 245, 245), (200, 110, 235), (80, 90, 240), (60, 210, 250)]):
        px = x - .75 + i * .3
        sw = .03 * math.sin(t * 2 + i)
        r.segs(np.array([[px, .18, z]]), np.array([[px + sw, .55, z]]), (60, 130, 60), 2)
        _circle3(r, px + sw, .6, z - .02, .1, col, 1, 10)


def _hoop(r, x, z):
    r.box(x - .04, x + .04, 0, 3.0, z - .04, z + .04, (110, 110, 125), lw=1)
    _quad_v(r, x - .6, x + .6, 2.8, 3.5, z - .06, (250, 250, 250), 2)
    _circle3(r, x, 2.95, z - .1, .22, (60, 130, 245), 2, 14)


def _log(r, x, z):
    r.box(x - .6, x + .6, 0, .35, z - .18, z + .18, (60, 95, 140), lw=2)
    _blob3(r, x - .6, .17, z - .19, .15, .17, (110, 160, 210), 2, 10)


def _snowman(r, x, z):
    _blob3(r, x, .5, z, .5, .5, (250, 248, 248), 2)
    _blob3(r, x, 1.25, z - .01, .36, .36, (250, 248, 248), 2)
    _blob3(r, x, 1.85, z - .02, .26, .26, (250, 248, 248), 2)
    _P(r, [[x, 1.85, z - .04], [x - .25, 1.82, z - .1], [x, 1.8, z - .04]], (40, 120, 250), 1)
    for dx in (-.08, .08):
        _circle3(r, x + dx, 1.92, z - .05, .025, OUT, 0, 6)


def _bollard(r, x, z):
    r.box(x - .13, x + .13, 0, .55, z - .13, z + .13, (60, 70, 200), lw=2)
    _blob3(r, x, .56, z - .1, .15, .07, (80, 90, 220), 1, 10)


# --------------------------------------------------------------------------------------------- ambient NPC extras (chhote background log) + animals
EXTRA_NAMES = {"market": ("ivy", "bruno", "paula", "dex", "tara", "sami"), "beach": ("dex", "ivy", "tara"), "farm": ("bruno", "gus", "tara"),
               "city": ("milo", "nina", "paula", "harlan", "dex"), "park": ("tara", "ivy", "milo", "gus", "sami"), "school": ("tara", "sami", "dex", "paula"),
               "camp": ("dex", "ivy"), "town": ("gus", "nina", "milo"), "port": ("bruno", "gus", "milo", "ivy")}


def npc_extras(spec, theme, t, main_slots):
    """Background mein chalte/khade logon ki list: [(pos_xyz, char_name, state)]. Slots se door rakhe jate hain."""
    th = THEMES[theme]
    d = spec["size"]["d"]
    names = EXTRA_NAMES.get(th["npc"], ("milo", "ivy"))
    zc = d * .32
    out = []
    seed = _stable_seed(theme + "npc")
    # (lane z offset, direction, speed, phase)
    lanes = [(5.4, 1, .55, 0.0), (7.0, -1, .42, .5), (6.2, 1, .35, .8)]
    for i, (dz, dr, sp, ph) in enumerate(lanes):
        span = 11.0
        x = ((ph * span + t * sp * dr + seed % 5) % span) - span / 2
        name = names[(i + seed) % len(names)]
        pos = np.array([x, 0.0, zc + dz])
        out.append((pos, name, {"t": t + i, "walk": True, "ph": t * 5.5 * (sp / .5) + i, "facing": "left" if dr < 0 else "right", "expr": "smile", "talk": 0.0,
                                "gesture": "idle", "extra": True}))
    # do khade hue log baatein karte hue
    if theme in ("village_market", "park", "city_street", "school_yard", "harbor"):
        for j, (x, nm) in enumerate(((-3.1, names[-1]), (-2.4, names[-2]))):
            out.append((np.array([x - 1.2 * (j == 0), 0.0, zc + 6.2 + j * .3]), nm,
                        {"t": t + j * .7, "expr": "happy", "talk": .5 + .4 * math.sin(t * 6 + j * 2) if j == 0 else 0.0, "gesture": "talk" if j == 0 else "idle",
                         "facing": "right" if j == 0 else "left", "extra": True}))
    return out


def draw_animals(r, spec, theme, t):
    """Chhote chalte jaanwar: murghiyan (farm), seagull (harbor/island), kekra (island), billi (city), titliyan (park/farm)."""
    w, d = spec["size"]["w"], spec["size"]["d"]
    zc = d * .32
    if theme == "farm":
        for i in range(3):
            x = -2.4 + i * 1.6 + .3 * math.sin(t * .8 + i)
            z = zc + 2.4 + i * .5
            peck = max(0, math.sin(t * 5 + i * 2)) * .14
            _chicken(r, x, z, peck, i)
    if theme in ("island", "harbor"):
        for i in range(3):
            bx = ((i * 3.1 + t * 1.2) % (w * 1.6)) - w * .8
            by = 4.8 + i * .5 + .3 * math.sin(t * 2 + i)
            fl = .22 * math.sin(t * 8 + i)
            z = zc + 9 + i
            r.segs(np.array([[bx - .4, by + fl, z]]), np.array([[bx, by, z]]), (250, 250, 250), 3)
            r.segs(np.array([[bx, by, z]]), np.array([[bx + .4, by + fl, z]]), (250, 250, 250), 3)
    if theme == "island":
        cx = 2.2 + .5 * math.sin(t * .9)
        _blob3(r, cx, .12, zc + 1.8, .17, .1, (70, 90, 230), 2, 12)
        for sd in (-1, 1):
            r.segs(np.array([[cx + sd * .12, .1, zc + 1.79]]), np.array([[cx + sd * .3, .22, zc + 1.79]]), (60, 80, 215), 2)
    if theme in ("park", "farm", "village_market"):
        for i in range(3):
            bx = ((i * 2.2 + t * .5) % 8) - 4 + .6 * math.sin(t * 1.3 + i)
            by = 1.0 + i * .35 + .25 * math.sin(t * 2.4 + i * 2)
            z = zc + 3 + i
            fl = .09 * math.sin(t * 16 + i)
            col = [(80, 150, 250), (200, 120, 240), (70, 220, 250)][i]
            _blob3(r, bx - .08, by + fl, z, .08, .06, col, 1, 8)
            _blob3(r, bx + .08, by + fl, z, .08, .06, col, 1, 8)
    if theme == "city_street":
        cx = -3.2
        _blob3(r, cx, .22, zc + 3.2, .28, .16, (120, 160, 220), 2, 12)
        _blob3(r, cx + .26, .34, zc + 3.19, .13, .12, (120, 160, 220), 2, 12)
        r.segs(np.array([[cx - .26, .22, zc + 3.19]]), np.array([[cx - .5, .4 + .08 * math.sin(t * 3), zc + 3.19]]), (120, 160, 220), 3)


def _chicken(r, x, z, peck, i):
    col = (250, 250, 250) if i % 2 == 0 else (90, 150, 215)
    _blob3(r, x, .28, z, .24, .19, col, 2, 14)
    _blob3(r, x + .2, .46 - peck, z - .01, .1, .1, col, 2, 12)
    _P(r, [[x + .29, .46 - peck, z - .02], [x + .38, .43 - peck, z - .02], [x + .29, .42 - peck, z - .02]], (40, 160, 250), 1)
    _P(r, [[x + .18, .55 - peck, z - .02], [x + .2, .62 - peck, z - .02], [x + .23, .55 - peck, z - .02]], (50, 50, 220), 1)
    for dx in (-.05, .05):
        r.segs(np.array([[x + dx, .1, z]]), np.array([[x + dx, 0, z]]), (40, 160, 250), 2)


# --------------------------------------------------------------------------------------------- particles (hawa mein dhool/patte/barf) aur vignette
def draw_particles(r, spec, theme, t):
    fr = r.fr
    kind = spec.get("weather", "clear")
    n = 26
    base = int(getattr(r, "fi", 0))
    for i in range(n):
        x = (i * 211 + base * (3 + i % 3)) % W
        y = (i * 137 + base * (1.4 + (i % 4) * .4) + 200 * math.sin(i)) % H
        if theme == "campfire_night":
            cv2.circle(fr, (int(x), int(H - y)), 2 + i % 2, (60, 170 + (i % 5) * 15, 255), -1, cv2.LINE_AA)           # chingariyan
        elif kind == "clear" and theme in ("park", "farm", "village_market"):
            if i % 4 == 0 and y > H * .55:
                cv2.ellipse(fr, (int(x), int(y)), (7, 3), 25 * math.sin(base * .1 + i), 0, 360, (70, 160, 110), -1, cv2.LINE_AA)  # udtay patte
        elif theme in ("island", "harbor", "city_street", "school_yard") and i % 5 == 0:
            cv2.circle(fr, (int(x), int(y)), 2, (235, 245, 255), -1, cv2.LINE_AA)                                    # dhool / jhag


def vignette(frame, strength=.28):
    key = ("vig", frame.shape[1], frame.shape[0], strength)
    m = _VIG.get(key)
    if m is None:
        yy, xx = np.mgrid[0:frame.shape[0], 0:frame.shape[1]].astype(np.float32)
        rr = np.sqrt(((xx - frame.shape[1] / 2) / (frame.shape[1] / 2)) ** 2 + ((yy - frame.shape[0] / 2) / (frame.shape[0] / 2)) ** 2)
        m = (1 - strength * np.clip(rr - .55, 0, 1) ** 1.6)[..., None].astype(np.float32)
        _VIG[key] = m
    frame[:] = np.clip(frame.astype(np.float32) * m, 0, 255).astype(np.uint8)


_VIG = {}


# --------------------------------------------------------------------------------------------- indoor decor (khidki, frames, rug, shelf, ceiling light)
def draw_indoor_decor(r, spec, t):
    w, d, h = spec["size"]["w"], spec["size"]["d"], spec["size"]["h"]
    wall = spec["walls"]["color"]
    night = spec.get("time") == "night"
    zb = d - .14
    # dado stripe + wainscot
    _P(r, [[-w / 2, .16, zb], [w / 2, .16, zb], [w / 2, .95, zb], [-w / 2, .95, zb]], dark(wall, .86), 0)
    r.segs(np.array([[-w / 2, .95, zb - .01]]), np.array([[w / 2, .95, zb - .01]]), dark(wall, .62), 3)
    # khidki (bahar ka aasman + patte) + dhoop ki roshni
    for wx in (-w * .27, w * .27):
        _quad_v(r, wx - .8, wx + .8, 1.2, 2.7, zb - .01, (150, 110, 80) if night else (225, 190, 130), 3)
        _quad_v(r, wx - .72, wx + .72, 1.28, 2.62, zb - .02, (80, 40, 25) if night else (245, 215, 150), 0)
        _blob3(r, wx + .25, 1.85, zb - .03, .35, .13, (255, 252, 250), 0, 12)
        _blob3(r, wx - .3, 2.3, zb - .03, .28, .1, (255, 252, 250), 0, 12)
        r.segs(np.array([[wx, 1.2, zb - .035]]), np.array([[wx, 2.7, zb - .035]]), (150, 110, 80), 3)
        r.segs(np.array([[wx - .8, 1.95, zb - .035]]), np.array([[wx + .8, 1.95, zb - .035]]), (150, 110, 80), 3)
        _P(r, [[wx - .8, 1.2, zb - .01], [wx + .8, 1.2, zb - .01], [wx + 1.9, .02, zb - 3.2], [wx - .1, .02, zb - 3.2]], mix(spec["ground"]["color"], (190, 235, 255), .30), 0)
    # frames, ghari, shelf
    for i, (fx, fy, c) in enumerate(((-.0, 2.3, (90, 130, 215)), (w * .43, 2.1, (110, 190, 120)), (-w * .43, 2.0, (200, 140, 90)))):
        _quad_v(r, fx - .38, fx + .38, fy - .3, fy + .3, zb - .015, (80, 125, 175), 3)
        _quad_v(r, fx - .3, fx + .3, fy - .22, fy + .22, zb - .02, c, 0)
        _P(r, [[fx - .3, fy - .22, zb - .025], [fx, fy + .05, zb - .025], [fx + .3, fy - .22, zb - .025]], dark(c, .75), 0)
    _circle3(r, w * .09, 3.0, zb - .02, .27, (248, 246, 244), 3, 20)
    ang = t * .2
    r.segs(np.array([[w * .09, 3.0, zb - .03]]), np.array([[w * .09 + .17 * math.sin(ang), 3.0 + .17 * math.cos(ang), zb - .03]]), OUT, 2)
    r.segs(np.array([[w * .09, 3.0, zb - .03]]), np.array([[w * .09 + .1 * math.sin(ang * 12), 3.0 + .1 * math.cos(ang * 12), zb - .03]]), OUT, 2)
    # chhat ki light
    for lx in (-w * .22, w * .22):
        r.segs(np.array([[lx, h, d * .45]]), np.array([[lx, h - .4, d * .45]]), OUT, 2)
        _blob3(r, lx, h - .48, d * .45, .3, .12, (200, 245, 255), 2, 14)
    # qaleen
    _P(r, [[-w * .22, .005, d * .12], [w * .22, .005, d * .12], [w * .22, .005, d * .55], [-w * .22, .005, d * .55]], (150, 110, 190) if not night else (110, 70, 130), 2)
    _P(r, [[-w * .17, .006, d * .16], [w * .17, .006, d * .16], [w * .17, .006, d * .51], [-w * .17, .006, d * .51]], (190, 160, 235), 1)
