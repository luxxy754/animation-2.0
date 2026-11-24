"""toon/rooms.py — 4 fixed rooms: office, corridor, break_room, night_exterior (har video mein wohi duniya).
ROOMS[name] = dict(dims=(xmin,xmax,zmax,H), slots={left,center,right,far,(seat)}, lights=[...], key=shadow light index,
door={"pos":(x,z)}, objs=[...], fx=[...], shell_draw(r), tone, amb0).  Static cheezon mein koi jitter nahi."""
import math
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont
from .core import clamp, OUT, FONT
from .lighting import light_k, proj_dir


def P3(*a): return np.array(a, float)
def _cs(c, k): return tuple(int(clamp(v * k, 0, 255)) for v in c)
def _tones(base, mult): return [_cs(base, m) for m in mult]


# ---------------------------------------------------------------- textures (import par ek baar)
def _font(sz):
    try: return ImageFont.truetype(FONT, sz)
    except Exception: return ImageFont.load_default()


def _text_tex(lines, sizes, colors, bg, wh, border=None):
    im = Image.new("RGB", wh, bg); d = ImageDraw.Draw(im); y = 6
    for ln, sz, c in zip(lines, sizes, colors):
        f = _font(sz); w = d.textlength(ln, font=f); d.text(((wh[0] - w) / 2, y), ln, font=f, fill=c); y += sz + 6
    if border: d.rectangle([0, 0, wh[0] - 1, wh[1] - 1], outline=border, width=6)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def _graph_tex():
    im = Image.new("RGB", (200, 260), (250, 245, 235)); d = ImageDraw.Draw(im)
    for i, h in enumerate((60, 110, 80, 160, 130)): d.rectangle([20 + i * 34, 220 - h, 46 + i * 34, 220], fill=((220, 80, 60), (60, 140, 220), (80, 190, 110))[i % 3], outline=(20, 20, 30), width=3)
    d.line([(10, 224), (190, 224)], fill=(20, 20, 30), width=4); d.rectangle([2, 2, 197, 257], outline=(30, 30, 40), width=5)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


SIGN = _text_tex(["CAREER", "DEVELOPMENT", "ROOM"], [30, 30, 40], [(150, 70, 20), (150, 70, 20), (40, 40, 220)], (255, 255, 255), (380, 150), (200, 120, 40))
CAL = _text_tex(["APR", "7", "MON"], [34, 80, 30], [(255, 255, 255), (30, 30, 40), (200, 40, 40)], (250, 250, 250), (150, 210), (200, 40, 40))
SAFE = _text_tex(["SAFETY", "FIRST"], [44, 52], [(30, 30, 40), (200, 40, 40)], (250, 210, 40), (230, 150), (30, 30, 40))
EXIT = _text_tex(["EXIT"], [60], [(255, 255, 255)], (40, 160, 60), (200, 90), (255, 255, 255))
TOWER = _text_tex(["CORP", "TOWER"], [50, 50], [(255, 255, 255), (255, 210, 90)], (30, 40, 90), (300, 140), (255, 210, 90))
GRAPH = _graph_tex()


# ---------------------------------------------------------------- shell: tiles/bricks (vectorised draw)
def _tiles(plane, a0, a1, b0, b1, da, db, stagger, seed, ntone, checker=False):
    rng = np.random.RandomState(seed); Q = []; T = []; A = []; Bq = []
    nrow = max(1, int(math.ceil((b1 - b0) / db - 1e-9)))
    for i in range(nrow):
        y0 = b0 + i * db; y1 = min(b0 + (i + 1) * db, b1); off = da / 2 if (stagger and i % 2) else 0.0
        e = [a0]; x = a0 + (off if off else da)
        while x < a1 - 1e-6: e.append(x); x += da
        e.append(a1)
        for j in range(len(e) - 1):
            Q.append([plane(e[j], y0), plane(e[j + 1], y0), plane(e[j + 1], y1), plane(e[j], y1)])
            T.append(((i + j) % 2) * 2 + rng.randint(2) if checker else rng.randint(ntone))
            if j > 0: A.append(plane(e[j], y0)); Bq.append(plane(e[j], y1))
        if i < nrow - 1: A.append(plane(a0, y1)); Bq.append(plane(a1, y1))
    z3 = np.zeros((0, 3))
    return np.array(Q, float), np.array(T, int), (np.array(A, float) if A else z3), (np.array(Bq, float) if Bq else z3)


def _surf(plane, a0, a1, b0, b1, da, db, stagger, seed, cols, line, lw, ctr, nrm, L, amb0, boost=1.0, checker=False):
    Q, T, A, Bq = _tiles(plane, a0, a1, b0, b1, da, db, stagger, seed, len(cols), checker)
    k = np.clip(light_k(L, ctr, nrm, False, amb0) * boost, 0.55, 1.45)      # surface ka roshni-rang bake
    pal = np.array([[int(clamp(c[i] * k[i], 0, 255)) for i in range(3)] for c in cols], np.uint8)
    return dict(Q=Q, T=T, pal=pal, A=A, B=Bq, lc=tuple(int(clamp(line[i] * k[i], 0, 255)) for i in range(3)), lw=lw)


def _draw_surfs(r, surfs):
    for s in surfs:
        r.fill_quads(s["Q"], s["T"], s["pal"])
        if len(s["A"]): r.segs(s["A"], s["B"], s["lc"], s["lw"])


def _wallks(dims, L, a0):
    x0, x1, zm, Hh = dims; cy = Hh * .45
    return {"left": light_k(L, (x0, cy, zm / 2), (1, 0, 0), False, a0), "right": light_k(L, (x1, cy, zm / 2), (-1, 0, 0), False, a0),
            "back": light_k(L, ((x0 + x1) / 2, cy, zm), (0, 0, -1), False, a0)}


class Wall:
    """Deewar par flat decor (u = deewar ke saath, y = unchai)."""
    def __init__(s, name, dims, k): s.name = name; s.k = k; s.x0, s.x1, s.zm, s.H = dims

    def P(s, u, y, off=0.0):
        if s.name == "back": return P3(u, y, s.zm - off)
        if s.name == "left": return P3(s.x0 + off, y, u)
        return P3(s.x1 - off, y, u)

    def rect(s, u0, u1, y0, y1, off=.01): return [s.P(u0, y0, off), s.P(u1, y0, off), s.P(u1, y1, off), s.P(u0, y1, off)]
    def c(s, col): return tuple(int(clamp(v * s.k[i], 0, 255)) for i, v in enumerate(col))

    def fill(s, r, u0, u1, y0, y1, col, lw=2, off=.01, raw=False):
        r.poly(s.rect(u0, u1, y0, y1, off), col if raw else s.c(col), lw, tint=False)

    def circ(s, r, u, y, rad, col, lw=2, off=.01, raw=False):
        r.poly([s.P(u + rad * math.cos(a), y + rad * math.sin(a), off) for a in np.linspace(0, 2 * math.pi, 26, endpoint=False)], col if raw else s.c(col), lw, tint=False)

    def line(s, r, u0, y0, u1, y1, col=OUT, lw=2, off=.012):
        r.segs(np.array([s.P(u0, y0, off)]), np.array([s.P(u1, y1, off)]), col, lw)

    def tex(s, r, img, u0, u1, y0, y1, off=.015):
        if s.name == "right": u0, u1 = u1, u0               # right wall par u ulta chalta hai
        r.tex(img, [s.P(u0, y1, off), s.P(u1, y1, off), s.P(u1, y0, off), s.P(u0, y0, off)], float(np.mean(s.k)))

    def door(s, r, u0, u1, col=(150, 175, 205)):
        s.fill(r, u0 - .07, u1 + .07, 0, 2.3, (92, 92, 105), 3); s.fill(r, u0, u1, 0, 2.2, col, 2, .015)
        s.fill(r, u0 + .18, u1 - .18, 1.4, 1.95, (170, 195, 110), 2, .02); s.fill(r, u1 - .17, u1 - .1, .98, 1.04, (60, 190, 230), 2, .02)

    def window(s, r, u0, u1, y0, y1, seed=3, sky=((255, 232, 190), (250, 190, 120))):
        h = y1 - y0; s.fill(r, u0 - .07, u1 + .07, y0 - .07, y1 + .07, (235, 205, 155), 4)
        for i in range(4):
            f = i / 3.0; s.fill(r, u0, u1, y0 + h * i / 4, y0 + h * (i + 1) / 4, tuple(int(a + (b - a) * f) for a, b in zip(*sky)), 0, .02, True)
        rng = np.random.RandomState(seed)
        for layer, (col, hmax) in enumerate((((215, 185, 150), .55), ((165, 135, 105), .38))):
            u = u0
            while u < u1:
                w = rng.uniform(.35, .8) * (1 - .25 * layer); hh = rng.uniform(.15, hmax) * h
                s.fill(r, u, min(u + w, u1), y0, y0 + hh, col, 1 if layer else 0, .03 + .01 * layer, True); u += w + rng.uniform(0, .1)
        for uu in np.linspace(u0, u1, 5)[1:-1]: s.line(r, uu, y0, uu, y1, OUT, 2, .045)
        s.line(r, u0, (y0 + y1) / 2, u1, (y0 + y1) / 2, OUT, 2, .045)


# ---------------------------------------------------------------- objects (parts = boxes; extras = flat details)
def O(c, parts, extras=(), boxes=None):
    cs = np.array([[(p[0] + p[1]) / 2, (p[2] + p[3]) / 2, (p[4] + p[5]) / 2] for p in parts], float) if parts else np.zeros((0, 3))
    return dict(c=P3(*c), parts=parts, cs=cs, extras=list(extras), boxes=[tuple(p[:6]) for p in parts] if boxes is None else boxes)


def draw_obj(r, o):
    if len(o["parts"]):
        z = r.c.cam(o["cs"])[:, 2]
        for i in np.argsort(-z):
            p = o["parts"][i]; r.box(*p[:7], lw=(p[7] if len(p) > 7 else 2))
    for f in o["extras"]: f(r)


def _desk(x0, x1, z0, z1, top, side):
    return [(x0, x1, .72, .78, z0, z1, top), (x0, x0 + .06, 0, .72, z0, z1, side), (x1 - .06, x1, 0, .72, z0, z1, side), (x0, x1, .25, .72, z0, z0 + .05, side)]


def _monitor(x, z):
    d = (70, 68, 72)
    parts = [(x - .09, x + .09, .78, .80, z - .07, z + .07, d, 1), (x - .025, x + .025, .80, .95, z - .02, z + .02, d, 1), (x - .30, x + .30, .95, 1.34, z - .03, z + .03, (55, 50, 55), 2)]

    def ex(r):
        r.poly([P3(x - .27, .98, z - .032), P3(x + .27, .98, z - .032), P3(x + .27, 1.31, z - .032), P3(x - .27, 1.31, z - .032)], (250, 215, 150), 1, tint=False)
        for yy, w in ((1.25, .2), (1.18, .15), (1.11, .22)): r.segs(np.array([[x - .22, yy, z - .034]]), np.array([[x - .22 + w, yy, z - .034]]), (190, 150, 90), 2)
    return parts, [ex]


def _ws(dx, mon, kb, mug, paper, dcol=(200, 215, 235), scol=(170, 190, 215), lamp=None):
    parts = _desk(*dx, dcol, scol); ex = []
    p, e = _monitor(*mon); parts += p; ex += e
    parts += [(kb[0] - .22, kb[0] + .22, .78, .80, kb[1] - .09, kb[1] + .09, (205, 205, 210), 1), (mug[0] - .05, mug[0] + .05, .78, .87, mug[1] - .05, mug[1] + .05, (60, 60, 220), 1),
              (paper[0] - .15, paper[0] + .15, .78, .783, paper[1] - .15, paper[1] + .15, (250, 250, 250), 1)]
    if lamp:
        lx, lz = lamp; parts.append((lx - .07, lx + .07, .78, .80, lz - .07, lz + .07, (70, 70, 75), 1))

        def lex(r):
            r.segs(np.array([[lx, .8, lz], [lx, 1.12, lz]]), np.array([[lx, 1.12, lz], [lx + .1, 1.2, lz - .05]]), OUT, 3)
            r.poly([P3(lx + .0, 1.1, lz - .06), P3(lx + .22, 1.1, lz - .06), P3(lx + .17, 1.27, lz - .06), P3(lx + .05, 1.27, lz - .06)], (80, 190, 240), 2, tint=False)
        ex.append(lex)
    return parts, ex


def _chair(x, z, col=(120, 90, 70)):
    parts = [(x - .24, x + .24, .43, .50, z - .24, z + .24, col), (x - .23, x + .23, .50, .98, z + .19, z + .26, col), (x - .025, x + .025, .12, .43, z - .025, z + .025, (70, 70, 75), 1)]

    def ex(r):
        Bq = np.array([[x + .27 * math.cos(a), .05, z + .27 * math.sin(a)] for a in np.linspace(.3, .3 + 2 * math.pi, 5, endpoint=False)])
        r.segs(np.array([[x, .1, z]] * 5), Bq, OUT, 2)
        for b in Bq: r.disc(b + P3(0, .03, 0), .035, (50, 50, 55), 1)
    return O((x, .5, z + .3), parts, [ex], boxes=[(x - .24, x + .24, 0, .5, z - .24, z + .24), (x - .23, x + .23, .5, .98, z + .19, z + .26)])


def _plant(x, z):
    def ex(r):
        r.cyl(x, z, 0, .35, .17, (70, 110, 170), 2)
        for dx, dz, hh in ((.25, 0, .55), (-.25, 0, .5), (0, .2, .65), (.12, -.15, .45), (-.1, -.1, .6)):
            r.poly([P3(x, .38, z), P3(x + dx * .5 - dz * .25, .38 + hh * .6, z + dz * .5 + dx * .25), P3(x + dx, .38 + hh, z + dz), P3(x + dx * .5 + dz * .25, .38 + hh * .55, z + dz * .5 - dx * .25)],
                   r.shade((70, 170, 60), (x, .6, z), None, True), 1.5, tint=False, boil=.7)
    return O((x, .4, z), [], [ex], boxes=[(x - .17, x + .17, 0, .35, z - .17, z + .17)])


def _cabinet(x0, x1, z0, z1, h, col=(165, 160, 150)):
    def ex(r):
        for yy in np.linspace(h / 3, h * 2 / 3, 2): r.segs(np.array([[x0, yy, z0 - .004]]), np.array([[x1, yy, z0 - .004]]), OUT, 2)
        for yy in (h / 6, h / 2, h * 5 / 6): r.poly([P3((x0 + x1) / 2 - .06, yy - .015, z0 - .006), P3((x0 + x1) / 2 + .06, yy - .015, z0 - .006), P3((x0 + x1) / 2 + .06, yy + .015, z0 - .006), P3((x0 + x1) / 2 - .06, yy + .015, z0 - .006)], (90, 90, 95), 1, tint=False)
    return O(((x0 + x1) / 2, h / 2, (z0 + z1) / 2), [(x0, x1, 0, h, z0, z1, col)], [ex])


def _cooler(x, z):
    def ex(r):
        r.cyl(x, z, .95, 1.45, .14, (255, 205, 130), 2); r.disc(P3(x - .06, .78, z - .185), .02, (60, 60, 220), 1); r.disc(P3(x + .06, .78, z - .185), .02, (230, 120, 40), 1)
    return O((x, .7, z), [(x - .18, x + .18, 0, .95, z - .18, z + .18, (228, 228, 232))], [ex])


def _bench(x0, x1, z0, z1):
    return O(((x0 + x1) / 2, .3, (z0 + z1) / 2), [(x0, x1, .40, .46, z0, z1, (90, 130, 190)), (x0, x0 + .05, 0, .40, z0, z0 + .06, (70, 70, 75), 1), (x0, x0 + .05, 0, .40, z1 - .06, z1, (70, 70, 75), 1),
                                                (x1 - .05, x1, .46, .95, z0, z1, (90, 130, 190))])


def _rack(x, z):
    def ex(r):
        for dx, dy, col in ((-.22, 1.62, (60, 70, 190)), (.22, 1.52, (170, 120, 50))):
            r.segs(np.array([[x, 1.78, z - .03]]), np.array([[x + dx, dy + .1, z - .03]]), OUT, 2)
            r.poly([P3(x + dx - .14, dy - .75, z - .08), P3(x + dx + .14, dy - .75, z - .08), P3(x + dx + .12, dy, z - .08), P3(x + dx - .12, dy, z - .08)], r.shade(col, (x, 1, z), None, True), 2, tint=False, boil=.6)
    return O((x, 1, z), [(x - .2, x + .2, 0, .04, z - .2, z + .2, (60, 60, 65)), (x - .025, x + .025, .04, 1.85, z - .025, z + .025, (60, 60, 65), 1)], [ex])


def _lampfx(pos, rad, col, pool=None):
    f = [dict(t="glow", pos=pos, rad=rad, col=col)]
    if pool: f.append(dict(t="pool", c=pool[0], rx=pool[1], rz=pool[2], y=pool[3], col=tuple(int(v * .35) for v in col)))
    return f


def _hull2(pts):
    h = cv2.convexHull(np.array(pts, np.float32)).reshape(-1, 2); return h


def _sun_fx(win, D, dims, pcol, scol, ln=6.0):
    """Khidki se shaft + farsh/deewar par hard-edged dhoop ke patches (world mein, camera ke saath perspective mein chalte hain)."""
    x0, x1, zm, Hh = dims; D = np.array(D, float); fx = [dict(t="shaft", quad=win, dir=D, len=ln, col=scol, n=7)]
    fl = [proj_dir(p, D, "y", 0.0) for p in win]
    pts = [(clamp(q[0], x0 + .01, x1 - .01), clamp(q[2], .01, zm - .01)) for q in fl if q is not None]
    if len(pts) >= 3: fx.append(dict(t="hard", quad=[(a, .012, b) for a, b in _hull2(pts)], col=pcol))
    if D[2] > .05:
        bk = [(clamp(q[0], x0 + .01, x1 - .01), clamp(q[1], .0, Hh)) for q in (proj_dir(p, D, "z", zm) for p in win) if q is not None]
        if len(bk) >= 3 and np.mean([b[1] for b in bk]) > .3: fx.append(dict(t="hard", quad=[(a, b, zm - .012) for a, b in _hull2(bk)], col=tuple(int(v * .8) for v in pcol)))
    if D[0] < -.05:
        lw_ = [(clamp(q[1], .0, Hh), clamp(q[2], .0, zm)) for q in (proj_dir(p, D, "x", x0) for p in win) if q is not None]
        if len(lw_) >= 3 and np.mean([b[0] for b in lw_]) > .3: fx.append(dict(t="hard", quad=[(x0 + .012, a, b) for a, b in _hull2(lw_)], col=tuple(int(v * .8) for v in pcol)))
    return fx


def _panels(pos, w, d, Hh):
    return [dict(t="panel", quad=[P3(x - w, Hh - .02, z - d), P3(x + w, Hh - .02, z - d), P3(x + w, Hh - .02, z + d), P3(x - w, Hh - .02, z + d)], col=(70, 70, 62)) for x, z in pos]


def _ceil_panels(r, pos, w, d, Hh):
    for x, z in pos: r.poly([P3(x - w, Hh - .01, z - d), P3(x + w, Hh - .01, z - d), P3(x + w, Hh - .01, z + d), P3(x - w, Hh - .01, z + d)], (255, 255, 252), 2, tint=False)


def _vent(r, x, z, Hh):
    r.poly([P3(x - .3, Hh - .01, z - .3), P3(x + .3, Hh - .01, z - .3), P3(x + .3, Hh - .01, z + .3), P3(x - .3, Hh - .01, z + .3)], (205, 200, 208), 2, tint=False)
    r.segs(np.array([[x - .3, Hh - .012, z + dz] for dz in (-.15, -.05, .05, .15)]), np.array([[x + .3, Hh - .012, z + dz] for dz in (-.15, -.05, .05, .15)]), (140, 135, 145), 1)


def _slot(x, z): return {"pos": (float(x), float(z))}


# ================================================================ OFFICE
def _office():
    dims = (-4.0, 4.0, 12.0, 3.2); x0, x1, zm, Hh = dims; a0 = .74
    PAN = [(x, z) for x in (-1.6, 1.6) for z in (2.5, 6.0, 9.5)]
    L = [dict(pos=(x, 3.15, z), color=(255, 250, 242), intensity=.12, kind="panel") for x, z in PAN]
    L += [dict(pos=(30., 14., 5.), color=(70, 225, 255), intensity=.38, kind="sun"),            # idx 6: khidki ki garam dhoop
          dict(pos=(.78, 1.25, 6.2), color=(110, 190, 255), intensity=.30, kind="lamp"),        # desk lamp (garam)
          dict(pos=(-3.15, 1.2, 7.9), color=(255, 170, 90), intensity=.25, kind="lamp")]        # thandi neeli lamp
    wk = _wallks(dims, L, a0); C = lambda p, n, **k: _surf(*p, L=L, amb0=a0, ctr=n[0], nrm=n[1], **k)
    surfs = [
        _surf(lambda u, v: (u, 0.0, v), x0, x1, 0, zm, 1.0, 1.0, False, 1, _tones((205, 62, 22), (.92, .97, 1.0, 1.04, 1.08)), (230, 115, 62), 1, (0, 0, 6), (0, 1, 0), L, a0),
        _surf(lambda u, v: (u, Hh, v), x0, x1, 0, zm, .8, .8, False, 2, _tones((246, 246, 246), (.96, .98, 1.0)), (205, 200, 205), 1, (0, Hh, 6), (0, -1, 0), L, a0, boost=1.3),
        _surf(lambda u, y: (u, y, zm), x0, x1, 0, Hh, .56, .28, True, 3, _tones((125, 185, 135), (.93, .97, 1.0, 1.04, 1.08)), (92, 148, 100), 1, (0, 1.4, zm), (0, 0, -1), L, a0),
        _surf(lambda u, y: (x0, y, u), 0, zm, 0, Hh, .56, .28, True, 4, _tones((125, 185, 135), (.93, .97, 1.0, 1.04, 1.08)), (92, 148, 100), 1, (x0, 1.4, 6), (1, 0, 0), L, a0),
        _surf(lambda u, y: (x1, y, u), 0, zm, 0, Hh, .56, .28, True, 5, _tones((125, 185, 135), (.93, .97, 1.0, 1.04, 1.08)), (92, 148, 100), 1, (x1, 1.4, 6), (-1, 0, 0), L, a0)]
    B_, Lf, Rt = Wall("back", dims, wk["back"]), Wall("left", dims, wk["left"]), Wall("right", dims, wk["right"])
    win = [P3(x1, 1.0, 2.5), P3(x1, 1.0, 8), P3(x1, 2.7, 8), P3(x1, 2.7, 2.5)]
    sunD = (-.62, -.40, .45)

    def shell(r):
        _draw_surfs(r, surfs)
        for w_, (u0, u1) in ((B_, (x0, x1)), (Lf, (0, zm)), (Rt, (0, zm))): w_.fill(r, u0, u1, 0, .12, (110, 80, 60), 1)
        r.poly([P3(-2.3, .01, 3.0), P3(2.3, .01, 3.0), P3(2.3, .01, 5.4), P3(-2.3, .01, 5.4)], r.shade((60, 70, 150), (0, 0, 4), (0, 1, 0)), 2, tint=False)     # rug
        r.poly([P3(-2.1, .012, 3.2), P3(2.1, .012, 3.2), P3(2.1, .012, 5.2), P3(-2.1, .012, 5.2)], r.shade((90, 120, 200), (0, 0, 4), (0, 1, 0)), 1, tint=False)
        _ceil_panels(r, PAN, .6, .6, Hh)
        for z in (2.5, 7.5, 10.5): _vent(r, 0, z, Hh)
        B_.door(r, -.54, .54, (205, 182, 150)); B_.tex(r, SIGN, -.36, .36, 1.03, 1.34, .03); B_.fill(r, 1.0, 1.18, 1.2, 1.45, (40, 40, 205), 3)
        B_.circ(r, -2.2, 2.45, .39, (40, 40, 200), 3); B_.circ(r, -2.2, 2.45, .33, (255, 255, 255), 2, .02); B_.line(r, -2.2, 2.45, -2.18, 2.68, OUT, 3, .03); B_.line(r, -2.2, 2.45, -2.03, 2.35, OUT, 3, .03)
        B_.fill(r, -3.4, -1.3, 1.1, 2.0, (250, 250, 250), 3); B_.line(r, -3.2, 1.7, -2.2, 1.7, (200, 60, 40), 2); B_.line(r, -3.2, 1.45, -1.9, 1.45, (60, 120, 220), 2)
        Lf.fill(r, 2.2, 3.6, 1.3, 2.3, (70, 120, 170), 4); Lf.fill(r, 2.28, 3.52, 1.38, 2.22, (110, 170, 210), 1, .02)
        for i, (u, y, c) in enumerate(((2.4, 1.9, (80, 220, 250)), (2.9, 1.6, (250, 200, 120)), (3.2, 2.0, (200, 160, 250)), (2.6, 1.5, (120, 240, 160)))): Lf.fill(r, u, u + .22, y, y + .22, c, 1, .03, True)
        Lf.tex(r, CAL, 7.0, 7.5, 1.5, 2.1); Lf.tex(r, GRAPH, 4.4, 5.1, 1.5, 2.4)
        Rt.window(r, 2.5, 8.0, 1.0, 2.7, 3); Rt.tex(r, SAFE, 9.0, 9.9, 1.3, 1.95)
        r.ao_room()
    objs = []
    p, e = _ws((-1.0, 1.0, 5.6, 6.4), (-.55, 6.15), (.25, 5.85), (.65, 5.85), (0, 6.2), lamp=(.78, 6.2)); objs.append(O((0, .4, 6.0), p, e))
    for dx, mon, kb, mug, pa, lp in (((-3.9, -2.9, 2.8, 4.4), (-3.45, 3.7), (-3.45, 3.05), (-3.05, 3.9), (-3.45, 3.5), None), ((-3.9, -2.9, 6.6, 8.2), (-3.45, 7.5), (-3.45, 6.85), (-3.1, 7.8), (-3.45, 7.4), (-3.15, 7.9)),
                                     ((2.9, 3.9, 2.8, 4.4), (3.45, 3.7), (3.45, 3.05), (3.05, 3.9), (3.45, 3.5), None), ((2.9, 3.9, 8.2, 9.8), (3.45, 9.1), (3.45, 8.45), (3.1, 9.4), (3.45, 9.0), None)):
        p, e = _ws(dx, mon, kb, mug, pa, lamp=lp); objs.append(O(((dx[0] + dx[1]) / 2, .4, (dx[2] + dx[3]) / 2), p, e))
    for x, z, c in ((0, 6.95, (120, 90, 70)), (-2.4, 3.6, (90, 130, 190)), (-2.4, 7.4, (150, 110, 80)), (2.4, 3.6, (80, 150, 110)), (2.4, 9.0, (120, 90, 70))): objs.append(_chair(x, z, c))
    for xa, xb, za in ((-3.9, -2.4, 5.2), (2.4, 3.9, 7.35), (-3.9, -2.4, 8.5), (2.4, 3.9, 5.2)): objs.append(O(((xa + xb) / 2, .6, za), [(xa, xb, 0, 1.35, za, za + .07, (150, 135, 125))]))
    bk = O((-3.8, 1.2, 11.2), [(-4.0, -3.6, 0, 2.3, 10.4, 11.9, (70, 90, 130))], [lambda r: [r.poly([P3(-3.595, y, z), P3(-3.595, y, z + .14), P3(-3.595, y + .5, z + .14), P3(-3.595, y + .5, z)], r.shade(c, (-3.6, y, z), (1, 0, 0), True), 1, tint=False)
                                                                                for y in (.55, 1.15, 1.75) for z, c in zip(np.arange(10.55, 11.85, .2), ((60, 60, 200), (200, 120, 40), (60, 170, 220), (90, 170, 80), (180, 80, 160), (60, 60, 200), (200, 160, 50)))]])
    objs += [bk, _cooler(-3.5, 1.5), _cabinet(3.3, 3.95, 10.2, 10.8, 1.3), _cabinet(3.3, 3.95, 10.85, 11.45, 1.3), _plant(3.5, 5.9), _plant(-3.5, 5.9), _rack(1.9, 11.4)]
    objs.append(O((3.45, .5, 1.6), [(3.0, 3.9, 0, .7, 1.2, 2.0, (120, 165, 205)), (3.1, 3.8, .7, 1.0, 1.3, 1.9, (220, 220, 225)), (3.15, 3.75, .98, 1.02, 1.3, 1.5, (250, 250, 250), 1)]))
    fx = _panels(PAN, .6, .6, Hh) + _sun_fx(win, sunD, dims, (40, 95, 120), (45, 105, 125)) + _lampfx((.78, 1.2, 6.2), .4, (70, 150, 210), ((.78, 6.1), .35, .3, .79)) + _lampfx((-3.05, 1.15, 7.9), .35, (200, 130, 60))
    return dict(name="office", dims=dims, lights=L, key=6, door=dict(pos=(0.0, zm)), amb0=a0, tone=(1.0, 1.0, 1.0), objs=objs, fx=fx, shell_draw=shell,
                slots=dict(left=_slot(-1.5, 4.3), center=_slot(0, 4.3), right=_slot(1.5, 4.3), far=_slot(.6, 10.2), seat=_slot(0, 6.65)))


# ================================================================ CORRIDOR
def _corridor():
    dims = (-1.6, 1.6, 14.0, 3.0); x0, x1, zm, Hh = dims; a0 = .74
    PZ = (2, 5, 8, 11)
    L = [dict(pos=(0, 2.95, z), color=(255, 250, 242), intensity=.15, kind="panel") for z in PZ]
    L += [dict(pos=(-1.5, 2.1, 5.6), color=(110, 190, 255), intensity=.35, kind="lamp"), dict(pos=(-1.5, 2.1, 9.5), color=(110, 190, 255), intensity=.35, kind="lamp"),
          dict(pos=(30., 12., 11.3), color=(230, 140, 210), intensity=.32, kind="sun")]       # idx 6: gulabi-jamni shaam ki roshni
    wk = _wallks(dims, L, a0); cr = _tones((235, 238, 242), (.97, 1.0, 1.03)); bl = _tones((200, 95, 40), (.95, 1.0, 1.05))
    surfs = [_surf(lambda u, v: (u, 0.0, v), x0, x1, 0, zm, .8, .8, False, 6, _tones((170, 120, 60), (.9, .96, 1.0, 1.05)), (200, 150, 90), 1, (0, 0, 7), (0, 1, 0), L, a0),
             _surf(lambda u, v: (u, Hh, v), x0, x1, 0, zm, .8, .8, False, 7, _tones((246, 246, 246), (.96, .98, 1.0)), (205, 200, 205), 1, (0, Hh, 7), (0, -1, 0), L, a0, boost=1.3)]
    for nm, pl, rng_, n, c_ in (("l", lambda u, y: (x0, y, u), (0, zm), (1, 0, 0), (x0, 1.5, 7)), ("r", lambda u, y: (x1, y, u), (0, zm), (-1, 0, 0), (x1, 1.5, 7)), ("b", lambda u, y: (u, y, zm), (x0, x1), (0, 0, -1), (0, 1.5, zm))):
        surfs.append(_surf(pl, rng_[0], rng_[1], 0, 1.0, 1.0, 1.0, False, 8, bl, (170, 70, 25), 1, (c_[0], .5, c_[2]), n, L, a0))
        surfs.append(_surf(pl, rng_[0], rng_[1], 1.0, Hh, 1.0, 1.0, False, 9, cr, (205, 208, 205), 1, c_, n, L, a0))
    B_, Lf, Rt = Wall("back", dims, wk["back"]), Wall("left", dims, wk["left"]), Wall("right", dims, wk["right"])
    win = [P3(x1, 1.0, 10.0), P3(x1, 1.0, 12.6), P3(x1, 2.5, 12.6), P3(x1, 2.5, 10.0)]

    def shell(r):
        _draw_surfs(r, surfs)
        for w_, (u0, u1) in ((B_, (x0, x1)), (Lf, (0, zm)), (Rt, (0, zm))):
            w_.fill(r, u0, u1, 0, .12, (110, 70, 40), 1); w_.fill(r, u0, u1, 1.0, 1.07, (170, 180, 80), 0, .01, True)
        _ceil_panels(r, [(0, z) for z in PZ], .5, .6, Hh)
        B_.door(r, -.54, .54, (205, 182, 150)); B_.tex(r, SIGN, -.36, .36, 1.03, 1.34, .03); B_.tex(r, EXIT, -.3, .3, 2.4, 2.65, .03)
        for u in ((3.2, 4.1), (7.2, 8.1), (11.0, 11.9)): Lf.door(r, *u)
        Rt.door(r, 2.6, 3.5); Lf.fill(r, 5.3, 5.5, 1.2, 1.5, (40, 40, 205), 3); Rt.fill(r, 4.2, 5.4, 1.2, 2.0, (70, 120, 170), 4); Rt.fill(r, 4.28, 5.32, 1.28, 1.92, (110, 170, 210), 1, .02)
        Rt.window(r, 10.0, 12.6, 1.0, 2.5, 5, ((240, 190, 230), (200, 120, 220)))
        for z in (5.6, 9.5): Lf.fill(r, z - .08, z + .08, 2.0, 2.25, (130, 225, 255), 3, .02, True)         # sconce
        r.ao_room()
    objs = [_bench(1.0, 1.6, 6.0, 7.3), _cooler(-1.38, 9.3), O((1.45, .5, 8.2), [], [lambda r: r.cyl(1.4, 8.2, 0, .5, .09, (50, 50, 215), 2)])]
    objs[1]["boxes"] = [(-1.56, -1.2, 0, .95, 9.12, 9.48)]
    fx = _panels([(0, z) for z in PZ], .5, .6, Hh) + _sun_fx(win, (-.9, -.35, .25), dims, (70, 45, 85), (80, 50, 95), 5.0) + _lampfx((-1.45, 2.12, 5.6), .4, (80, 160, 210)) + _lampfx((-1.45, 2.12, 9.5), .4, (80, 160, 210)) + [dict(t="glow", pos=(0, 2.5, zm - .1), rad=.5, col=(40, 130, 50))]
    return dict(name="corridor", dims=dims, lights=L, key=6, door=dict(pos=(0.0, zm)), amb0=a0, tone=(1.0, .99, 1.0), objs=objs, fx=fx, shell_draw=shell,
                slots=dict(left=_slot(-.75, 4.8), center=_slot(0, 4.4), right=_slot(.75, 4.8), far=_slot(0, 11.5), seat=_slot(1.15, 6.65)))


# ================================================================ BREAK ROOM
def _break():
    dims = (-3.2, 3.2, 8.0, 3.0); x0, x1, zm, Hh = dims; a0 = .74
    PAN = [(x, z) for x in (-1.2, 1.2) for z in (2.5, 5.5)]
    L = [dict(pos=(x, 2.95, z), color=(255, 250, 242), intensity=.14, kind="panel") for x, z in PAN]
    L += [dict(pos=(30., 12., 4.3), color=(70, 225, 255), intensity=.35, kind="sun"),         # idx 4
          dict(pos=(-2.45, 1.5, 3.5), color=(255, 230, 120), intensity=.35, kind="lamp"),      # vending machine ka cyan glow
          dict(pos=(-1.0, 1.45, 7.5), color=(110, 190, 255), intensity=.30, kind="lamp")]
    wk = _wallks(dims, L, a0); tile = _tones((228, 228, 215), (.98, 1.0)); mus = _tones((80, 185, 235), (.95, 1.0, 1.05))
    chk = [(228, 228, 215), (222, 222, 208), (165, 195, 150), (160, 188, 145)]
    surfs = [_surf(lambda u, v: (u, 0.0, v), x0, x1, 0, zm, .6, .6, False, 11, chk, (150, 170, 140), 1, (0, 0, 4), (0, 1, 0), L, a0, checker=True),
             _surf(lambda u, v: (u, Hh, v), x0, x1, 0, zm, .8, .8, False, 12, _tones((246, 246, 246), (.96, .98, 1.0)), (205, 200, 205), 1, (0, Hh, 4), (0, -1, 0), L, a0, boost=1.3)]
    for nm, pl, rng_, n, c_ in (("l", lambda u, y: (x0, y, u), (0, zm), (1, 0, 0), (x0, 1.5, 4)), ("r", lambda u, y: (x1, y, u), (0, zm), (-1, 0, 0), (x1, 1.5, 4)), ("b", lambda u, y: (u, y, zm), (x0, x1), (0, 0, -1), (0, 1.5, zm))):
        surfs.append(_surf(pl, rng_[0], rng_[1], 0, 1.2, .3, .15, True, 13, tile, (190, 190, 180), 1, (c_[0], .6, c_[2]), n, L, a0))
        surfs.append(_surf(pl, rng_[0], rng_[1], 1.2, Hh, 1.0, 1.8, False, 14, mus, (60, 150, 195), 1, c_, n, L, a0))
    B_, Lf, Rt = Wall("back", dims, wk["back"]), Wall("left", dims, wk["left"]), Wall("right", dims, wk["right"])
    win = [P3(x1, 1.0, 2.2), P3(x1, 1.0, 5.8), P3(x1, 2.5, 5.8), P3(x1, 2.5, 2.2)]

    def shell(r):
        _draw_surfs(r, surfs)
        for w_, (u0, u1) in ((B_, (x0, x1)), (Lf, (0, zm)), (Rt, (0, zm))): w_.fill(r, u0, u1, 0, .1, (90, 110, 130), 1)
        _ceil_panels(r, PAN, .55, .6, Hh); _vent(r, 0, 4.0, Hh)
        B_.door(r, 1.35, 2.45, (170, 130, 90)); B_.circ(r, -.5, 2.45, .3, (40, 40, 200), 3); B_.circ(r, -.5, 2.45, .25, (255, 255, 255), 2, .02); B_.line(r, -.5, 2.45, -.5, 2.65, OUT, 3, .03); B_.line(r, -.5, 2.45, -.38, 2.4, OUT, 3, .03)
        Lf.tex(r, GRAPH, 5.5, 6.2, 1.5, 2.4); Lf.tex(r, CAL, 6.6, 7.1, 1.6, 2.2)
        Rt.window(r, 2.2, 5.8, 1.0, 2.5, 7); Rt.fill(r, 6.6, 7.6, 1.3, 2.0, (70, 120, 170), 4); Rt.fill(r, 6.68, 7.52, 1.38, 1.92, (110, 170, 210), 1, .02)
        r.poly([P3(-1.4, .01, 3.9), P3(1.4, .01, 3.9), P3(1.4, .01, 5.9), P3(-1.4, .01, 5.9)], r.shade((140, 90, 190), (0, 0, 5), (0, 1, 0)), 2, tint=False)     # gol/chhota rug
        r.ao_room()
    wood = (120, 175, 225); cab = (70, 120, 190); objs = []
    ctr = [(-3.0, 0.6, 0, .9, 7.15, 7.95, cab), (-3.05, .65, .9, .95, 7.05, 7.97, (205, 208, 212)), (-3.0, .6, 1.55, 2.35, 7.6, 7.97, cab)]
    ctr += [(-3.0, -.6, 0, .9, 7.15, 7.95, cab), (-.6, .6, 0, .9, 7.15, 7.95, cab)]
    ctr = [(-3.0, .6, 0, .9, 7.15, 7.95, cab), (-3.05, .65, .9, .95, 7.05, 7.97, (205, 208, 212)), (-3.0, .6, 1.55, 2.35, 7.6, 7.97, cab),
           (-2.4, -1.6, .95, .97, 7.3, 7.8, (160, 160, 165), 1), (-.9, -.35, .95, 1.28, 7.35, 7.85, (215, 215, 220)), (.05, .55, .95, 1.45, 7.35, 7.85, (50, 50, 60))]
    
    def cex(r):
        for xx in (-2.0, -1.0, 0.0): r.segs(np.array([[xx, .05, 7.145]]), np.array([[xx, .88, 7.145]]), OUT, 2)
        r.segs(np.array([[-2.0, .97, 7.5], [-2.0, 1.2, 7.5]]), np.array([[-2.0, 1.2, 7.5], [-1.85, 1.2, 7.5]]), OUT, 3)
        r.poly([P3(-.85, 1.0, 7.34), P3(-.5, 1.0, 7.34), P3(-.5, 1.24, 7.34), P3(-.85, 1.24, 7.34)], (150, 190, 220), 1, tint=False); r.disc(P3(-.42, 1.12, 7.34), .02, (60, 60, 220), 1)
        r.cyl(.3, 7.6, .95, 1.2, .08, (60, 90, 160), 2); r.disc(P3(.3, 1.38, 7.33), .018, (60, 60, 230), 1)
    objs.append(O((-1.2, .6, 7.5), ctr, [cex]))
    objs.append(O((2.5, 1, 6.7), [(1.9, 3.1, 0, 1.95, 6.2, 7.2, (225, 225, 230)), (1.9, 1.95, .5, 1.3, 6.15, 6.2, (90, 90, 95), 1)],
                  [lambda r: [r.poly([P3(xx, y, 6.19), P3(xx + .1, y, 6.19), P3(xx + .1, y + .1, 6.19), P3(xx, y + .1, 6.19)], c, 1, tint=False) for (xx, y, c) in ((2.2, 1.5, (60, 60, 220)), (2.4, 1.2, (60, 200, 240)), (2.6, 1.6, (230, 130, 60)))]]))
    objs.append(O((-2.85, .95, 3.5), [(-3.2, -2.5, 0, 1.9, 3.0, 4.0, (60, 60, 200))],
                  [lambda r: [r.poly([P3(-2.495, y, z), P3(-2.495, y, z + .2), P3(-2.495, y + .14, z + .2), P3(-2.495, y + .14, z)], c, 1, tint=False)
                              for y in (1.2, 1.4, 1.6) for z, c in zip((3.1, 3.35, 3.6), ((60, 200, 240), (80, 90, 230), (230, 160, 60)))]]))
    top = [(-.9, .9, .72, .77, 4.3, 5.3, wood)] + [(x - .04, x + .04, 0, .72, z - .04, z + .04, (70, 100, 150), 1) for x in (-.8, .8) for z in (4.4, 5.2)]
    top += [(-.45, -.35, .77, .85, 4.7, 4.8, (60, 60, 220), 1), (.3, .6, .77, .82, 4.6, 4.9, (230, 120, 60), 1), (.0, .15, .77, .85, 5.0, 5.1, (250, 250, 250), 1)]
    objs.append(O((0, .4, 4.8), top))
    for x, z, c in ((-1.45, 4.8, (90, 130, 190)), (1.45, 4.8, (80, 150, 110)), (0, 5.95, (150, 110, 80))): objs.append(_chair(x, z, c))
    objs += [_plant(2.9, 5.0), O((-2.9, .3, 5.6), [], [lambda r: r.cyl(-2.9, 5.6, 0, .55, .17, (110, 110, 118), 2)], boxes=[(-3.07, -2.73, 0, .55, 5.43, 5.77)])]
    fx = _panels(PAN, .55, .6, Hh) + _sun_fx(win, (-.7, -.45, .3), dims, (40, 95, 120), (45, 105, 125)) + _lampfx((-2.4, 1.5, 3.5), .5, (160, 150, 60)) + _lampfx((-1.0, 1.4, 7.5), .35, (70, 140, 200), ((-1.0, 7.5), .6, .3, .96))
    return dict(name="break_room", dims=dims, lights=L, key=4, door=dict(pos=(1.9, zm)), amb0=a0, tone=(1.0, 1.0, 1.0), objs=objs, fx=fx, shell_draw=shell,
                slots=dict(left=_slot(-1.5, 3.0), center=_slot(0, 3.2), right=_slot(1.5, 3.0), far=_slot(.6, 6.4), seat=_slot(0, 5.7)))


# ================================================================ NIGHT EXTERIOR
def _wgrid(plane, u0, du, nu, v0, dv, nv, w, h, seed, probs):
    rng = np.random.RandomState(seed); Q = []; T = []
    for i in range(nv):
        for j in range(nu):
            u = u0 + j * du; v = v0 + i * dv
            Q.append([plane(u, v), plane(u + w, v), plane(u + w, v + h), plane(u, v + h)]); T.append(int(np.searchsorted(np.cumsum(probs), rng.rand())))
    return np.array(Q, float), np.array(T, int)


def _skyline(zp, seed, n, hmax, span=(-90, 90)):
    rng = np.random.RandomState(seed); Q = []; T = []; x = span[0]
    while x < span[1]:
        w = rng.uniform(6, 14); h = rng.uniform(.25, 1) * hmax; Q.append([P3(x, 0, zp), P3(x + w, 0, zp), P3(x + w, h, zp), P3(x, h, zp)]); T.append(0)
        for _ in range(int(w * h / 90)):
            wx = x + rng.uniform(.5, w - 1.3); wy = rng.uniform(1, h - 1.5)
            if rng.rand() < .5: Q.append([P3(wx, wy, zp - .1), P3(wx + .8, wy, zp - .1), P3(wx + .8, wy + 1.0, zp - .1), P3(wx, wy + 1.0, zp - .1)]); T.append(1)
        x += w + rng.uniform(0, 2)
    return np.array(Q, float), np.array(T, int)


def _night():
    dims = (-7.0, 7.0, 18.0, 14.0); x0, x1, zm, Hh = dims; a0 = .55
    L = [dict(pos=(-25., 30., -10.), color=(255, 210, 170), intensity=.30, kind="sun"),            # chaand (thandi roshni) idx 0
         dict(pos=(-4.5, 3.3, 7.0), color=(110, 195, 255), intensity=.60, kind="lamp"), dict(pos=(4.5, 3.3, 7.0), color=(110, 195, 255), intensity=.60, kind="lamp"),
         dict(pos=(0, 1.4, 17.6), color=(120, 210, 255), intensity=.50, kind="lamp")]
    g = lambda u, v: (u, 0.0, v)
    road = _surf(g, -14, 14, 0, 2.2, 1.0, 1.0, False, 21, _tones((60, 50, 48), (.9, 1.0, 1.1)), (45, 38, 36), 1, (0, 0, 1), (0, 1, 0), L, a0)
    walk = _surf(g, -14, 14, 2.2, 9.2, 1.0, 1.0, False, 22, _tones((130, 120, 115), (.92, 1.0, 1.08)), (95, 88, 85), 1, (0, 0, 6), (0, 1, 0), L, a0)
    plaza = _surf(g, -14, 14, 9.2, 18.0, 1.0, 1.0, False, 23, _tones((105, 95, 92), (.92, 1.0, 1.08)), (78, 70, 68), 1, (0, 0, 13), (0, 1, 0), L, a0)
    fac = _surf(lambda u, y: (u, y, zm), -8, 8, 0, Hh, 2.0, 2.0, False, 24, _tones((100, 62, 45), (.9, 1.0, 1.1)), (70, 42, 32), 1, (0, 6, zm), (0, 0, -1), L, a0)
    WP = [(70, 48, 38), (110, 215, 255), (255, 225, 170), (210, 150, 255)]
    WQ, WT = _wgrid(lambda u, y: (u, y, zm - .02), -7.2, 1.6, 9, 2.8, 2.0, 5, 1.0, 1.2, 31, (.28, .42, .2, .1))
    lit = [i for i, t in enumerate(WT) if t > 0]
    DQ = np.array([[WQ[i][0] + P3(.08, .05, -.01), WQ[i][1] - P3(.08, 0, 0) + P3(0, .05, -.01), WQ[i][1] - P3(.08, 0, 0) + P3(0, .22, -.01), WQ[i][0] + P3(.08, .22, -.01)] for i in lit], float)
    MQ = np.array([[WQ[i][0] + P3(.55, .3, -.012), WQ[i][0] + P3(.85, .3, -.012), WQ[i][0] + P3(.85, .55, -.012), WQ[i][0] + P3(.55, .55, -.012)] for i in lit], float)
    LQ = np.array([[P3(-3, .1, zm - .02), P3(3, .1, zm - .02), P3(3, 2.4, zm - .02), P3(-3, 2.4, zm - .02)]], float)
    SQ1, ST1 = _skyline(110, 41, 0, 34); SQ2, ST2 = _skyline(70, 42, 0, 24)
    LWQ, LWT = _wgrid(lambda u, y: (u, y, 9.98), -16.5, 2.0, 4, 2.0, 2.2, 4, .9, 1.2, 33, (.4, .35, .15, .1))
    RWQ, RWT = _wgrid(lambda u, y: (u, y, 11.98), 9.0, 2.0, 4, 2.0, 2.4, 5, .9, 1.2, 34, (.4, .35, .15, .1))
    stars = np.random.RandomState(7).uniform([-130, 50, 150], [130, 130, 150], (60, 3))
    P1 = lambda c: np.array([c], np.uint8)
    wglow = [dict(t="panel", quad=[WQ[i][0], WQ[i][1], WQ[i][2], WQ[i][3]], col=_cs(WP[WT[i]], .22)) for i in lit]

    def shell(r):
        r.sky([(-.5, (70, 50, 70)), (0.0, (120, 130, 245)), (.18, (150, 80, 190)), (.5, (110, 45, 90)), (1.0, (70, 28, 35)), (1.6, (45, 18, 18))])
        pc = r.c.cam(stars); ok = pc[:, 2] > .5
        if ok.any():
            for u, v in r.c.proj(pc[ok]):
                if 0 <= u < r.fr.shape[1] and 0 <= v < r.fr.shape[0]: r.fr[int(v), int(u)] = (235, 225, 215)
        r.disc(P3(-50, 75, 150), 7, (235, 240, 250), 0)
        r.poly([P3(-300, 0, 8), P3(300, 0, 8), P3(300, 0, 300), P3(-300, 0, 300)], (36, 28, 26), 0, tint=False)
        r.fill_quads(SQ1, ST1, [(70, 40, 40), (110, 200, 240)]); r.fill_quads(SQ2, ST2, [(52, 30, 32), (100, 190, 235)])
        r.box(-18, -8.2, 0, 11, 10, 30, (95, 62, 52), 2); r.box(8.2, 18, 0, 13, 12, 30, (85, 55, 48), 2)
        r.fill_quads(LWQ, LWT, WP); r.fill_quads(RWQ, RWT, WP)
        _draw_surfs(r, [road, walk, plaza, fac])
        r.fill_quads(WQ, WT, WP); r.fill_quads(DQ, np.zeros(len(DQ), int), [(50, 38, 34)]); r.fill_quads(MQ, np.zeros(len(MQ), int), [(255, 240, 190)])
        r.fill_quads(LQ, np.zeros(1, int), [(100, 200, 250)])
        r.poly([P3(-.7, 0, zm - .03), P3(.7, 0, zm - .03), P3(.7, 2.2, zm - .03), P3(-.7, 2.2, zm - .03)], (60, 50, 45), 3, tint=False)
        r.poly([P3(-.6, 0, zm - .04), P3(.6, 0, zm - .04), P3(.6, 2.1, zm - .04), P3(-.6, 2.1, zm - .04)], (140, 225, 255), 2, tint=False)
        r.segs(np.array([[0, 0, zm - .05]]), np.array([[0, 2.1, zm - .05]]), OUT, 3)
        r.box(-3.2, 3.2, 2.45, 2.62, 16.2, 18, (60, 50, 70), 3); r.tex(TOWER, [P3(-1.3, 3.9, zm - .05), P3(1.3, 3.9, zm - .05), P3(1.3, 2.9, zm - .05), P3(-1.3, 2.9, zm - .05)], .9)
        for z in (2.2, 9.2): r.segs(np.array([[x0 - 7, 0.0, z]]), np.array([[x1 + 7, 0.0, z]]), (160, 150, 140), 2)
        for x in range(-12, 13, 3): r.segs(np.array([[x, 0.01, 1.1]]), np.array([[x + 1.2, 0.01, 1.1]]), (200, 200, 180), 3)       # sadak ki lakeer
        r.ao_edge(P3(-8, 0, zm), P3(8, 0, zm), P3(0, 0, -1), (.8, .45, .2))
    objs = []
    for x in (-4.5, 4.5):
        objs.append(O((x, 1.6, 7), [(x - .05, x + .05, 0, 3.2, 6.95, 7.05, (60, 60, 68), 1), (x - .25, x + .25, 3.2, 3.32, 6.8, 7.2, (60, 60, 68), 1)], [lambda r, x=x: r.disc(P3(x, 3.15, 7.0), .13, (130, 225, 255), 2)]))
    objs.append(_bench(4.3, 4.9, 5.2, 6.6)); objs.append(O((-5.5, .3, 5), [(-5.7, -5.3, 0, .6, 4.8, 5.2, (90, 85, 95))]))
    fx = wglow + [dict(t="glow", pos=(-4.5, 3.15, 7.0), rad=.9, col=(80, 150, 210)), dict(t="glow", pos=(4.5, 3.15, 7.0), rad=.9, col=(80, 150, 210)), dict(t="glow", pos=(0, 1.2, zm - .3), rad=2.4, col=(70, 130, 170)),
                   dict(t="pool", c=(-4.5, 7.0), rx=1.8, rz=1.8, y=.012, col=(35, 65, 85)), dict(t="pool", c=(4.5, 7.0), rx=1.8, rz=1.8, y=.012, col=(35, 65, 85))]
    return dict(name="night_exterior", dims=dims, lights=L, key=0, door=dict(pos=(0.0, zm)), amb0=a0, tone=(1.06, .98, .92), objs=objs, fx=fx, shell_draw=shell,
                slots=dict(left=_slot(-2.2, 5.0), center=_slot(0, 5.4), right=_slot(2.2, 5.0), far=_slot(0, 11.5), seat=_slot(4.6, 5.9)))


ROOMS = {"office": _office(), "corridor": _corridor(), "break_room": _break(), "night_exterior": _night()}
