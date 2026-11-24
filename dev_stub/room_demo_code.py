"""Demo: code se 3D toon room + camera + characters. Koi API/key/model nahi. numpy + cv2 + PIL only."""
import math, os, subprocess, time
import numpy as np, cv2
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 540, 960, 30
OUT = (22, 18, 26)
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Poppins-Bold.ttf")
RW, RH, RD = 4.0, 3.2, 12.0
NEAR = 0.1


def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(t): t = clamp(t); return t * t * (3 - 2 * t)
def lerp(a, b, t): return a + (b - a) * t
def lerpv(a, b, t): return np.array([lerp(x, y, t) for x, y in zip(a, b)], float)


class Cam:
    def __init__(s, pos, yaw, pitch, roll=0.0, hfov=70):
        s.pos = np.array(pos, float); s.roll = roll
        s.f = (W / 2) / math.tan(math.radians(hfov) / 2)
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)
        fw = np.array([sy * cp, sp, cy * cp]); rt = np.array([cy, 0, -sy]); up = np.array([-sy * sp, cp, -cy * sp])
        s.M = np.stack([rt, up, fw])

    def cam(s, P): return (np.atleast_2d(np.array(P, float)) - s.pos) @ s.M.T

    def proj(s, pc):
        z = pc[:, 2:3]; x = W / 2 + s.f * pc[:, 0:1] / z; y = H / 2 - s.f * pc[:, 1:2] / z
        if s.roll:
            c, sn = math.cos(s.roll), math.sin(s.roll); dx, dy = x - W / 2, y - H / 2
            x = W / 2 + dx * c - dy * sn; y = H / 2 + dx * sn + dy * c
        return np.hstack([x, y])


def look_at(cp, tg):
    d = np.array(tg, float) - np.array(cp, float)
    return math.atan2(d[0], d[2]), math.atan2(d[1], math.hypot(d[0], d[2]))


def clip_poly(pc):
    out = []; n = len(pc)
    for i in range(n):
        a, b = pc[i], pc[(i + 1) % n]
        ai, bi = a[2] >= NEAR, b[2] >= NEAR
        if ai: out.append(a)
        if ai != bi:
            t = (NEAR - a[2]) / (b[2] - a[2]); out.append(a + (b - a) * t)
    return np.array(out)


class R:
    def __init__(s, fr, cam, fi, amb=(1, 1, 1)):
        s.fr, s.c, s.ph, s.amb = fr, cam, (fi // 3) * 1.9, np.array(amb, float)

    def col(s, c, k=1.0): return tuple(int(clamp(v * a * k, 0, 255)) for v, a in zip(c, s.amb))

    def jit(s, p, amp=1.3):   # line-boil: har 3 frame par outline halka hilta hai
        return p + amp * np.stack([np.sin(0.07 * p[:, 0] + 0.11 * p[:, 1] + s.ph), np.cos(0.09 * p[:, 0] - 0.06 * p[:, 1] + s.ph * 1.3)], 1)

    def pts(s, P, amp=1.3):
        pc = clip_poly(s.c.cam(P))
        if len(pc) < 3: return None
        p = np.clip(s.jit(s.c.proj(pc), amp), -20000, 20000)
        return np.round(p).astype(np.int32)

    def poly(s, P, fill, lw=3, k=1.0, tint=True):
        p = s.pts(P)
        if p is None: return
        if fill is not None: cv2.fillPoly(s.fr, [p], s.col(fill, k) if tint else fill, cv2.LINE_AA)
        if lw: cv2.polylines(s.fr, [p], True, OUT, lw, cv2.LINE_AA)

    def lines(s, A, B, color, lw=1):
        a = s.c.cam(A); b = s.c.cam(B); za, zb = a[:, 2].copy(), b[:, 2].copy()
        keep = (za >= NEAR) | (zb >= NEAR)
        a, b, za, zb = a[keep], b[keep], za[keep], zb[keep]
        m = za < NEAR
        if m.any(): t = (NEAR - za[m]) / (zb[m] - za[m]); a[m] = a[m] + (b[m] - a[m]) * t[:, None]
        m = zb < NEAR
        if m.any(): t = (NEAR - zb[m]) / (za[m] - zb[m]); b[m] = b[m] + (a[m] - b[m]) * t[:, None]
        pa = np.round(np.clip(s.jit(s.c.proj(a), 0.9), -20000, 20000)).astype(int)
        pb = np.round(np.clip(s.jit(s.c.proj(b), 0.9), -20000, 20000)).astype(int)
        for (x0, y0), (x1, y1) in zip(pa, pb): cv2.line(s.fr, (x0, y0), (x1, y1), color, lw, cv2.LINE_AA)

    def tex(s, img, P):
        pc = s.c.cam(P)
        if (pc[:, 2] < NEAR).any(): return
        dst = s.jit(s.c.proj(pc), 0.8).astype(np.float32); h, w = img.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst)
        wp = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR)
        mk = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (W, H)) > 127
        s.fr[mk] = (wp[mk] * s.amb).clip(0, 255).astype(np.uint8)

    def patch(s, P, color, alpha):
        p = s.pts(P, 0)
        if p is None: return
        ov = s.fr.copy(); cv2.fillPoly(ov, [p], color, cv2.LINE_AA); s.fr[:] = cv2.addWeighted(ov, alpha, s.fr, 1 - alpha, 0)

    def box(s, x0, x1, y0, y1, z0, z1, color, lw=3):
        V = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
        F = [((0, 1, 3, 2), (-1, 0, 0), .78), ((4, 6, 7, 5), (1, 0, 0), .92), ((0, 4, 5, 1), (0, -1, 0), .6),
             ((2, 3, 7, 6), (0, 1, 0), 1.12), ((0, 2, 6, 4), (0, 0, -1), .85), ((1, 5, 7, 3), (0, 0, 1), .95)]
        for idx, n, k in F:
            q = V[list(idx)]
            if np.dot(n, s.c.pos - q.mean(0)) > 0: s.poly(q, color, lw, k)


def txt_tex(lines, size, colors, bg, wh):
    im = Image.new("RGB", wh, bg); d = ImageDraw.Draw(im)
    y = 8
    for ln, sz, c in zip(lines, size, colors):
        f = ImageFont.truetype(FONT, sz); w = d.textlength(ln, font=f)
        d.text(((wh[0] - w) / 2, y), ln, font=f, fill=c); y += sz + 6
    d.rectangle([0, 0, wh[0] - 1, wh[1] - 1], outline=(200, 120, 40), width=6)
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


SIGN = txt_tex(["CAREER", "DEVELOPMENT", "ROOM"], [30, 30, 40], [(150, 70, 20), (150, 70, 20), (40, 40, 220)], (255, 255, 255), (380, 150))


def bricks():
    A, B = [], []; bh, bw = .28, .56; n = int(RH / bh)
    for wall in ("back", "left", "right"):
        a0, a1 = (-RW, RW) if wall == "back" else (0, RD)
        P = (lambda u, y: (u, y, RD)) if wall == "back" else (lambda u, y: (-RW, y, u)) if wall == "left" else (lambda u, y: (RW, y, u))
        for k in range(1, n + 1):
            A.append(P(a0, min(k * bh, RH))); B.append(P(a1, min(k * bh, RH)))
        for row in range(n):
            u = a0 + (bw / 2 if row % 2 else 0) + bw
            while u < a1:
                A.append(P(u, row * bh)); B.append(P(u, min((row + 1) * bh, RH))); u += bw
    return np.array(A), np.array(B)


def grid(plane, step, a0, a1, b0, b1, c):
    A, B = [], []
    for u in np.arange(a0, a1 + 1e-6, step):
        A.append(plane(u, b0, c)); B.append(plane(u, b1, c))
    for v in np.arange(b0, b1 + 1e-6, step):
        A.append(plane(a0, v, c)); B.append(plane(a1, v, c))
    return np.array(A), np.array(B)


BR_A, BR_B = bricks()
FL_A, FL_B = grid(lambda u, v, c: (u, c, v), 1.0, -RW, RW, 0, RD, 0.0)
CE_A, CE_B = grid(lambda u, v, c: (u, c, v), 0.8, -RW, RW, 0, RD, RH)
BOOKS = [((-3.62, y, z), c) for y in (0.55, 1.15, 1.75) for z, c in
         zip(np.arange(2.7, 4.3, 0.2), [(60, 60, 200), (200, 120, 40), (60, 170, 220), (90, 170, 80), (180, 80, 160), (60, 60, 200), (200, 160, 50), (80, 120, 220)])]


def room(r, night):
    """Fixed room design -> har video mein bilkul same duniya."""
    wall = (215, 175, 140) if night else (125, 185, 135)
    wline = (185, 140, 110) if night else (92, 148, 100)
    floor = (170, 70, 28) if night else (205, 62, 22)
    fline = (190, 95, 50) if night else (230, 115, 62)
    ceil = (190, 165, 150) if night else (246, 246, 246)
    P = lambda *a: np.array(a, float)
    r.poly([P(-RW, 0, 0), P(RW, 0, 0), P(RW, 0, RD), P(-RW, 0, RD)], floor, 0, tint=False)
    r.poly([P(-RW, RH, 0), P(RW, RH, 0), P(RW, RH, RD), P(-RW, RH, RD)], ceil, 0, tint=False)
    r.poly([P(-RW, 0, 0), P(-RW, 0, RD), P(-RW, RH, RD), P(-RW, RH, 0)], wall, 0, .88, tint=False)
    r.poly([P(RW, 0, 0), P(RW, 0, RD), P(RW, RH, RD), P(RW, RH, 0)], wall, 0, .94, tint=False)
    r.poly([P(-RW, 0, RD), P(RW, 0, RD), P(RW, RH, RD), P(-RW, RH, RD)], wall, 0, 1.0, tint=False)
    r.lines(FL_A, FL_B, fline, 1); r.lines(CE_A, CE_B, (205, 200, 205) if not night else (150, 125, 110), 1)
    if not night: r.lines(BR_A, BR_B, wline, 1)
    else:   # neeche neela paint + teal patti
        for wl in ((-RW, "x"), (RW, "x")):
            r.poly([P(wl[0], 0, 0), P(wl[0], 0, RD), P(wl[0], .95, RD), P(wl[0], .95, 0)], (200, 95, 40), 0, tint=False)
            r.poly([P(wl[0], .95, 0), P(wl[0], .95, RD), P(wl[0], 1.0, RD), P(wl[0], 1.0, 0)], (170, 180, 80), 0, tint=False)
        r.poly([P(-RW, 0, RD), P(RW, 0, RD), P(RW, .95, RD), P(-RW, .95, RD)], (200, 95, 40), 0, tint=False)
        r.poly([P(-RW, .95, RD), P(RW, .95, RD), P(RW, 1.0, RD), P(-RW, 1.0, RD)], (170, 180, 80), 0, tint=False)
    for z0 in (2.0, 5.5, 9.0):  # chhat ki lights
        r.poly([P(-.6, RH - .01, z0), P(.6, RH - .01, z0), P(.6, RH - .01, z0 + 1.2), P(-.6, RH - .01, z0 + 1.2)], (255, 255, 255), 2, tint=False)
    # darwaza + sign (back wall)
    z = RD - .01
    r.poly([P(-.62, 0, z), P(.62, 0, z), P(.62, 2.3, z), P(-.62, 2.3, z)], (92, 92, 105), 3)
    r.poly([P(-.54, 0, z - .01), P(.54, 0, z - .01), P(.54, 2.22, z - .01), P(-.54, 2.22, z - .01)], (205, 182, 150), 2)
    r.poly([P(-.2, 1.45, z - .02), P(.2, 1.45, z - .02), P(.2, 2.0, z - .02), P(-.2, 2.0, z - .02)], (170, 195, 110), 3)
    r.poly([P(.38, .98, z - .02), P(.5, .98, z - .02), P(.5, 1.04, z - .02), P(.38, 1.04, z - .02)], (60, 190, 230), 2)
    r.tex(SIGN, [P(-.36, 1.34, z - .03), P(.36, 1.34, z - .03), P(.36, 1.03, z - .03), P(-.36, 1.03, z - .03)])
    r.poly([P(1.0, 1.2, z), P(1.18, 1.2, z), P(1.18, 1.45, z), P(1.0, 1.45, z)], (40, 40, 205), 3)
    if not night:
        # gol ghadi
        cx, cy, rad = -2.2, 2.45, .34
        ring = [P(cx + rad * math.cos(a), cy + rad * math.sin(a), z) for a in np.linspace(0, 2 * math.pi, 28, endpoint=False)]
        r.poly([P(x, y, z - .02) for x, y, _ in [(q[0], q[1], 0) for q in [P(cx + (rad + .05) * math.cos(a), cy + (rad + .05) * math.sin(a), 0) for a in np.linspace(0, 2 * math.pi, 28, endpoint=False)]]], (40, 40, 200), 3)
        r.poly(ring, (255, 255, 255), 2)
        r.lines(np.array([P(cx, cy, z - .03), P(cx, cy, z - .03)]), np.array([P(cx + .02, cy + .23, z - .03), P(cx + .17, cy - .1, z - .03)]), OUT, 3)
        # khidki (right wall)
        x = RW - .01
        r.poly([P(x, 1.0, 2.5), P(x, 1.0, 8), P(x, 2.7, 8), P(x, 2.7, 2.5)], (235, 205, 155), 4)
        r.lines(np.array([P(x, 1.0, zz) for zz in (3.6, 4.7, 5.8, 6.9)] + [P(x, 1.85, 2.5)]),
                np.array([P(x, 2.7, zz) for zz in (3.6, 4.7, 5.8, 6.9)] + [P(x, 1.85, 8)]), OUT, 2)
        # dhoop ke patches
        r.patch([P(-RW, 1.0, RD), P(-RW, 2.1, RD), P(1.6, 3.2, RD), P(1.6, 2.1, RD)], (70, 225, 255), .42)
        r.patch([P(-RW, .6, 3), P(-RW, 2.4, 3), P(-RW, 2.4, 9), P(-RW, .6, 9)], (70, 225, 255), .16)
    else:
        r.poly([P(-1.45, 2.0, z), P(-1.15, 2.0, z), P(-1.2, 2.25, z - .05), P(-1.4, 2.25, z - .05)], (130, 225, 255), 3, tint=False)


def room_objs(night):
    """Furniture list: (centre, draw-fn). Distance se sort hota hai."""
    P = lambda *a: np.array(a, float)
    O = []
    if not night:
        desk, side = (200, 215, 235), (170, 190, 215)
        O.append((P(0, .4, 6), lambda r: (r.box(-.9, -.84, 0, .72, 5.6, 6.4, side), r.box(.84, .9, 0, .72, 5.6, 6.4, side),
                                          r.box(-.84, .84, .22, .72, 5.6, 5.65, side), r.box(-.9, .9, .72, .78, 5.6, 6.4, desk))))
        O.append((P(.55, 1.0, 6.2), lambda r: (r.box(.4, .7, .78, .8, 6.1, 6.3, (90, 90, 95)), r.box(.3, .8, .98, 1.3, 6.2, 6.26, (70, 60, 60)),
                                               r.poly([P(.33, 1.0, 6.19), P(.77, 1.0, 6.19), P(.77, 1.27, 6.19), P(.33, 1.27, 6.19)], (230, 190, 110), 2))))
        O.append((P(0, .9, 6.87), lambda r: r.box(-.3, .3, .5, 1.25, 6.82, 6.9, (95, 95, 100))))
        O.append((P(-3.8, 1.2, 3.5), lambda r: (r.box(-RW, -3.6, 0, 2.3, 2.5, 4.5, (70, 90, 130)),
                                                 [r.poly([P(x, y, z), P(x, y, z + .14), P(x, y + .5, z + .14), P(x, y + .5, z)], c, 1) for (x, y, z), c in BOOKS])))
        O.append((P(3.4, .3, 4), lambda r: (r.box(2.9, 3.8, 0, .55, 3.6, 4.5, (120, 165, 205)), r.box(3.0, 3.6, .55, .95, 3.7, 4.3, (135, 180, 218)))))
    else:
        O.append((P(-3.8, 1.4, 9), lambda r: (r.box(-RW, -3.55, .8, 2.1, 8.4, 9.6, (112, 102, 96)),
                                               [r.poly([P(-3.55, y, z), P(-3.55, y, z + .26), P(-3.55, y + .2, z + .26), P(-3.55, y + .2, z)], (90, 200, 240), 2)
                                                for y in (1.0, 1.35, 1.7) for z in (8.5, 8.85, 9.2)])))
        O.append((P(3.5, .4, 8), lambda r: r.box(3.1, 3.9, 0, .8, 7.6, 8.6, (120, 165, 205))))
    return O


def draw_char(r, Pw, ch, st):
    c = r.c; pc = c.cam(Pw)
    if pc[0, 2] < .35: return
    u, v = c.proj(pc)[0]; sc = c.f / pc[0, 2]; t = st["t"]
    u += 1.2 * math.sin(r.ph * 2.1); v += 1.2 * math.cos(r.ph * 1.7)
    sit = st.get("sit", False); wk = st.get("walk", False); ph = st.get("ph", 0.0)
    bob = .015 * abs(math.sin(ph)) if wk else .008 * math.sin(t * 3.1)
    base = -.25 if sit else 0.0
    I = lambda x: int(round(x)); col = r.col
    X = lambda lx, ly: (I(u + lx * sc), I(v - (ly + bob + base) * sc))
    L = lambda m: max(1, I(m * sc)); ow = L(.022)

    def ell(lx, ly, ax, ay, fill, w=ow):
        p, ax_, ay_ = X(lx, ly), (max(1, I(ax * sc)), max(1, I(ay * sc)))[0], max(1, I(ay * sc))
        cv2.ellipse(r.fr, p, (ax_, ay_), 0, 0, 360, col(fill), -1, cv2.LINE_AA)
        if w: cv2.ellipse(r.fr, p, (ax_, ay_), 0, 0, 360, OUT, w, cv2.LINE_AA)

    def seg(a, b, w): cv2.line(r.fr, X(*a), X(*b), OUT, w, cv2.LINE_AA)

    def pg(P_, fill, w):
        p = np.array([X(*q) for q in P_], np.int32)
        if fill is not None: cv2.fillPoly(r.fr, [p], col(fill), cv2.LINE_AA)
        if w: cv2.polylines(r.fr, [p], True, OUT, w, cv2.LINE_AA)

    if not sit:
        for sd in (-1, 1):
            q = ph + (math.pi if sd > 0 else 0); lift = max(0, math.sin(q)) * .10 if wk else 0
            hip, knee, foot = (sd * .07, .80), (sd * .09, .42 + lift * .5), (sd * .11, lift)
            seg(hip, knee, L(.032)); seg(knee, foot, L(.032)); ell(sd * .135, lift + .02, .075, .035, (28, 25, 30), 0)
    ell(0, .98, .20, .26, ch["shirt"])
    if ch.get("tie"):
        pg([(-.055, 1.22), (.055, 1.22), (0, 1.0)], (255, 255, 255), L(.01))
        pg([(-.025, 1.21), (.025, 1.21), (.035, 1.08), (0, .86), (-.035, 1.08)], (40, 40, 210), L(.01))
    arms = st.get("arms", {"l": (-.30, .80), "r": (.30, .80)})
    for sd, k in ((-1, "l"), (1, "r")):
        sh = (sd * .18, 1.15); hd = arms[k]; el = ((sh[0] + hd[0]) / 2 + sd * .05, (sh[1] + hd[1]) / 2 - .02)
        seg(sh, el, L(.03)); seg(el, hd, L(.03))
        cv2.circle(r.fr, X(*hd), L(.04), col(ch["skin"]), -1, cv2.LINE_AA); cv2.circle(r.fr, X(*hd), L(.04), OUT, L(.012), cv2.LINE_AA)
    if st.get("key"):
        h = arms["r"]; cv2.circle(r.fr, X(h[0], h[1] + .05), L(.03), (130, 200, 240), L(.01), cv2.LINE_AA)
        pg([(h[0] - .035, h[1] - .01), (h[0] + .035, h[1] - .01), (h[0] + .035, h[1] - .17), (h[0] - .035, h[1] - .17)], (50, 50, 215), L(.012))
    if st.get("phone"):
        h = arms["r"]; pg([(h[0] - .04, h[1] + .07), (h[0] + .04, h[1] + .07), (h[0] + .04, h[1] - .06), (h[0] - .04, h[1] - .06)], (50, 45, 48), L(.012))
        pg([(h[0] - .03, h[1] + .06), (h[0] + .03, h[1] + .06), (h[0] + .03, h[1] - .04), (h[0] - .03, h[1] - .04)], (235, 200, 120), 0)
    cy_ = 1.45; ell(0, cy_, .215, .25, ch["skin"])
    # baal
    A_, B_ = .215, .25; fy = ch.get("fy", .4); outer, inner = [], []
    for a in np.linspace(math.pi * .96, math.pi * .04, 16): outer.append((1.05 * A_ * math.cos(a), cy_ + 1.07 * B_ * math.sin(a)))
    for a in np.linspace(math.pi * .04, math.pi * .96, 16): inner.append((.98 * A_ * math.cos(a), cy_ + B_ * (fy + .1 * math.cos(5 * a) * ch.get("wave", 1)) * 1.0))
    pg(outer + inner, ch["hair"], ow)
    ex, lk, talk = st.get("expr", "smile"), st.get("look", (0, 0)), st.get("talk", 0.0)
    blink = (t % 3.3) < .12; ey = 1.47
    for sd in (-1, 1):
        cx = sd * .078
        if ex == "shock":
            ell(cx, ey, .058, .062, (255, 255, 255), L(.014)); cv2.circle(r.fr, X(cx + lk[0] * .02, ey + lk[1] * .02), max(1, L(.014)), OUT, -1, cv2.LINE_AA)
        elif ex == "happy":
            cv2.ellipse(r.fr, X(cx, ey - .012), (L(.04), L(.035)), 0, 200, 340, OUT, L(.014), cv2.LINE_AA)
        elif blink: seg((cx - .03, ey), (cx + .03, ey), L(.012))
        else: cv2.circle(r.fr, X(cx + lk[0] * .015, ey + lk[1] * .012), max(2, L(.026)), OUT, -1, cv2.LINE_AA)
    if ch.get("glasses"):
        for sd in (-1, 1): cv2.circle(r.fr, X(sd * .078, ey), L(.062), OUT, L(.01), cv2.LINE_AA)
        seg((-.016, ey), (.016, ey), L(.01))
    my = 1.355
    if ex in ("grin", "shock"):
        y0, y1 = my + .035, my - .035 - .04 * talk
        cv2.rectangle(r.fr, X(-.10, y0), X(.10, y1), (255, 255, 255), -1, cv2.LINE_AA); cv2.rectangle(r.fr, X(-.10, y0), X(.10, y1), OUT, L(.012), cv2.LINE_AA)
        for x in (-.06, -.02, .02, .06): seg((x, y0), (x, y1), max(1, L(.006)))
        seg((-.10, (y0 + y1) / 2), (.10, (y0 + y1) / 2), max(1, L(.006)))
    elif talk > .05:
        ell(0, my, .05 + .02 * talk, .008 + .045 * talk, (40, 25, 70), L(.012))
    else:
        cv2.ellipse(r.fr, X(0, my + .03), (L(.07), L(.045)), 0, 25, 155, OUT, L(.014), cv2.LINE_AA)


BOSS = dict(skin=(170, 215, 250), shirt=(72, 36, 26), hair=(178, 172, 168), fy=.55, tie=True)
GUY = dict(skin=(140, 190, 238), shirt=(150, 170, 40), hair=(38, 58, 100), fy=.42, wave=1.6)
STAFF = dict(skin=(170, 215, 250), shirt=(235, 170, 90), hair=(30, 28, 34), fy=.2, glasses=True, wave=0)

CAPS = [(.2, .7, "Imagine"), (.7, 1.1, "a"), (1.1, 1.8, "normal"), (1.8, 3.0, "Monday."), (3.4, 4.2, "Everything"), (4.2, 4.8, "looks"),
        (4.8, 5.9, "fine."), (6.6, 7.0, "Then"), (7.0, 7.3, "the"), (7.3, 7.8, "boss"), (7.8, 8.6, "calls"), (8.6, 9.5, "you."),
        (9.9, 10.5, "Don't"), (10.5, 11.0, "ask"), (11.0, 11.9, "why.")]
FNT = ImageFont.truetype(FONT, 60)


def captions(fr, t):
    w = next((c[2] for c in CAPS if c[0] <= t < c[1]), None)
    if not w: return fr
    im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(im)
    tw = d.textlength(w, font=FNT); d.text(((W - tw) / 2, H * .78), w, font=FNT, fill=(255, 255, 255), stroke_width=6, stroke_fill=(0, 0, 0))
    return cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)


def vignette():
    yy, xx = np.mgrid[0:H, 0:W]; d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return (1 - .28 * np.clip(d - .55, 0, 1) ** 1.3)[..., None].astype(np.float32)


VIG = vignette()
SHOTS = [3.2, 3.2, 3.2, 2.4]


def render(fi):
    t = fi / FPS; acc = 0
    for si, d in enumerate(SHOTS):
        if t < acc + d or si == len(SHOTS) - 1: break
        acc += d
    lt = t - acc; u = lt / SHOTS[si]; night = si >= 2
    amb = (1.0, 1.0, 1.0)
    P = lambda *a: np.array(a, float)
    chars = []
    if si == 0:
        pos = lerpv((.5, 1.55, .2), (.15, 1.5, 3.7), ease(u)); yaw, pit = lerp(.07, -.01, u), -.04
        cam = Cam(pos, yaw, pit, 0, 78)
        cx = lerp(1.9, .95, ease(u)); cz = lerp(2.4, 5.0, ease(u)); ph = t * 8; sw = math.sin(ph) * .08
        chars.append((P(cx, 0, cz), STAFF, dict(t=t, walk=True, ph=ph, expr="smile", look=(-1, 0), arms={"l": (-.30, .80 + sw), "r": (.30, .80 - sw)})))
        chars.append((P(-.2, 0, 6.6), GUY, dict(t=t, sit=True, expr="smile", look=(math.sin(t * 2), 0), phone=True, arms={"l": (-.22, 1.1), "r": (.14, 1.2 + .02 * math.sin(t * 4))})))
    elif si == 1:
        T = P(-.2, 1.2, 6.6); a = lerp(-.45, .35, ease(u)); cp = P(T[0] + 2.6 * math.sin(a), 1.62 + .08 * u, T[2] - 2.6 * math.cos(a))
        yaw, pit = look_at(cp, T + P(0, .05, 0)); cam = Cam(cp, yaw, pit, 0, 58)
        chars.append((P(.9, 0, 5.3), STAFF, dict(t=t, expr="smile", look=(-1, 0), arms={"l": (-.30, .80), "r": (.34, 1.0 + .12 * math.sin(t * 6))})))
        chars.append((P(-.2, 0, 6.6), GUY, dict(t=t, sit=True, expr="happy" if u < .5 else "smile", look=(math.sin(t * 2), 0), phone=True, arms={"l": (-.22, 1.1), "r": (.14, 1.2 + .02 * math.sin(t * 4))})))
    elif si == 2:
        cp = lerpv((-1.0, 1.5, 6.2), (-.35, 1.42, 8.5), ease(u)); yaw, pit = look_at(cp, P(0, 1.35, 10.5)); cam = Cam(cp, yaw, pit, lerp(.05, -.035, u), 66)
        chars.append((P(.1, 0, 10.6), BOSS, dict(t=t, expr="grin", look=(0, 0), arms={"l": (-.30, .80), "r": (.34, 1.08 + .1 * math.sin(t * 5))}, key=True)))
    else:
        cp = lerpv((.35, 1.52, 9.2), (.12, 1.5, 9.75), ease(u)) + P(.01 * math.sin(t * 31), .008 * math.sin(t * 27), 0)
        yaw, pit = look_at(cp, P(0, 1.5, 10.6)); cam = Cam(cp, yaw, pit, .012 * math.sin(t * 23), 46)
        chars.append((P(.1, 0, 10.6), BOSS, dict(t=t, expr="shock", look=(math.sin(t * 5), 0), talk=abs(math.sin(t * 13)) * .9, arms={"l": (-.30, .80), "r": (.34, 1.1)}, key=True)))
    fr = np.zeros((H, W, 3), np.uint8); r = R(fr, cam, fi, amb)
    room(r, night)
    objs = [(c, fn) for c, fn in room_objs(night)] + [(p, (lambda rr, p=p, ch=ch, st=st: draw_char(rr, p, ch, st))) for p, ch, st in chars]
    for c, fn in sorted(objs, key=lambda o: -cam.cam(o[0])[0, 2]): fn(r)
    f = fr.astype(np.float32)
    if night:
        f *= np.array([1.02, .96, .90], np.float32)
        yy, xx = np.mgrid[0:H, 0:W]; pc = cam.cam(P(-1.3, 2.1, RD - .1))
        if pc[0, 2] > .2:
            lu, lv = cam.proj(pc)[0]; g = np.exp(-(((xx - lu) ** 2 + (yy - lv) ** 2) / (2 * (cam.f * .45 / pc[0, 2]) ** 2)))
            f += g[..., None] * np.array([40, 110, 170], np.float32) * 1.0
    fr = np.clip(f * VIG, 0, 255).astype(np.uint8)
    return captions(fr, t)


if __name__ == "__main__":
    os.makedirs("/home/claude/demo/frames", exist_ok=True)
    n = int(sum(SHOTS) * FPS); vw = cv2.VideoWriter("/home/claude/demo/raw.mp4", cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H)); t0 = time.time()
    keys = {int(x * FPS) for x in (.5, 1.5, 2.6, 3.8, 4.6, 5.6, 7.0, 8.2, 10.0, 11.2)}
    for i in range(n):
        f = render(i); vw.write(f)
        if i in keys: cv2.imwrite(f"/home/claude/demo/frames/k_{i:03d}.png", f)
        if i == 10: print("10 frames:", round(time.time() - t0, 1), "s", flush=True)
    vw.release(); print("done", round(time.time() - t0, 1), "s")
    subprocess.run("ffmpeg -v error -y -i /home/claude/demo/raw.mp4 -c:v libx264 -pix_fmt yuv420p -crf 20 -movflags +faststart /mnt/user-data/outputs/code_room_demo.mp4", shell=True)
