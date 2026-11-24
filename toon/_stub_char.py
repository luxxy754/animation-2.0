"""toon/_stub_char.py — PART B ka toon/characters.py na ho to ye prototype ka draw_char stub chalega."""
import math
import numpy as np, cv2
from .core import OUT


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
INTERN = dict(skin=(150, 200, 245), shirt=(60, 150, 235), hair=(40, 40, 90), fy=.35, wave=1.2)
HR = dict(skin=(170, 215, 250), shirt=(170, 90, 200), hair=(60, 40, 30), fy=.5, wave=0.8, glasses=True)
CHARS = dict(boss=BOSS, guy=GUY, staff=STAFF, intern=INTERN, hr=HR)

