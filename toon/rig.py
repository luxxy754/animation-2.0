"""Character rig + draw_char (PART B). 2-tone cel shading, rim light, light-tint, boil outline, chalna/ishare/blink/expressions/lipsync.
Sirf numpy + cv2 chahiye. Renderer r se ye cheezein padhta hai: r.c (Cam), r.fr (frame), r.col(c,k), r.lights (optional), r.amb."""
import math

import cv2
import numpy as np

OUT = (22, 18, 26)                 # mote kaale outline ka rang (BGR)
SHADE = (0.80, 0.66, 0.64)         # saaye ka guna (BGR): thoda thanda-jamunai flat band
EXPRS = ("smile", "grin", "shock", "happy", "sad")


def _cl(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def light_for(r, Pw):
    """Character ke sabse qareebi (sabse taqatwar) light ki disha camera-space mein + rang. d = screen-disha (x right, y up)."""
    best = None
    for L in (getattr(r, "lights", None) or []):
        p = np.array(L["pos"], float)
        d = float(np.linalg.norm(p - Pw))
        w = float(L.get("intensity", 1.0)) * (0.55 if L.get("kind") == "sun" else 1.0 / (1.0 + 0.12 * d * d))
        if best is None or w > best[0]:
            best = (w, L, p)
    if best is None:
        return dict(d=np.array([0.45, 0.8]), k=0.0, color=(255, 255, 255))
    w, L, p = best
    c = r.c
    v = c.cam(p)[0] - c.cam(Pw + np.array([0, 1.1, 0.0]))[0]
    d = np.array([v[0], v[1]], float)
    m = float(np.linalg.norm(d))
    d = np.array([0.45, 0.8]) if m < 0.3 * max(1e-6, float(np.linalg.norm(v))) or m < 1e-6 else d / m   # light seedhi aage/peeche ho to upar-daayen se
    return dict(d=d, k=_cl(w * 0.5, 0, 0.45), color=tuple(L.get("color", (255, 255, 255))))


def gesture(kind, t, talk=0.0):
    """Haath ki pose (character-local x, y). kind: idle, talk, wave, point, shrug, hands_up, cheer, sad, type."""
    s = math.sin
    if kind == "wave":
        return {"l": (-.30, .80), "r": (.38 + .05 * s(t * 9), 1.42 + .05 * s(t * 9 + 1))}
    if kind == "point":
        return {"l": (-.30, .80), "r": (.62, 1.22)}
    if kind == "shrug":
        return {"l": (-.42, 1.05 + .03 * s(t * 3)), "r": (.42, 1.05 + .03 * s(t * 3))}
    if kind == "hands_up":
        return {"l": (-.38 + .01 * s(t * 40), 1.50), "r": (.38 - .01 * s(t * 40), 1.50)}
    if kind == "cheer":
        b = .05 * abs(s(t * 6))
        return {"l": (-.36, 1.55 + b), "r": (.36, 1.55 + b)}
    if kind == "sad":
        return {"l": (-.22, .66), "r": (.22, .66)}
    if kind == "type":
        return {"l": (-.18 + .02 * s(t * 17), 1.02), "r": (.18 - .02 * s(t * 15), 1.02)}
    if kind == "talk":
        return {"l": (-.30, .82 + .03 * s(t * 3.1)), "r": (.34 + .07 * s(t * 6.3) * talk, .98 + .14 * s(t * 4.7) * talk)}
    return {"l": (-.30, .80 + .01 * s(t * 1.7)), "r": (.30, .80 - .01 * s(t * 1.7))}


def auto_arms(st, ex, talk):
    """st mein arms na ho to expr/talk/sit se khud pose chunta hai."""
    t = st.get("t", 0.0)
    if st.get("sit"):
        return gesture("type", t) if talk < .05 else gesture("talk", t, talk)
    if ex == "shock":
        return gesture("hands_up", t)
    if ex == "sad":
        return gesture("sad", t) if talk < .05 else gesture("talk", t, talk * .6)
    if ex == "happy":
        return gesture("cheer", t)
    return gesture("talk", t, talk) if talk > .05 else gesture("idle", t)


def draw_char(r, Pw, ch, st):
    c = r.c
    Pw = np.asarray(Pw, float)
    pc = c.cam(Pw)
    if pc[0, 2] < .35:
        return
    fr = r.fr
    Hf, Wf = fr.shape[:2]
    u, v = c.proj(pc)[0]
    sc = c.f / pc[0, 2]
    t = float(st.get("t", 0.0))
    PH = int(t * 10) * 1.9 + ch.get("seed", 0) * .7                 # outline boil: har ~3 frame par badalta hai
    pk = Wf / 540.0
    u += 1.2 * pk * math.sin(PH * 2.1)
    v += 1.2 * pk * math.cos(PH * 1.7)
    sit, wk, ph = bool(st.get("sit")), bool(st.get("walk")), float(st.get("ph", 0.0))
    ex, lk, talk = st.get("expr", "smile"), st.get("look", (0, 0)), _cl(float(st.get("talk", 0.0)))
    if ex not in EXPRS:
        ex = "smile"
    bob = .015 * abs(math.sin(ph)) if wk else .008 * math.sin(t * 3.1 + ch.get("seed", 0))
    base = -.25 if sit else 0.0
    I = lambda x: int(round(max(-30000, min(30000, x))))
    X = lambda lx, ly: (I(u + lx * sc), I(v - (ly + bob + base) * sc))
    L = lambda m: max(1, I(m * sc))
    ow = max(2, L(.024))
    LI = light_for(r, Pw)
    d2, kk, tint = LI["d"], LI["k"], LI["color"]
    skin, hair, shirt = ch["skin"], ch["hair"], ch["shirt"]

    def lit(b, k=1.0):                                              # roshni wali taraf ka rang: base + light ka tint
        b = np.array(b, float)
        o = b * (1 - kk) + b * (np.array(tint, float) / 255.0) * 1.25 * kk
        return r.col(tuple(float(x) for x in np.clip(o, 0, 255)), k)

    def shd(b):
        return tuple(int(x * s) for x, s in zip(b, SHADE))

    def solid(bbox, draw, fill, ss=.42):
        """Flat fill + ek hi flat saaya band (shape minus roshni ki taraf khiska hua shape) = 2-tone cel."""
        b = lit(fill)
        draw(fr, 0, 0, b)
        x0, y0, x1, y1 = bbox
        sz = min(x1 - x0, y1 - y0)
        if sz < 6 or (x1 - x0) * (y1 - y0) > 3_500_000:
            return
        sh = (I(d2[0] * sz * ss), I(-d2[1] * sz * ss))
        ox, oy, w_, h_ = x0 - 3, y0 - 3, x1 - x0 + 6, y1 - y0 + 6
        m1 = np.zeros((h_, w_), np.uint8)
        m2 = np.zeros_like(m1)
        draw(m1, -ox, -oy, 255)
        draw(m2, -ox + sh[0], -oy + sh[1], 255)
        m = cv2.subtract(m1, m2)
        ax0, ay0, ax1, ay1 = max(0, ox), max(0, oy), min(Wf, ox + w_), min(Hf, oy + h_)
        if ax1 <= ax0 or ay1 <= ay0:
            return
        mm = m[ay0 - oy:ay1 - oy, ax0 - ox:ax1 - ox].astype(np.float32)[..., None] / 255.0
        roi = fr[ay0:ay1, ax0:ax1]
        roi[:] = (roi * (1 - mm) + np.array(shd(b), np.float32) * mm).astype(np.uint8)

    def ell(lx, ly, ax, ay, fill, w=ow, cel=True, rim=False):
        cx, cy = X(lx, ly)
        a_, b_ = max(1, I(ax * sc)), max(1, I(ay * sc))
        dr = lambda im, ox, oy, col: cv2.ellipse(im, (cx + ox, cy + oy), (a_, b_), 0, 0, 360, col, -1, cv2.LINE_AA)
        if cel:
            solid((cx - a_, cy - b_, cx + a_, cy + b_), dr, fill)
        else:
            dr(fr, 0, 0, r.col(fill))
        ang = math.degrees(math.atan2(-d2[1], d2[0]))
        if rim and kk > .04:                                         # kinare par rim light (roshni wali taraf)
            rc = tuple(int(min(255, a * .45 + 150)) for a in tint)
            cv2.ellipse(fr, (cx, cy), (max(1, I(a_ * .92)), max(1, I(b_ * .92))), 0, ang - 55, ang + 55, rc, max(1, I(ow * (.5 + kk * 2))), cv2.LINE_AA)
        if w:
            th = np.linspace(0, 2 * math.pi, 40 if max(a_, b_) < 80 else 72, endpoint=False)
            kb = 1 + .014 * np.sin(3 * th + PH) + .009 * np.sin(5 * th - PH * 1.3)
            pts = np.stack([cx + a_ * kb * np.cos(th), cy + b_ * kb * np.sin(th)], 1).astype(np.int32)
            cv2.polylines(fr, [pts], True, OUT, w, cv2.LINE_AA)
            if cel:                                                   # saaye wali taraf outline thodi mote
                cv2.ellipse(fr, (cx, cy), (a_, b_), 0, ang + 180 - 70, ang + 180 + 70, OUT, w + max(1, w // 3), cv2.LINE_AA)

    def pg(P_, fill, w=ow, cel=True):
        p = np.array([X(*q) for q in P_], np.int32)
        if fill is not None:
            if cel:
                dr = lambda im, ox, oy, col: cv2.fillPoly(im, [p + np.array([ox, oy], np.int32)], col, cv2.LINE_AA)
                solid((int(p[:, 0].min()), int(p[:, 1].min()), int(p[:, 0].max()), int(p[:, 1].max())), dr, fill, .35)
            else:
                cv2.fillPoly(fr, [p], r.col(fill), cv2.LINE_AA)
        if w:
            cv2.polylines(fr, [p], True, OUT, w, cv2.LINE_AA)

    def seg(a, b, w):
        cv2.line(fr, X(*a), X(*b), OUT, w, cv2.LINE_AA)

    def arc(cx_, cy_, ax, ay, a0, a1, w):
        cv2.ellipse(fr, X(cx_, cy_), (L(ax), L(ay)), 0, a0, a1, OUT, w, cv2.LINE_AA)

    # ---------------- pichhle baal (lambe baal) + tangein
    A_, B_, cy_ = .215, .25, 1.45
    sty = ch.get("style", "messy")
    if sty == "long":
        pg([(-A_ * 1.12, cy_ + .02), (-A_ * 1.2, 1.08), (-A_ * .5, 1.04), (A_ * .5, 1.04), (A_ * 1.2, 1.08), (A_ * 1.12, cy_ + .02)], hair, ow)
    elif sty == "bob":
        pg([(-A_ * 1.2, cy_ + .05), (-A_ * 1.22, cy_ - B_ * .8), (A_ * 1.22, cy_ - B_ * .8), (A_ * 1.2, cy_ + .05)], hair, ow)
    if not sit:
        for sd in (-1, 1):
            q = ph + (math.pi if sd > 0 else 0)
            lift = max(0, math.sin(q)) * .10 if wk else 0
            seg((sd * .07, .80), (sd * .09, .42 + lift * .5), L(.032))
            seg((sd * .09, .42 + lift * .5), (sd * .11, lift), L(.032))
            ell(sd * .135, lift + .02, .075, .035, ch.get("shoe", (28, 25, 30)), 0, cel=False)

    # ---------------- dhad + kapde
    ell(0, .98, .20, .26, shirt, rim=True)
    out = ch.get("outfit", "tee")
    if out == "suit":
        pg([(-.07, 1.22), (.07, 1.22), (0, .95)], (255, 255, 255), L(.01), cel=False)
        seg((-.07, 1.22), (-.12, .98), L(.012))
        seg((.07, 1.22), (.12, .98), L(.012))
        if ch.get("tie"):
            pg([(-.025, 1.21), (.025, 1.21), (.035, 1.08), (0, .84), (-.035, 1.08)], ch.get("tie_col", (40, 40, 210)), L(.01), cel=False)
    elif out == "hoodie":
        pg([(-.11, .78), (.11, .78), (.13, .92), (-.13, .92)], tuple(int(x * .8) for x in shirt), L(.012))
        seg((-.04, 1.2), (-.05, 1.04), L(.008))
        seg((.04, 1.2), (.05, 1.04), L(.008))
    elif out == "polo":
        pg([(-.07, 1.23), (0, 1.15), (-.02, 1.1)], (240, 240, 240), L(.01), cel=False)
        pg([(.07, 1.23), (0, 1.15), (.02, 1.1)], (240, 240, 240), L(.01), cel=False)
    elif out == "blouse":
        pg([(-.07, 1.22), (.07, 1.22), (0, 1.07)], skin, L(.01), cel=False)
    if ch.get("id"):                                                  # lanyard + ID card
        seg((-.06, 1.22), (-.03, 1.0), L(.01))
        seg((.06, 1.22), (.03, 1.0), L(.01))
        pg([(-.05, 1.0), (.05, 1.0), (.05, .86), (-.05, .86)], (255, 255, 255), L(.01), cel=False)
        pg([(-.05, 1.0), (.05, 1.0), (.05, .96), (-.05, .96)], ch["id"], 0, cel=False)

    # ---------------- bazu (patli stick) + haath
    arms = dict(auto_arms(st, ex, talk))
    arms.update(st.get("arms") or {})
    for sd, k in ((-1, "l"), (1, "r")):
        sh = (sd * .18, 1.15)
        hd = arms[k]
        el = ((sh[0] + hd[0]) / 2 + sd * .05, (sh[1] + hd[1]) / 2 - .02)
        seg(sh, el, L(.03))
        seg(el, hd, L(.03))
        cv2.circle(fr, X(*hd), L(.04), r.col(skin), -1, cv2.LINE_AA)
        cv2.circle(fr, X(*hd), L(.04), OUT, L(.012), cv2.LINE_AA)
    h = arms["r"]
    pr = set(st.get("props") or [])
    if st.get("key"):
        pr.add("key")
    if st.get("phone"):
        pr.add("phone")
    if "key" in pr:
        cv2.circle(fr, X(h[0], h[1] + .05), L(.03), (130, 200, 240), L(.01), cv2.LINE_AA)
        pg([(h[0] - .035, h[1] - .01), (h[0] + .035, h[1] - .01), (h[0] + .035, h[1] - .17), (h[0] - .035, h[1] - .17)], (50, 50, 215), L(.012), cel=False)
    if "phone" in pr:
        pg([(h[0] - .04, h[1] + .07), (h[0] + .04, h[1] + .07), (h[0] + .04, h[1] - .06), (h[0] - .04, h[1] - .06)], (50, 45, 48), L(.012), cel=False)
        pg([(h[0] - .03, h[1] + .06), (h[0] + .03, h[1] + .06), (h[0] + .03, h[1] - .04), (h[0] - .03, h[1] - .04)], (235, 200, 120), 0, cel=False)
    if "coffee" in pr or "mug" in pr:
        pg([(h[0] - .05, h[1] + .08), (h[0] + .05, h[1] + .08), (h[0] + .035, h[1] - .05), (h[0] - .035, h[1] - .05)], (250, 250, 250), L(.012), cel=False)
        pg([(h[0] - .045, h[1] + .03), (h[0] + .045, h[1] + .03), (h[0] + .04, h[1] - .02), (h[0] - .04, h[1] - .02)], (60, 90, 150), 0, cel=False)
        for dx in (-.02, .02):
            arc(h[0] + dx, h[1] + .13, .015, .03, 200, 340, max(1, L(.006)))
    if "papers" in pr or "clipboard" in pr:
        pg([(h[0] - .09, h[1] + .14), (h[0] + .09, h[1] + .14), (h[0] + .09, h[1] - .10), (h[0] - .09, h[1] - .10)], (245, 245, 245) if "papers" in pr else (70, 110, 170), L(.012), cel=False)
        for yy in (.08, .03, -.02):
            seg((h[0] - .06, h[1] + yy), (h[0] + .06, h[1] + yy), max(1, L(.005)))

    # ---------------- sar
    sx_, sy_ = lk[0] * .03, 0.0                                       # sar thoda mudta hai (look.x ki taraf)
    for sd in (-1, 1):
        ell(sd * (A_ * .98), cy_ - .01, .035, .05, skin, max(1, L(.014)), cel=False)   # kaan
    ell(0, cy_, A_, B_, skin, rim=True)
    # baal (aage ka hissa)
    fy = ch.get("fy", .4)
    outer = [(1.05 * A_ * math.cos(a), cy_ + 1.07 * B_ * math.sin(a)) for a in np.linspace(math.pi * .96, math.pi * .04, 16)]
    inner = [(.98 * A_ * math.cos(a), cy_ + B_ * (fy + .1 * math.cos(5 * a) * ch.get("wave", 1))) for a in np.linspace(math.pi * .04, math.pi * .96, 16)]
    if sty == "bald":
        for sd in (-1, 1):
            ell(sd * A_ * .95, cy_ + .03, .03, .07, hair, max(1, L(.014)), cel=False)
    else:
        pg(outer + inner, hair, ow)
    if sty == "bun":
        ell(0, cy_ + B_ * 1.18, .085, .075, hair, ow)
    elif sty == "spiky":
        for i, a in enumerate(np.linspace(.22, .78, 5)):
            x0 = A_ * math.cos(math.pi * (1 - a))
            pg([(x0 - .05, cy_ + B_ * 1.0), (x0 + .01 * (i - 2), cy_ + B_ * (1.38 + .05 * (i % 2))), (x0 + .05, cy_ + B_ * 1.0)], hair, max(1, L(.016)), cel=False)
    elif sty == "messy":
        for dx in (-.09, .0, .09):
            pg([(dx - .04, cy_ + B_ * 1.0), (dx + .02, cy_ + B_ * 1.28), (dx + .05, cy_ + B_ * 1.0)], hair, max(1, L(.016)), cel=False)

    # ---------------- chehra: aankhein, bhaunein, chashma, munh
    blink = ((t + ch.get("seed", 0) * .9) % 3.3) < .12
    ey = 1.47
    for sd in (-1, 1):
        cx = sd * .078 + sx_
        if ex == "shock":
            ell(cx, ey, .058, .062, (255, 255, 255), L(.014), cel=False)
            cv2.circle(fr, X(cx + lk[0] * .02, ey + lk[1] * .02), max(1, L(.014)), OUT, -1, cv2.LINE_AA)
        elif ex == "happy":
            cv2.ellipse(fr, X(cx, ey - .012), (L(.04), L(.035)), 0, 200, 340, OUT, L(.014), cv2.LINE_AA)
        elif blink:
            seg((cx - .03, ey), (cx + .03, ey), L(.012))
        else:
            cv2.circle(fr, X(cx + lk[0] * .015, ey + lk[1] * .012), max(2, L(.026)), OUT, -1, cv2.LINE_AA)
        by = ey + (.095 if ex == "shock" else .068 if ex in ("happy", "grin") else .058)          # bhaunein
        if ex == "sad":
            seg((cx + sd * .03, by + .005), (cx - sd * .03, by + .035), max(1, L(.012)))        # andar ki taraf upar
            if (t * 1.3) % 2 < 1.6:
                ell(cx + sd * .02, ey - .06 - ((t * .12) % .05), .012, .022, (255, 215, 150), 0, cel=False)   # aansu
        else:
            seg((cx - .035, by), (cx + .035, by + (.012 if ex == "smile" else 0)), max(1, L(.012)))
    if ch.get("glasses"):
        for sd in (-1, 1):
            cv2.circle(fr, X(sd * .078 + sx_, ey), L(.062), OUT, L(.01), cv2.LINE_AA)
        seg((-.016 + sx_, ey), (.016 + sx_, ey), L(.01))
    my = 1.355
    mo = sx_
    if ch.get("mustache"):
        pg([(mo - .07, my + .07), (mo, my + .055), (mo + .07, my + .07), (mo + .05, my + .035), (mo, my + .045), (mo - .05, my + .035)], hair, max(1, L(.01)), cel=False)
    if ex in ("grin", "shock"):                                       # bade dant: safed patti + khaane
        y0, y1 = my + .035, my - .035 - .04 * talk
        p0, p1 = X(mo - .10, y0), X(mo + .10, y1)
        cv2.rectangle(fr, p0, p1, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.rectangle(fr, p0, p1, OUT, L(.012), cv2.LINE_AA)
        for x in (-.06, -.02, .02, .06):
            seg((mo + x, y0), (mo + x, y1), max(1, L(.006)))
        seg((mo - .10, (y0 + y1) / 2), (mo + .10, (y0 + y1) / 2), max(1, L(.006)))
    elif ex == "happy":                                               # khula D-munh
        hh = .05 + .04 * talk
        cv2.ellipse(fr, X(mo, my + .035), (L(.075), L(hh)), 0, 0, 180, (40, 25, 70), -1, cv2.LINE_AA)
        cv2.ellipse(fr, X(mo, my + .035), (L(.075), L(hh)), 0, 0, 180, OUT, L(.014), cv2.LINE_AA)
        seg((mo - .075, my + .035), (mo + .075, my + .035), L(.014))
    elif talk > .05:
        ell(mo, my, .05 + .02 * talk, .008 + .045 * talk, (40, 25, 70), L(.012), cel=False)
    elif ex == "sad":
        cv2.ellipse(fr, X(mo, my - .015), (L(.06), L(.035)), 0, 205, 335, OUT, L(.014), cv2.LINE_AA)
    else:
        cv2.ellipse(fr, X(mo, my + .03), (L(.07), L(.045)), 0, 25, 155, OUT, L(.014), cv2.LINE_AA)
