"""Character renderer v2 (detailed toon): rounded shaded body, sleeves/pants, hands with thumbs, shaded face, iris+highlight eyes, brows, nose, blush,
teeth/tongue, hair highlights, hats, beards, accessories, head tilt, breathing, arm-swing walk, jump. Sirf character ke ROI par supersampling (tez).
Public contract: EXPRS, HAIRSTYLES, OUTFITS, draw_char(r, P_world, ch, st). Comments Roman Urdu mein."""
import math
import cv2
import numpy as np

OUT = (8, 6, 10)
OW = .030                      # mota cartoon outline
SHADE = (0.80, 0.72, 0.76)
BOIL_AMPLITUDE = 0.0
SS = 3

EXPRS = ("neutral", "smile", "grin", "laugh", "happy", "sad", "cry", "angry", "shock", "scared", "smug", "tired", "confused", "embarrassed")
HAIRSTYLES = ("cap", "bowl", "messy", "side_part", "bald_sides", "ponytail", "bun", "curly", "long", "spiky", "hijab", "topi", "bald", "afro", "braids", "crop")
HATS = ("crown", "straw", "beanie", "chef", "cap_hat", "turban", "bandana", "hardhat")
OUTFITS = ("suit", "shirt", "hoodie", "tee", "dress", "kurta", "lab_coat", "uniform", "apron", "vest", "robe", "overalls", "jacket", "blouse")
SLEEVELESS = ("tee", "dress", "blouse", "apron", "vest", "overalls")


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def smooth(a, b, t):
    x = clamp((t - a) / max(1e-8, b - a))
    return x * x * (3.0 - 2.0 * x)


def _d(c, k):
    return tuple(int(clamp(v * k, 0, 255)) for v in c)


def _l(c, k):
    return tuple(int(min(255, v + (255 - v) * k)) for v in c)


def light_for(r, Pw):
    """Qareebi light se halka rang aur direction hasil karo."""
    lights = getattr(r, "lights", None) or []
    if not lights:
        return (255, 255, 255), 0.0, np.array([0.5, 0.8])
    best = None
    for light in lights:
        pos = np.asarray(light.get("pos", Pw), dtype=np.float64)
        dist = float(np.linalg.norm(pos - Pw))
        power = float(light.get("intensity", 1.0)) / (1.0 + 0.12 * dist * dist)
        if best is None or power > best[0]:
            best = (power, light, pos)
    power, light, pos = best
    delta = pos - Pw
    direction = np.array([delta[0], max(0.15, delta[1])], dtype=np.float64)
    length = float(np.linalg.norm(direction))
    if length > 1e-7:
        direction /= length
    else:
        direction = np.array([0.5, 0.8])
    return tuple(light.get("color", (255, 255, 255))), clamp(power * 0.35, 0, 0.4), direction


def gesture(kind, t, talk=0.0):
    """Arms ki local pose: (x, y) haath ki jagah (character units)."""
    s = math.sin
    if kind == "wave":
        return {"l": (-.30, .82), "r": (.40 + .06 * s(t * 8), 1.45 + .05 * s(t * 8))}
    if kind == "point":
        return {"l": (-.30, .82), "r": (.68, 1.24 + .02 * s(t * 3))}
    if kind == "point_up":
        return {"l": (-.30, .82), "r": (.30, 1.78 + .03 * s(t * 4))}
    if kind == "shrug":
        return {"l": (-.46, 1.04 + .03 * s(t * 2)), "r": (.46, 1.04 + .03 * s(t * 2))}
    if kind in ("cheer", "clap"):
        y = 1.48 + .05 * abs(s(t * 6))
        return {"l": (-.36, y), "r": (.36, y)}
    if kind == "facepalm":
        return {"l": (-.30, .80), "r": (.10, 1.50)}
    if kind == "sad":
        return {"l": (-.22, .66), "r": (.22, .66)}
    if kind == "type":
        return {"l": (-.18 + .012 * s(t * 8), 1.02), "r": (.18 - .012 * s(t * 8), 1.02)}
    if kind == "think":
        return {"l": (.0, .92), "r": (.09, 1.32 + .01 * s(t * 2))}
    if kind == "cross":
        return {"l": (.16, 1.0), "r": (-.16, 1.0)}
    if kind == "hold_up":
        return {"l": (-.30, .80), "r": (.34, 1.52 + .03 * s(t * 3))}
    if kind == "bow":
        return {"l": (-.14, .70), "r": (.14, .70)}
    if kind == "talk":
        return {"l": (-.30 - .03 * s(t * 2.1) * talk, .82 + .05 * s(t * 3.1) * talk),
                "r": (.36 + .08 * s(t * 3) * talk, 1.0 + .15 * s(t * 2.5) * talk)}
    return {"l": (-.30, .80 + .012 * s(t * 1.6)), "r": (.30, .80 + .012 * s(t * 1.6 + 1))}


def auto_arms(st, expr, talk):
    kind = st.get("gesture", "idle")
    t = st.get("t", 0.0)
    if kind != "idle":
        return gesture(kind, t, talk)
    if st.get("sit"):
        return gesture("type" if talk < .05 else "talk", t, talk)
    if expr in ("shock", "scared"):
        return gesture("cheer", t, talk)
    if expr in ("sad", "cry"):
        return gesture("sad", t, talk)
    if expr == "confused" and talk < .05:
        return gesture("think", t, talk)
    if talk > .05:
        return gesture("talk", t, talk)
    return gesture("idle", t, talk)


def _draw(r, Pw, ch, st):
    frame = r.fr
    H, W = frame.shape[:2]
    t = float(st.get("t", 0.0))
    Pw = np.asarray(Pw, dtype=np.float64)
    pc = r.c.cam(Pw)[0]
    depth = float(pc[2])
    if depth < 0.35:
        return
    uv = r.c.proj(pc.reshape(1, 3))[0]
    u, v = float(uv[0]), float(uv[1])
    size = float(ch.get("size", 1.0))
    scale = float(r.c.f) / depth * size

    # ROI: sirf character ke aas-paas supersample (poori frame nahi)
    hw, top, bot = 1.05 * scale, 2.45 * scale, .38 * scale
    rx0, ry0 = int(math.floor(u - hw)), int(math.floor(v - top))
    rw, rh = int(math.ceil(2 * hw)) + 2, int(math.ceil(top + bot)) + 2
    if rx0 > W or ry0 > H or rx0 + rw < 0 or ry0 + rh < 0 or rw < 4 or rh < 4:
        return
    canvas = np.zeros((rh * SS, rw * SS, 3), dtype=np.uint8)
    alpha = np.zeros((rh * SS, rw * SS), dtype=np.uint8)

    facing = st.get("facing", "front")
    face_sign = -1.0 if facing == "left" else 1.0
    face_scale = .90 if facing in ("three_quarter", "left", "right") else 1.0
    face_off = .035 if facing in ("three_quarter", "left", "right") else 0.0
    shear = [0.0]                                           # sar ka tilt: y > gardan par x ko khiskao

    def xy(x, y):
        lx = float(x)
        if y > 1.3:
            lx += shear[0] * (y - 1.3) + face_off
        lx = lx * face_sign * face_scale
        return (int(round((u - rx0 + lx * scale) * SS)), int(round((v - ry0 - y * scale) * SS)))

    def width(value):
        return max(1, int(round(abs(value * scale) * SS)))

    tint, power, direction = light_for(r, Pw)
    lightv = np.asarray(tint, dtype=np.float32)

    def color(value, factor=1.0):
        base = np.asarray(value, dtype=np.float32)
        lit = base * (1 - power) + base * (lightv / 255.0) * power
        return tuple(int(x) for x in np.clip(lit * factor, 0, 255))

    skin = tuple(ch.get("skin", (232, 242, 250)))
    hair = tuple(ch.get("hair", (35, 30, 40)))
    shirt = tuple(ch.get("shirt", (80, 150, 210)))
    pants = tuple(ch.get("pants", (70, 60, 55)))
    shoe = tuple(ch.get("shoe", (25, 22, 30)))
    iris = tuple(ch.get("eye", (80, 55, 40)))
    seed = int(ch.get("seed", 1))
    blush = _d(skin, .9)
    blush = (int(blush[0] * .72), int(blush[1] * .78), min(255, int(blush[2] * 1.0) + 8))

    def ell(cx, cy, rx, ry, fill, outline=True, ow=OW, ang=0):
        c = xy(cx, cy)
        ax = (width(rx), width(ry))
        cv2.ellipse(canvas, c, ax, ang, 0, 360, color(fill), -1, cv2.LINE_AA)
        cv2.ellipse(alpha, c, ax, ang, 0, 360, 255, -1, cv2.LINE_AA)
        if outline:
            cv2.ellipse(canvas, c, ax, ang, 0, 360, OUT, width(ow), cv2.LINE_AA)
            cv2.ellipse(alpha, c, ax, ang, 0, 360, 255, width(ow), cv2.LINE_AA)

    def poly(points, fill, outline=True, ow=OW):
        p = np.asarray([xy(x, y) for x, y in points], dtype=np.int32)
        cv2.fillPoly(canvas, [p], color(fill), cv2.LINE_AA)
        cv2.fillPoly(alpha, [p], 255, cv2.LINE_AA)
        if outline:
            cv2.polylines(canvas, [p], True, OUT, width(ow), cv2.LINE_AA)
            cv2.polylines(alpha, [p], True, 255, width(ow), cv2.LINE_AA)

    def line(a, b, thick=.025, col=OUT):
        cv2.line(canvas, xy(*a), xy(*b), col, width(thick), cv2.LINE_AA)
        cv2.line(alpha, xy(*a), xy(*b), 255, width(thick), cv2.LINE_AA)

    def curve(points, thick=.025, col=OUT):
        p = np.asarray([xy(x, y) for x, y in points], np.int32)
        cv2.polylines(canvas, [p], False, col, width(thick), cv2.LINE_AA)
        cv2.polylines(alpha, [p], False, 255, width(thick), cv2.LINE_AA)

    def limb(a, b, thick, col):
        """Outline wala mota limb (pehle kaala, phir rang)."""
        cv2.line(canvas, xy(*a), xy(*b), OUT, width(thick + OW * 1.6), cv2.LINE_AA)
        cv2.line(alpha, xy(*a), xy(*b), 255, width(thick + OW * 1.6), cv2.LINE_AA)
        cv2.line(canvas, xy(*a), xy(*b), color(col), width(thick), cv2.LINE_AA)

    # ---------------------------------------------------------------- motion (sirf continuous time se)
    sit = bool(st.get("sit", False))
    walk = bool(st.get("walk", False))
    phase = float(st.get("ph", t * 2.0))
    expr = st.get("expr", "smile")
    if expr not in EXPRS:
        expr = "smile"
    talk = clamp(st.get("talk", 0))
    jump = st.get("gesture") == "jump"
    bob = .028 * abs(math.sin(phase)) if walk else .012 * math.sin(t * 2.4 + seed)
    if jump:
        bob += .35 * abs(math.sin(t * 5))
    if st.get("gesture") == "bow":
        bob -= .04
    base = -.22 if sit else 0.0
    breathe = 1.0 + .012 * math.sin(t * 2.2 + seed)
    squash = 1.0 + (.02 * math.sin(t * 14) * talk)
    Y = lambda y: (y * (breathe if y > .85 else 1.0)) + bob + base
    shear[0] = (.05 * math.sin(t * 1.7 + seed) + (.05 if talk > .05 else 0.0) * math.sin(t * 5)
                + {"confused": .13, "sad": -.04, "smug": .07, "embarrassed": -.05, "shock": 0.0}.get(expr, 0.0)) * (1.0 if not jump else .4)
    if st.get("gesture") == "bow":
        shear[0] += .35

    head_w = float(ch.get("head_w", .255))
    head_h = float(ch.get("head_h", .275))
    body_w = float(ch.get("body_w", .17))
    body_h = float(ch.get("body_h", .29))
    leg_w = float(ch.get("limb_w", .034))
    style = ch.get("style", "messy")
    outfit = ch.get("outfit", "tee")
    hat = ch.get("hat")
    HY = 1.48

    # ---------------------------------------------------------------- peechhe ke baal
    if style in ("long", "ponytail", "braids"):
        poly([(-head_w * 1.02, Y(1.55)), (-head_w * 1.16, Y(1.02)), (head_w * 1.16, Y(1.02)), (head_w * 1.02, Y(1.55))], _d(hair, .85))
    if style == "ponytail":
        sway = .03 * math.sin(t * 3 + seed)
        ell(head_w * 1.25 + sway, Y(1.36), .085, .17, hair)
    if style == "braids":
        for sd in (-1, 1):
            for k in range(4):
                ell(sd * head_w * 1.0 + .01 * math.sin(t * 2 + k), Y(1.30 - k * .09), .05, .05, hair if k % 2 else _d(hair, .85))
    if style == "afro":
        ell(0, Y(1.62), head_w * 1.45, head_h * 1.25, hair)

    # ---------------------------------------------------------------- pair (pants + shoes), walk cycle
    if not sit:
        for side in (-1, 1):
            p = phase + (math.pi if side > 0 else 0)
            lift = max(0.0, math.sin(p)) * (.14 if walk else 0.0)
            swing = math.cos(p) * (.06 if walk else 0.0)
            hip = (side * .075, Y(.80))
            knee = (side * .095 + swing * .5, Y(.42 + lift * .5))
            foot = (side * .115 + swing, Y(lift) if not jump else Y(.05))
            limb(hip, knee, leg_w * 1.25, pants)
            limb(knee, foot, leg_w * 1.2, pants)
            ell(foot[0] + side * .03 + .02, foot[1] - .02, .085, .042, shoe, True, OW * .8)
            ell(foot[0] + side * .03 + .045, foot[1] - .005, .04, .016, _l(shoe, .35), False)
    else:
        for side in (-1, 1):
            limb((side * .08, Y(.80)), (side * .20, Y(.76)), leg_w * 1.25, pants)
            limb((side * .20, Y(.76)), (side * .22, Y(.42)), leg_w * 1.2, pants)
            ell(side * .24, Y(.40), .075, .04, shoe, True, OW * .8)
    ell(0, Y(.80), body_w * .98, .075, pants, True, OW * .8)

    # ---------------------------------------------------------------- body (shaded, rounded) + outfit
    bw = body_w * squash
    ell(0, Y(.99), bw, body_h, shirt)
    ell(-bw * .55, Y(.97), bw * .42, body_h * .86, _d(shirt, .82), False)          # shade
    ell(bw * .42, Y(1.05), bw * .16, body_h * .40, _l(shirt, .18), False)           # highlight
    ell(0, Y(.99), bw, body_h, shirt, True, OW)                                     # outline dobara upar

    if outfit == "suit":
        poly([(-.075, Y(1.25)), (0, Y(1.10)), (.075, Y(1.25)), (.115, Y(.80)), (-.115, Y(.80))], (240, 240, 240))
        poly([(-.075, Y(1.25)), (-.02, Y(1.12)), (-.12, Y(.92))], _d(shirt, .9), True, OW * .7)
        poly([(.075, Y(1.25)), (.02, Y(1.12)), (.12, Y(.92))], _d(shirt, .9), True, OW * .7)
        if ch.get("tie"):
            tc = ch.get("tie_col", (40, 40, 200))
            poly([(-.022, Y(1.22)), (.022, Y(1.22)), (.03, Y(1.15)), (-.03, Y(1.15))], tc, True, OW * .7)
            poly([(-.03, Y(1.15)), (.03, Y(1.15)), (.045, Y(.98)), (0, Y(.82)), (-.045, Y(.98))], tc, True, OW * .8)
        for yy in (.97, .88):
            ell(.075, Y(yy), .012, .012, (60, 60, 70), False)
    elif outfit == "shirt":
        poly([(-.06, Y(1.24)), (0, Y(1.14)), (.06, Y(1.24)), (.03, Y(1.28)), (-.03, Y(1.28))], _l(shirt, .5), True, OW * .7)
        line((0, Y(1.14)), (0, Y(.72)), .008)
        for yy in (1.05, .93, .81):
            ell(0, Y(yy), .011, .011, (240, 240, 240), False)
    elif outfit == "hoodie":
        ell(0, Y(1.23), .13, .095, _d(shirt, .78))
        ell(0, Y(1.19), .075, .06, _d(skin, .8), False)
        poly([(-.10, Y(.88)), (.10, Y(.88)), (.085, Y(.74)), (-.085, Y(.74))], _d(shirt, .85), True, OW * .7)
        line((-.05, Y(1.18)), (-.04, Y(.98)), .012, (245, 245, 245))
        line((.05, Y(1.18)), (.04, Y(.98)), .012, (245, 245, 245))
    elif outfit == "dress":
        poly([(-.15, Y(1.18)), (.15, Y(1.18)), (.27, Y(.62)), (-.27, Y(.62))], shirt)
        poly([(-.27, Y(.62)), (.27, Y(.62)), (.25, Y(.56)), (-.25, Y(.56))], _d(shirt, .8), True, OW * .7)
        line((-.15, Y(.98)), (.15, Y(.98)), .028, _d(shirt, .65))
        curve([(-.06, Y(1.2)), (0, Y(1.14)), (.06, Y(1.2))], .012, _d(shirt, .6))
    elif outfit == "kurta":
        poly([(-.14, Y(1.22)), (.14, Y(1.22)), (.20, Y(.62)), (-.20, Y(.62))], shirt)
        line((0, Y(1.22)), (0, Y(.92)), .012, _d(shirt, .6))
        for yy in (1.12, 1.03, .94):
            ell(0, Y(yy), .011, .011, (235, 240, 245), False)
        curve([(-.07, Y(1.26)), (0, Y(1.20)), (.07, Y(1.26))], .016, _l(shirt, .5))
    elif outfit == "lab_coat":
        poly([(-.13, Y(1.24)), (-.20, Y(.62)), (-.04, Y(.62)), (0, Y(1.10)), (.04, Y(.62)), (.20, Y(.62)), (.13, Y(1.24))], (240, 242, 245))
        poly([(-.09, Y(.84)), (-.03, Y(.84)), (-.03, Y(.74)), (-.09, Y(.74))], (225, 228, 232), True, OW * .6)
        line((-.075, Y(.84)), (-.075, Y(.9)), .012, (60, 60, 200))
    elif outfit == "uniform":
        line((-.08, Y(1.19)), (.08, Y(1.19)), .022, (245, 245, 245))
        ell(.07, Y(1.02), .028, .034, (60, 200, 240), True, OW * .6)
        line((-.13, Y(.84)), (.13, Y(.84)), .03, _d(shirt, .55))
        ell(0, Y(.84), .028, .02, (60, 200, 240), True, OW * .5)
    elif outfit == "apron":
        poly([(-.11, Y(1.16)), (.11, Y(1.16)), (.14, Y(.64)), (-.14, Y(.64))], (240, 242, 245))
        line((-.11, Y(1.16)), (-.07, Y(1.28)), .014, (240, 242, 245))
        line((.11, Y(1.16)), (.07, Y(1.28)), .014, (240, 242, 245))
        poly([(-.06, Y(.82)), (.06, Y(.82)), (.06, Y(.72)), (-.06, Y(.72))], (225, 228, 232), True, OW * .6)
    elif outfit == "vest":
        poly([(-.14, Y(1.20)), (-.04, Y(1.22)), (0, Y(1.0)), (.04, Y(1.22)), (.14, Y(1.20)), (.15, Y(.8)), (-.15, Y(.8))], _d(pants, 1.1))
        for yy in (.98, .9):
            ell(0, Y(yy), .012, .012, (80, 190, 235), False)
    elif outfit == "robe":
        poly([(-.16, Y(1.22)), (.16, Y(1.22)), (.26, Y(.52)), (-.26, Y(.52))], shirt)
        poly([(-.05, Y(1.24)), (.05, Y(1.24)), (.07, Y(.52)), (-.07, Y(.52))], (60, 205, 245), True, OW * .7)
        for k in range(5):
            ell(-.0, Y(1.1 - k * .12), .014, .014, (50, 50, 210), False)
        poly([(-.26, Y(.52)), (.26, Y(.52)), (.25, Y(.47)), (-.25, Y(.47))], (60, 205, 245), True, OW * .6)
    elif outfit == "overalls":
        poly([(-.11, Y(1.15)), (.11, Y(1.15)), (.15, Y(.80)), (-.15, Y(.80))], (190, 120, 60))
        line((-.11, Y(1.15)), (-.10, Y(1.28)), .016, (190, 120, 60))
        line((.11, Y(1.15)), (.10, Y(1.28)), .016, (190, 120, 60))
        poly([(-.05, Y(.98)), (.05, Y(.98)), (.05, Y(.88)), (-.05, Y(.88))], (170, 100, 45), True, OW * .6)
    elif outfit == "jacket":
        poly([(-.13, Y(1.24)), (-.02, Y(1.26)), (-.01, Y(.72)), (-.16, Y(.76))], _d(shirt, .88))
        poly([(.13, Y(1.24)), (.02, Y(1.26)), (.01, Y(.72)), (.16, Y(.76))], _d(shirt, .88))
        line((0, Y(1.26)), (0, Y(.72)), .012, _d(shirt, .5))
    elif outfit == "blouse":
        curve([(-.08, Y(1.24)), (0, Y(1.14)), (.08, Y(1.24))], .016, _l(shirt, .55))
        poly([(-.05, Y(1.24)), (0, Y(1.14)), (.05, Y(1.24)), (.0, Y(1.28))], _l(shirt, .4), True, OW * .6)

    # ---------------------------------------------------------------- arms + hands
    arms = auto_arms(st, expr, talk)
    arms.update(st.get("arms") or {})
    sleeves = outfit not in SLEEVELESS
    arm_col = shirt if sleeves else skin
    if outfit == "hoodie":
        arm_col = _d(shirt, .95)
    for side, key in ((-1, "l"), (1, "r")):
        shoulder = (side * body_w * .90, Y(1.14))
        hx, hy = float(arms[key][0]), float(arms[key][1])
        if walk and st.get("gesture", "idle") == "idle":
            hx += -side * math.sin(phase + (math.pi if side > 0 else 0)) * .06
            hy += .03 * math.cos(phase * 2)
        hand = (hx, Y(hy))
        elbow = ((shoulder[0] + hand[0]) * .5 + side * .05, (shoulder[1] + hand[1]) * .5 - .04)
        limb(shoulder, elbow, leg_w * 1.35, arm_col)
        limb(elbow, hand, leg_w * 1.25, arm_col if sleeves else skin)
        if sleeves:
            ell(hand[0], hand[1] + (elbow[1] - hand[1]) * .08, leg_w * .92, leg_w * .55, _l(arm_col, .15), False)
        ell(hand[0], hand[1], .052, .052, skin, True, OW * .85)
        ell(hand[0] + side * .045, hand[1] + .02, .022, .03, skin, True, OW * .6)       # angootha

    # ---------------------------------------------------------------- hand props
    props = set(st.get("props") or [])
    if "paper" in props:
        props.add("papers")
    if st.get("phone"):
        props.add("phone")
    if st.get("key"):
        props.add("key")
    hand = arms["r"]
    hx, hy = float(hand[0]), float(hand[1])
    if "phone" in props:
        poly([(hx - .04, Y(hy + .08)), (hx + .04, Y(hy + .08)), (hx + .04, Y(hy - .07)), (hx - .04, Y(hy - .07))], (45, 45, 50), True, OW * .7)
        poly([(hx - .03, Y(hy + .06)), (hx + .03, Y(hy + .06)), (hx + .03, Y(hy - .04)), (hx - .03, Y(hy - .04))], (230, 190, 90), False)
    if "mug" in props or "coffee" in props:
        ell(hx, Y(hy + .03), .058, .072, (245, 245, 240))
        ell(hx + .068, Y(hy + .03), .036, .042, (245, 245, 240), True, OW * .7)
        ell(hx, Y(hy + .085), .045, .016, (50, 70, 110), False)
    if "papers" in props:
        poly([(hx - .09, Y(hy + .11)), (hx + .09, Y(hy + .11)), (hx + .09, Y(hy - .11)), (hx - .09, Y(hy - .11))], (245, 245, 240), True, OW * .7)
        for k in range(4):
            line((hx - .06, Y(hy + .07 - k * .045)), (hx + .06, Y(hy + .07 - k * .045)), .008, (170, 170, 175))
    if "key" in props:
        ell(hx, Y(hy + .04), .038, .038, (60, 205, 245), True, OW * .7)
        line((hx, Y(hy)), (hx, Y(hy - .10)), .016, (60, 205, 245))
    if "laptop" in props:
        poly([(-.20, Y(.91)), (.20, Y(.91)), (.17, Y(.83)), (-.17, Y(.83))], (75, 85, 100), True, OW * .7)
        poly([(-.16, Y(.94)), (.16, Y(.94)), (.16, Y(1.12)), (-.16, Y(1.12))], (45, 55, 70), True, OW * .7)
        poly([(-.13, Y(.97)), (.13, Y(.97)), (.13, Y(1.09)), (-.13, Y(1.09))], (115, 175, 205), False)
    if "bag" in props:
        curve([(-.13, Y(1.20)), (-.24, Y(1.02)), (-.22, Y(.78))], .025, (35, 30, 40))
        poly([(-.29, Y(.91)), (-.12, Y(.91)), (-.10, Y(.68)), (-.31, Y(.68))], (125, 85, 55), True, OW * .7)
    if "umbrella" in props:
        line((hx, Y(hy)), (hx, Y(hy + .45)), .02, (45, 45, 55))
        poly([(hx - .2, Y(hy + .42)), (hx, Y(hy + .57)), (hx + .2, Y(hy + .42))], (75, 105, 185), True, OW * .8)
    if "food" in props:
        ell(hx, Y(hy + .02), .08, .06, (75, 155, 205))
        ell(hx, Y(hy + .05), .062, .036, (145, 195, 225), False)
    if "coin" in props:
        ell(hx, Y(hy + .10), .07, .07, (50, 200, 250))
        ell(hx, Y(hy + .10), .048, .048, (80, 225, 255), False)
        line((hx, Y(hy + .13)), (hx, Y(hy + .07)), .014, (40, 150, 200))
        ell(hx - .02, Y(hy + .125), .012, .012, (240, 252, 255), False)
    if "money_bag" in props:
        poly([(hx - .10, Y(hy + .02)), (hx + .10, Y(hy + .02)), (hx + .13, Y(hy - .13)), (hx, Y(hy - .17)), (hx - .13, Y(hy - .13))], (90, 150, 190), True, OW * .8)
        poly([(hx - .05, Y(hy + .02)), (hx + .05, Y(hy + .02)), (hx + .07, Y(hy + .09)), (hx - .07, Y(hy + .09))], (70, 120, 160), True, OW * .6)
        curve([(hx - .03, Y(hy - .04)), (hx + .03, Y(hy - .04)), (hx + .03, Y(hy - .09)), (hx - .03, Y(hy - .09))], .012, (50, 200, 250))
    if "bread" in props:
        ell(hx, Y(hy + .06), .11, .06, (90, 160, 215))
        for k in (-1, 0, 1):
            line((hx + k * .04 - .015, Y(hy + .09)), (hx + k * .04 + .015, Y(hy + .03)), .008, (60, 110, 160))
    if "fish" in props:
        ell(hx, Y(hy + .05), .10, .05, (200, 160, 80))
        poly([(hx + .08, Y(hy + .05)), (hx + .16, Y(hy + .10)), (hx + .16, Y(hy)), ], (180, 140, 60), True, OW * .6)
        ell(hx - .05, Y(hy + .06), .01, .01, OUT, False)
    if "apple" in props:
        ell(hx, Y(hy + .06), .06, .06, (60, 60, 220))
        line((hx, Y(hy + .12)), (hx + .02, Y(hy + .17)), .012, (40, 80, 110))
        ell(hx + .04, Y(hy + .15), .028, .014, (80, 190, 80), False)

    # ---------------------------------------------------------------- gardan, kaan, sar (shaded)
    ell(0, Y(1.30), .08, .09, _d(skin, .88), False)
    for side in (-1, 1):
        ell(side * head_w * .96, Y(HY - .02), .048, .068, skin)
        ell(side * head_w * .96, Y(HY - .02), .022, .036, _d(skin, .82), False)
    ell(0, Y(HY), head_w, head_h, _d(skin, .84), False)                                # shade layer
    ell(.022, Y(HY + .018), head_w * .93, head_h * .94, skin, False)                   # roshan hissa
    ell(0, Y(HY), head_w, head_h, skin, True, OW)                                      # outline
    if hat not in ("turban",) and style not in ("hijab",):
        ell(-head_w * .38, Y(HY + head_h * .55), head_w * .22, head_h * .09, _l(skin, .5), False)  # forehead shine

    # ---------------------------------------------------------------- baal front
    if style == "cap":
        ell(0, Y(1.70), head_w * 1.04, .09, hair)
        poly([(-head_w, Y(1.69)), (head_w, Y(1.69)), (head_w * 1.3, Y(1.62)), (-head_w * 1.3, Y(1.62))], _d(hair, .9))
    elif style == "topi":
        poly([(-head_w * .9, Y(1.66)), (head_w * .9, Y(1.66)), (head_w * .78, Y(1.84)), (-head_w * .78, Y(1.84))], (240, 240, 232))
        for k in range(5):
            line((-head_w * .7 + k * head_w * .35, Y(1.69)), (-head_w * .7 + k * head_w * .35, Y(1.80)), .008, (200, 205, 195))
    elif style == "hijab":
        poly([(-head_w * 1.18, Y(1.70)), (-head_w * 1.28, Y(1.36)), (-head_w * 1.0, Y(1.10)), (head_w * 1.0, Y(1.10)), (head_w * 1.28, Y(1.36)), (head_w * 1.18, Y(1.70)),
              (0, Y(1.80))], hair)
        ell(0, Y(HY), head_w * .84, head_h * .82, skin, True, OW)
    elif style == "bald_sides":
        for side in (-1, 1):
            ell(side * head_w * .90, Y(1.55), .05, .105, hair)
            ell(side * head_w * .86, Y(1.60), .02, .05, _l(hair, .35), False)
        ell(-head_w * .3, Y(1.72), .05, .02, _l(skin, .55), False)
    elif style == "bun":
        ell(0, Y(HY + head_h * 1.12), .095, .085, hair)
        poly([(-head_w * 1.02, Y(1.56)), (-head_w * .9, Y(1.74)), (0, Y(1.78)), (head_w * .9, Y(1.74)), (head_w * 1.02, Y(1.56)), (head_w * .5, Y(1.64)), (-head_w * .5, Y(1.66))], hair)
        curve([(-head_w * .5, Y(1.74)), (0, Y(1.77)), (head_w * .5, Y(1.74))], .012, _l(hair, .35))
    elif style == "curly":
        for i in range(11):
            a = math.pi * (i / 10.0)
            ell(math.cos(a) * head_w * .95, Y(HY + math.sin(a) * head_h * .95 + .02), .062, .066, hair if i % 2 else _d(hair, .85))
        ell(-.06, Y(1.78), .04, .025, _l(hair, .3), False)
    elif style == "ponytail":
        poly([(-head_w * 1.02, Y(1.55)), (-head_w * .7, Y(1.76)), (head_w * .8, Y(1.76)), (head_w * 1.02, Y(1.55)), (head_w * .4, Y(1.66)), (-head_w * .4, Y(1.68))], hair)
        ell(head_w * 1.02, Y(1.62), .035, .035, (60, 70, 220), False)
    elif style == "long":
        poly([(-head_w * 1.02, Y(1.62)), (-head_w * 1.16, Y(1.10)), (-head_w * .86, Y(1.12)), (-head_w * .80, Y(1.50)), (-head_w * .3, Y(1.62)), (head_w * .4, Y(1.64)),
              (head_w * .80, Y(1.50)), (head_w * .86, Y(1.12)), (head_w * 1.16, Y(1.10)), (head_w * 1.02, Y(1.62)), (0, Y(1.78))], hair)
        curve([(-head_w * .5, Y(1.74)), (0, Y(1.77)), (head_w * .5, Y(1.74))], .012, _l(hair, .35))
    elif style == "spiky":
        for i in range(6):
            x = (i - 2.5) * head_w * .33
            poly([(x - .055, Y(1.66)), (x + .005, Y(1.90 + .03 * (i % 2))), (x + .06, Y(1.66))], hair if i % 2 else _d(hair, .88))
        poly([(-head_w * 1.0, Y(1.58)), (-head_w * .9, Y(1.7)), (head_w * .9, Y(1.7)), (head_w * 1.0, Y(1.58))], hair, False)
    elif style == "afro":
        poly([(-head_w * 1.0, Y(1.62)), (-head_w * .7, Y(1.72)), (head_w * .7, Y(1.72)), (head_w * 1.0, Y(1.62)), (head_w * .6, Y(1.66)), (-head_w * .6, Y(1.68))], hair, False)
    elif style == "braids":
        poly([(-head_w * 1.02, Y(1.56)), (-head_w * .8, Y(1.74)), (head_w * .8, Y(1.74)), (head_w * 1.02, Y(1.56)), (head_w * .4, Y(1.66)), (-head_w * .4, Y(1.68))], hair)
    elif style == "crop":
        poly([(-head_w * 1.02, Y(1.56)), (-head_w * .85, Y(1.72)), (head_w * .85, Y(1.72)), (head_w * 1.02, Y(1.56)), (head_w * .5, Y(1.62)), (-head_w * .5, Y(1.63))], hair)
    elif style in ("messy", "side_part", "bowl"):
        poly([(-head_w * 1.06, Y(1.55)), (-head_w * 1.0, Y(1.76)), (-head_w * .4, Y(1.80)), (head_w * .25, Y(1.74)), (head_w * 1.0, Y(1.72)), (head_w * 1.06, Y(1.53)),
              (head_w * .6, Y(1.64)), (0, Y(1.66)), (-head_w * .6, Y(1.62))], hair)
        curve([(-head_w * .7, Y(1.75)), (-head_w * .2, Y(1.78)), (head_w * .3, Y(1.74))], .014, _l(hair, .35))
        if style == "side_part":
            line((-.02, Y(1.78)), (.09, Y(1.68)), .012, _d(hair, .5))
        if style == "bowl":
            curve([(-head_w, Y(1.60)), (-head_w * .5, Y(1.55)), (0, Y(1.54)), (head_w * .5, Y(1.55)), (head_w, Y(1.60))], .04, hair)
    # ---------------------------------------------------------------- topiyan
    if hat == "crown":
        poly([(-head_w * .72, Y(1.72)), (head_w * .72, Y(1.72)), (head_w * .78, Y(1.90)), (head_w * .42, Y(1.82)), (0, Y(1.96)), (-head_w * .42, Y(1.82)), (-head_w * .78, Y(1.90))],
             (55, 205, 250))
        for k, c in ((-1, (50, 50, 215)), (0, (215, 120, 60)), (1, (50, 50, 215))):
            ell(k * head_w * .4, Y(1.76), .022, .022, c, False)
        ell(-head_w * .25, Y(1.84), .02, .03, (140, 235, 255), False)
    elif hat == "straw":
        ell(0, Y(1.70), head_w * 1.75, .06, (110, 200, 235))
        poly([(-head_w * .8, Y(1.70)), (head_w * .8, Y(1.70)), (head_w * .66, Y(1.88)), (-head_w * .66, Y(1.88))], (125, 215, 245))
        line((-head_w * .78, Y(1.73)), (head_w * .78, Y(1.73)), .03, (70, 80, 200))
    elif hat == "beanie":
        poly([(-head_w * 1.04, Y(1.62)), (head_w * 1.04, Y(1.62)), (head_w * .9, Y(1.84)), (0, Y(1.90)), (-head_w * .9, Y(1.84))], (60, 90, 220))
        ell(0, Y(1.64), head_w * 1.06, .045, (50, 70, 190))
        ell(0, Y(1.93), .04, .04, (245, 245, 245), True, OW * .7)
    elif hat == "chef":
        ell(0, Y(1.80), head_w * .95, .13, (250, 250, 250))
        ell(-head_w * .55, Y(1.88), .10, .09, (250, 250, 250))
        ell(head_w * .55, Y(1.88), .10, .09, (250, 250, 250))
        poly([(-head_w * .85, Y(1.66)), (head_w * .85, Y(1.66)), (head_w * .85, Y(1.76)), (-head_w * .85, Y(1.76))], (245, 245, 245))
    elif hat == "cap_hat":
        poly([(-head_w * 1.0, Y(1.70)), (head_w * 1.0, Y(1.70)), (head_w * .8, Y(1.86)), (-head_w * .8, Y(1.86))], (60, 80, 215))
        poly([(head_w * .1, Y(1.68)), (head_w * 1.6, Y(1.68)), (head_w * 1.5, Y(1.64)), (head_w * .1, Y(1.64))], _d((60, 80, 215), .8), True, OW * .8)
    elif hat == "turban":
        ell(0, Y(1.72), head_w * 1.12, head_h * .45, (235, 230, 245))
        curve([(-head_w, Y(1.70)), (0, Y(1.80)), (head_w, Y(1.70))], .015, (180, 170, 200))
        curve([(-head_w * .9, Y(1.76)), (0, Y(1.66)), (head_w * .9, Y(1.76))], .015, (180, 170, 200))
    elif hat == "bandana":
        poly([(-head_w * 1.02, Y(1.64)), (head_w * 1.02, Y(1.64)), (head_w * .9, Y(1.74)), (-head_w * .9, Y(1.74))], (60, 60, 215))
        for k in range(4):
            ell(-head_w * .6 + k * head_w * .4, Y(1.69), .01, .01, (245, 245, 245), False)
    elif hat == "hardhat":
        ell(0, Y(1.74), head_w * 1.0, .11, (45, 205, 250))
        ell(0, Y(1.66), head_w * 1.25, .035, (40, 180, 225))
        line((0, Y(1.64)), (0, Y(1.84)), .02, _d((45, 205, 250), .8))

    # ---------------------------------------------------------------- chehra: aankhein, bhawein, naak, blush
    look = st.get("look", (0, 0))
    lx, ly = float(look[0]), float(look[1])
    eye_y = Y(1.50)
    blink_period = 3.1 + (seed % 4) * .43
    blink_phase = (t + seed * .37) % blink_period
    blink = smooth(0.0, .045, blink_phase) * (1 - smooth(.11, .17, blink_phase))
    eye_open = 1.0 - blink
    if expr in ("tired", "smug"):
        eye_open *= .55
    if expr in ("shock", "scared"):
        eye_open = min(1.15, eye_open * 1.2)
    brow_raise = {"shock": .05, "scared": .045, "angry": -.02, "confused": .015, "smug": .0, "sad": .01}.get(expr, 0.0)
    brow_col = _d(hair, .8) if sum(hair) > 120 else hair
    ex = .088
    for side in (-1, 1):
        cx = side * ex
        if expr in ("happy", "laugh"):
            curve([(cx - .04, eye_y - .005), (cx - .02, eye_y + .022), (cx + .02, eye_y + .022), (cx + .04, eye_y - .005)], .018)
        else:
            ell(cx, eye_y, .038, .050 * max(.06, eye_open), (252, 252, 250), True, OW * .55)
            if eye_open > .15:
                px = cx + clamp(lx, -1, 1) * .018
                py = eye_y + clamp(ly, -1, 1) * .018
                ell(px, py, .021, .033 * max(.3, eye_open), iris, False)
                ell(px, py, .012, .020 * max(.3, eye_open), OUT, False)
                ell(px - .007, py + .011, .006, .008, (255, 255, 255), False)
            if expr in ("tired", "smug"):
                poly([(cx - .042, eye_y + .06), (cx + .042, eye_y + .06), (cx + .042, eye_y + .004), (cx - .042, eye_y + .004)], skin, False)
                line((cx - .042, eye_y + .004), (cx + .042, eye_y + .004), .016)
            if ch.get("lashes"):
                line((cx + side * .036, eye_y + .01), (cx + side * .06, eye_y + .035), .012)
                line((cx + side * .03, eye_y + .03), (cx + side * .048, eye_y + .06), .012)
        by = eye_y + .088 + brow_raise
        bt = .022
        if expr == "angry":
            line((cx - side * .05, by + .03), (cx + side * .05, by - .015), bt, brow_col)
        elif expr in ("sad", "cry", "scared"):
            line((cx + side * .05, by + .035), (cx - side * .05, by - .005), bt, brow_col)
        elif expr == "confused":
            line((cx - .045, by), (cx + .045, by + .03 * side), bt, brow_col)
        elif expr in ("smug",):
            line((cx - .04, by - .005), (cx + .04, by + .012 * side), bt, brow_col)
        else:
            curve([(cx - .042, by - .008), (cx, by + .012), (cx + .042, by - .004)], bt, brow_col)
    if ch.get("glasses"):
        for side in (-1, 1):
            c = xy(side * ex, eye_y)
            cv2.circle(canvas, c, width(.066), OUT, width(.014), cv2.LINE_AA)
            cv2.circle(alpha, c, width(.066), 255, width(.014), cv2.LINE_AA)
            cv2.ellipse(canvas, c, (width(.04), width(.04)), 0, 200, 250, (255, 255, 255), width(.008), cv2.LINE_AA)
        line((-.022, eye_y + .01), (.022, eye_y + .01), .014)
        line((ex + .066, eye_y), (head_w * .98, eye_y - .005), .012)
        line((-ex - .066, eye_y), (-head_w * .98, eye_y - .005), .012)
    # naak aur blush
    curve([(.006, Y(1.455)), (.028, Y(1.415)), (-.004, Y(1.405))], .015, _d(skin, .45))
    for side in (-1, 1):
        ell(side * .16, Y(1.40), .045, .028, blush, False)
        if ch.get("freckles"):
            for k in range(3):
                ell(side * (.13 + k * .018), Y(1.425 - (k % 2) * .012), .007, .007, _d(skin, .5), False)
    if expr == "cry":
        for side in (-1, 1):
            k = (t * 1.3 + side * .3) % 1.0
            ell(side * .125, Y(1.43 - k * .12), .019, .034, (235, 200, 120), False)
            ell(side * .125, Y(1.43 - k * .12 + .012), .008, .014, (255, 240, 220), False)
    if expr == "embarrassed":
        for side in (-1, 1):
            for k in range(3):
                line((side * (.12 + k * .03), Y(1.44)), (side * (.115 + k * .03), Y(1.39)), .008, (70, 70, 215))
    if expr in ("scared", "shock"):
        ell(head_w * .8, Y(HY + head_h * .55), .016, .03, (240, 215, 140), False)     # paseena

    # beard / moonchh
    beard = ch.get("beard")
    if beard:
        poly([(-.14, Y(1.40)), (-.10, Y(1.22)), (0, Y(1.17)), (.10, Y(1.22)), (.14, Y(1.40)), (.09, Y(1.33)), (0, Y(1.35)), (-.09, Y(1.33))], hair)
        curve([(-.08, Y(1.26)), (0, Y(1.22)), (.08, Y(1.26))], .01, _l(hair, .3))
    if ch.get("mustache"):
        poly([(-.095, Y(1.375)), (-.04, Y(1.395)), (0, Y(1.375)), (.04, Y(1.395)), (.095, Y(1.375)), (.05, Y(1.345)), (0, Y(1.36)), (-.05, Y(1.345))], hair)

    # ---------------------------------------------------------------- munh (viseme + expression)
    visemes = ("closed", "A", "O", "E", "M/B", "F/V", "grin", "shock", "smile", "frown")
    viseme = st.get("viseme")
    if viseme not in visemes:
        idx = int((t * 7.0 + seed * 1.7) % len(visemes))
        viseme = visemes[idx] if talk > .05 else "closed"
    my = Y(1.345)
    mc = (40, 25, 90)
    tongue = (110, 110, 235)

    def mouth_open(rx, ry, teeth=True):
        ell(0, my - ry * .15, rx, ry, mc, True, OW * .6)
        if ry > .02:
            ell(0, my - ry * .62, rx * .62, ry * .38, tongue, False)
            if teeth:
                poly([(-rx * .78, my + ry * .5), (rx * .78, my + ry * .5), (rx * .70, my + ry * .1), (-rx * .70, my + ry * .1)], (250, 250, 248), False)

    if expr in ("grin", "laugh") or viseme == "grin":
        k = .02 * math.sin(t * 14) if expr == "laugh" else 0
        poly([(-.115, my + .035), (.115, my + .035), (.095, my - .05 - k), (-.095, my - .05 - k)], (252, 252, 250), True, OW * .6)
        line((-.11, my - .005), (.11, my - .005), .008, (190, 190, 190))
        for sd in (-1, 1):
            curve([(sd * .125, my + .03), (sd * .135, my + .01)], .01)
    elif expr == "shock" or viseme == "shock":
        mouth_open(.05, .075, False)
    elif expr in ("sad", "cry") or viseme == "frown":
        curve([(-.07, my - .02), (-.035, my - .04), (0, my - .048), (.035, my - .04), (.07, my - .02)], .018)
    elif expr == "scared":
        mouth_open(.07, .035)
    elif viseme == "A":
        mouth_open(.058, .055)
    elif viseme == "O":
        mouth_open(.032, .052, False)
    elif viseme == "E":
        mouth_open(.066, .022)
    elif viseme in ("M/B", "F/V") and talk > .05:
        line((-.055, my), (.055, my), .016)
    elif talk > .05:
        mouth_open(.048, .014 + .04 * talk)
    elif expr in ("angry", "tired"):
        line((-.06, my), (.06, my), .018)
    elif expr == "embarrassed":
        curve([(-.06, my), (-.03, my + .01), (0, my - .005), (.03, my + .01), (.06, my)], .015)
    elif expr == "smug":
        curve([(-.06, my - .005), (.0, my - .02), (.06, my + .02), (.085, my + .04)], .016)
    elif expr in ("happy", "smile"):
        curve([(-.085, my + .028), (-.04, my - .012), (0, my - .026), (.04, my - .012), (.085, my + .028)], .018)
    elif expr == "confused":
        curve([(-.06, my), (-.02, my + .012), (.02, my - .012), (.06, my + .004)], .016)
    else:
        curve([(-.06, my + .004), (0, my - .01), (.06, my + .004)], .016)

    # ---------------------------------------------------------------- rim light
    if power > .08:
        rim = tuple(int(min(255, x * .55 + 125)) for x in tint)
        cv2.ellipse(canvas, xy(0, Y(HY)), (width(head_w * 1.03), width(head_h * 1.03)), 0, -65, 65, rim, width(.009), cv2.LINE_AA)
        cv2.ellipse(alpha, xy(0, Y(HY)), (width(head_w * 1.03), width(head_h * 1.03)), 0, -65, 65, 255, width(.009), cv2.LINE_AA)

    # ---------------------------------------------------------------- ROI ko frame par composite
    small = cv2.resize(canvas, (rw, rh), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(alpha, (rw, rh), interpolation=cv2.INTER_AREA)
    x0, y0 = max(rx0, 0), max(ry0, 0)
    x1, y1 = min(rx0 + rw, W), min(ry0 + rh, H)
    if x1 <= x0 or y1 <= y0:
        return
    sm = small[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0].astype(np.float32)
    a = (mask[y0 - ry0:y1 - ry0, x0 - rx0:x1 - rx0].astype(np.float32) / 255.0)[..., None]
    roi = frame[y0:y1, x0:x1].astype(np.float32)
    frame[y0:y1, x0:x1] = np.clip(roi * (1.0 - a) + sm * a, 0, 255).astype(np.uint8)


def draw_char(r, P_world, ch, st):
    """Required public signature."""
    _draw(r, P_world, ch, st)
