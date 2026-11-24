
"""Stable character renderer. Comments Roman Urdu mein hain."""

import math
import cv2
import numpy as np

OUT = (18, 15, 22)
SHADE = (0.78, 0.68, 0.72)
BOIL_AMPLITUDE = 0.0
SS = 3

EXPRS = (
    "neutral", "smile", "grin", "laugh", "happy", "sad", "cry",
    "angry", "shock", "scared", "smug", "tired", "confused",
    "embarrassed"
)

HAIRSTYLES = (
    "cap", "bowl", "messy", "side_part", "bald_sides", "ponytail",
    "bun", "curly", "long", "spiky", "hijab", "topi"
)

OUTFITS = (
    "suit", "shirt", "hoodie", "tee", "dress", "kurta",
    "lab_coat", "uniform"
)


def clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, float(v)))


def smooth(a, b, t):
    """Smooth interpolation."""
    x = clamp((t - a) / max(1e-8, b - a))
    return x * x * (3.0 - 2.0 * x)


def light_for(r, Pw):
    """Qareebi light se halka rang aur direction hasil karo."""
    lights = getattr(r, "lights", None) or []
    if not lights:
        return (255, 255, 255), 0.0, np.array([0.5, 0.8])

    best = None
    for light in lights:
        pos = np.asarray(light.get("pos", Pw), dtype=np.float64)
        dist = float(np.linalg.norm(pos - Pw))
        power = float(light.get("intensity", 1.0))
        power /= 1.0 + 0.12 * dist * dist
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
    """Time-based arms ki smooth local pose."""
    s = math.sin
    if kind == "wave":
        return {"l": (-.30, .82), "r": (.38 + .045*s(t*7), 1.40 + .06*s(t*7))}
    if kind == "point":
        return {"l": (-.30, .82), "r": (.60, 1.22)}
    if kind == "shrug":
        return {"l": (-.42, 1.04 + .025*s(t*2)), "r": (.42, 1.04 + .025*s(t*2))}
    if kind in ("cheer", "clap"):
        y = 1.45 + .035 * abs(s(t*5))
        return {"l": (-.35, y), "r": (.35, y)}
    if kind == "facepalm":
        return {"l": (-.30, .80), "r": (.12, 1.48)}
    if kind == "sad":
        return {"l": (-.23, .67), "r": (.23, .67)}
    if kind == "type":
        return {"l": (-.18 + .012*s(t*8), 1.02), "r": (.18 - .012*s(t*8), 1.02)}
    if kind == "talk":
        return {"l": (-.30, .82), "r": (.34 + .055*s(t*3)*talk, .98 + .11*s(t*2.5)*talk)}
    return {"l": (-.30, .80), "r": (.30, .80)}


def auto_arms(st, expr, talk):
    """Gesture na di ho to expression ke mutabiq pose chuno."""
    kind = st.get("gesture", "idle")
    if kind != "idle":
        return gesture(kind, st.get("t", 0.0), talk)
    if st.get("sit"):
        return gesture("type" if talk < .05 else "talk", st.get("t", 0.0), talk)
    if expr in ("shock", "scared"):
        return gesture("cheer", st.get("t", 0.0), talk)
    if expr in ("sad", "cry"):
        return gesture("sad", st.get("t", 0.0), talk)
    if talk > .05:
        return gesture("talk", st.get("t", 0.0), talk)
    return gesture("idle", st.get("t", 0.0), talk)


def _draw(r, Pw, ch, st):
    """Supersampled transparent character canvas par drawing."""
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
    scale = float(r.c.f) / depth

    # 3x canvas subpixel position ko stable rakhta hai.
    canvas = np.zeros((H*SS, W*SS, 3), dtype=np.uint8)
    alpha = np.zeros((H*SS, W*SS), dtype=np.uint8)

    facing = st.get("facing", "front")
    face_sign = -1.0 if facing == "left" else 1.0
    face_scale = .86 if facing == "three_quarter" else 1.0

    def xy(x, y):
        # Left-facing mein character ko mirror aur three-quarter mein halka compress karo.
        local_x = float(x) * face_sign * face_scale
        return (
            int(round((u + local_x*scale) * SS)),
            int(round((v - y*scale) * SS))
        )

    def width(value):
        return max(1, int(round(abs(value*scale)*SS)))

    def color(value, factor=1.0):
        base = np.asarray(value, dtype=np.float32)
        tint, power, direction = light_for(r, Pw)
        light = np.asarray(tint, dtype=np.float32)
        lit = base*(1-power) + base*(light/255.0)*power
        return tuple(int(x) for x in np.clip(lit*factor, 0, 255))

    skin = ch.get("skin", (150, 190, 230))
    hair = ch.get("hair", (35, 30, 40))
    shirt = ch.get("shirt", (80, 150, 210))
    seed = int(ch.get("seed", 1))

    def shape_ellipse(cx, cy, rx, ry, fill, outline=True):
        center = xy(cx, cy)
        axes = (width(rx), width(ry))
        cv2.ellipse(canvas, center, axes, 0, 0, 360, color(fill), -1, cv2.LINE_AA)
        cv2.ellipse(alpha, center, axes, 0, 0, 360, 255, -1, cv2.LINE_AA)
        if outline:
            cv2.ellipse(canvas, center, axes, 0, 0, 360, OUT, width(.018), cv2.LINE_AA)
            cv2.ellipse(alpha, center, axes, 0, 0, 360, 255, width(.018), cv2.LINE_AA)

    def polygon(points, fill, outline=True):
        p = np.asarray([xy(x, y) for x, y in points], dtype=np.int32)
        cv2.fillPoly(canvas, [p], color(fill), cv2.LINE_AA)
        cv2.fillPoly(alpha, [p], 255, cv2.LINE_AA)
        if outline:
            cv2.polylines(canvas, [p], True, OUT, width(.018), cv2.LINE_AA)
            cv2.polylines(alpha, [p], True, 255, width(.018), cv2.LINE_AA)

    def line(a, b, thickness=.025, col=OUT):
        cv2.line(canvas, xy(*a), xy(*b), col, width(thickness), cv2.LINE_AA)
        cv2.line(alpha, xy(*a), xy(*b), 255, width(thickness), cv2.LINE_AA)

    def curve(points, thickness=.025, col=OUT):
        p = np.asarray([xy(x, y) for x, y in points], np.int32)
        cv2.polylines(canvas, [p], False, col, width(thickness), cv2.LINE_AA)
        cv2.polylines(alpha, [p], False, 255, width(thickness), cv2.LINE_AA)

    # Animation sirf continuous time se; koi random offset nahi.
    sit = bool(st.get("sit", False))
    walk = bool(st.get("walk", False))
    phase = float(st.get("ph", t*2.0))
    bob = .014*abs(math.sin(phase)) if walk else .009*math.sin(t*2.4)
    base = -.22 if sit else 0.0
    Y = lambda y: y + bob + base

    # Body proportions har character ke data se badal sakte hain.
    head_w = float(ch.get("head_w", .215))
    head_h = float(ch.get("head_h", .255))
    body_w = float(ch.get("body_w", .20))
    body_h = float(ch.get("body_h", .27))
    leg_w = float(ch.get("limb_w", .025))

    # Pehle peechhe ke baal.
    style = ch.get("style", "messy")
    if style in ("long", "ponytail"):
        polygon([(-head_w, Y(1.48)), (-head_w*1.12, Y(1.05)),
                 (head_w*1.1, Y(1.05)), (head_w, Y(1.48))], hair)
    if style == "ponytail":
        shape_ellipse(head_w*1.25, Y(1.38), .09, .13, hair)

    # Pair: walk cycle mein planted contact aur passing pose.
    if not sit:
        for side in (-1, 1):
            p = phase + (math.pi if side > 0 else 0)
            lift = max(0.0, math.sin(p)) * (.12 if walk else 0.0)
            hip = (side*.075, Y(.80))
            knee = (side*.095, Y(.40 + lift*.5))
            foot = (side*.12, Y(lift))
            line(hip, knee, leg_w)
            line(knee, foot, leg_w)
            shape_ellipse(foot[0]+side*.025, foot[1]-.015, .075, .035, ch.get("shoe", (35,30,35)), False)

    # Body aur kapron ki shading.
    shape_ellipse(0, Y(.99), body_w, body_h, shirt)
    shape_ellipse(-body_w*.48, Y(.99), body_w*.30, body_h*.86,
                  tuple(int(x*v) for x, v in zip(shirt, SHADE)), False)

    outfit = ch.get("outfit", "tee")
    if outfit == "suit":
        polygon([(-.07,Y(1.22)), (0,Y(1.08)), (.07,Y(1.22)), (.13,Y(.78)),
                 (-.13,Y(.78))], (225,225,225))
        if ch.get("tie"):
            polygon([(-.025,Y(1.20)), (.025,Y(1.20)), (.038,Y(1.02)),
                     (0,Y(.84)), (-.038,Y(1.02))], ch.get("tie_col",(40,40,200)))
    elif outfit == "hoodie":
        shape_ellipse(0,Y(1.20),.11,.10,tuple(int(x*.75) for x in shirt))
        line((-.045,Y(1.23)),(-.035,Y(1.02)),.009)
        line((.045,Y(1.23)),(.035,Y(1.02)),.009)
    elif outfit == "dress":
        polygon([(-.14,Y(1.15)),(.14,Y(1.15)),(.24,Y(.72)),(-.24,Y(.72))],shirt)
    elif outfit == "kurta":
        polygon([(-.13,Y(1.2)),(.13,Y(1.2)),(.17,Y(.72)),(-.17,Y(.72))],shirt)
        line((0,Y(1.2)),(0,Y(.96)),.009)
    elif outfit == "lab_coat":
        polygon([(-.13,Y(1.22)),(-.20,Y(.76)),(-.08,Y(.76)),(0,Y(1.1)),
                 (.08,Y(.76)),(.20,Y(.76)),(.13,Y(1.22))],(225,230,235))
    elif outfit == "uniform":
        line((-.08,Y(1.17)),(.08,Y(1.17)),.018,(245,245,245))

    # Arms aur haath.
    arms = auto_arms(st, st.get("expr","smile"), clamp(st.get("talk",0)))
    arms.update(st.get("arms") or {})
    for side, key in ((-1,"l"),(1,"r")):
        shoulder = (side*body_w*.88,Y(1.14))
        hand = (float(arms[key][0]),Y(float(arms[key][1])))
        elbow = ((shoulder[0]+hand[0])*.5+side*.045,
                 (shoulder[1]+hand[1])*.5-.025)
        line(shoulder,elbow,leg_w*1.25)
        line(elbow,hand,leg_w*1.25)
        shape_ellipse(hand[0],hand[1],.045,.045,skin)

    # Accessories.
    props = set(st.get("props") or [])
    if "paper" in props:
        props.add("papers")
    if st.get("phone"):
        props.add("phone")
    if st.get("key"):
        props.add("key")
    hand = arms["r"]
    if "phone" in props:
        polygon([(hand[0]-.04,Y(hand[1]+.07)),(hand[0]+.04,Y(hand[1]+.07)),
                 (hand[0]+.04,Y(hand[1]-.07)),(hand[0]-.04,Y(hand[1]-.07))],(45,45,50))
    if "mug" in props or "coffee" in props:
        shape_ellipse(hand[0],Y(hand[1]),.055,.07,(240,240,235))
        shape_ellipse(hand[0]+.065,Y(hand[1]),.035,.04,(240,240,235),False)
    if "papers" in props:
        polygon([(hand[0]-.08,Y(hand[1]+.10)),(hand[0]+.08,Y(hand[1]+.10)),
                 (hand[0]+.08,Y(hand[1]-.10)),(hand[0]-.08,Y(hand[1]-.10))],(245,245,240))
    if "key" in props:
        shape_ellipse(hand[0],Y(hand[1]),.035,.035,(180,190,205))
    if "laptop" in props:
        polygon([(-.20,Y(.91)),(.20,Y(.91)),(.17,Y(.83)),(-.17,Y(.83))],(75,85,100))
        polygon([(-.16,Y(.94)),(.16,Y(.94)),(.16,Y(1.12)),(-.16,Y(1.12))],(45,55,70))
        polygon([(-.13,Y(.97)),(.13,Y(.97)),(.13,Y(1.09)),(-.13,Y(1.09))],(115,175,205),False)
    if "bag" in props:
        curve([(-.13,Y(1.20)),(-.24,Y(1.02)),(-.22,Y(.78))],.025,(35,30,40))
        polygon([(-.29,Y(.91)),(-.12,Y(.91)),(-.10,Y(.68)),(-.31,Y(.68))],(125,85,55))
    if "umbrella" in props:
        line((hand[0],Y(hand[1])),(hand[0],Y(hand[1]+.42)),.018,(45,45,55))
        polygon([(hand[0]-.18,Y(hand[1]+.40)),(hand[0],Y(hand[1]+.53)),
                 (hand[0]+.18,Y(hand[1]+.40))],(75,105,185))
    if "food" in props:
        shape_ellipse(hand[0],Y(hand[1]),.075,.055,(75,155,205))
        shape_ellipse(hand[0],Y(hand[1]+.025),.060,.035,(145,195,225),False)

    # Gardan aur kaan.
    shape_ellipse(0,Y(1.30),.075,.09,skin,False)
    for side in (-1,1):
        shape_ellipse(side*head_w*.94,Y(1.47),.045,.065,skin)

    # Sar ka stable anda shape.
    shape_ellipse(0,Y(1.48),head_w,head_h,skin)

    # Hair styles: 12 alag silhouettes.
    if style == "cap":
        shape_ellipse(0,Y(1.70),head_w*1.04,.085,hair)
        polygon([(-head_w,Y(1.69)),(head_w,Y(1.69)),(head_w*1.25,Y(1.63)),
                 (-head_w*1.25,Y(1.63))],hair)
    elif style == "topi":
        shape_ellipse(0,Y(1.73),head_w*.85,.075,(235,235,225))
    elif style == "hijab":
        polygon([(-head_w*1.1,Y(1.65)),(-head_w*1.2,Y(1.40)),
                 (-head_w*.9,Y(1.15)),(head_w*.9,Y(1.15)),
                 (head_w*1.2,Y(1.40)),(head_w*1.1,Y(1.65))],hair)
    elif style == "bald_sides":
        for side in (-1,1):
            shape_ellipse(side*head_w*.92,Y(1.55),.045,.10,hair)
    elif style == "bun":
        shape_ellipse(0,Y(1.48+head_h*1.10),.085,.075,hair)
        polygon([(-head_w,Y(1.55)),(-head_w*.9,Y(1.72)),(head_w*.9,Y(1.72)),
                 (head_w,Y(1.55))],hair)
    elif style == "curly":
        for i in range(9):
            x = (i-4)*head_w*.22
            shape_ellipse(x,Y(1.68+.035*math.sin(i*2)),.055,.065,hair,False)
    elif style == "ponytail":
        shape_ellipse(0,Y(1.70),head_w,.075,hair)
    elif style == "long":
        polygon([(-head_w,Y(1.65)),(-head_w*1.15,Y(1.18)),
                 (-head_w*.70,Y(1.10)),(head_w*.70,Y(1.10)),
                 (head_w*1.15,Y(1.18)),(head_w,Y(1.65))],hair)
    elif style == "spiky":
        for i in range(5):
            x=(i-2)*head_w*.37
            polygon([(x-.05,Y(1.65)),(x,Y(1.86+.025*(i%2))),
                     (x+.055,Y(1.65))],hair)
    elif style in ("messy","side_part","bowl"):
        polygon([(-head_w*1.05,Y(1.55)),(-head_w,Y(1.75)),
                 (-head_w*.4,Y(1.78)),(head_w*.25,Y(1.72)),
                 (head_w,Y(1.70)),(head_w*1.05,Y(1.53))],hair)
        if style == "side_part":
            line((-.02,Y(1.76)),(.09,Y(1.68)),.012)
        if style == "bowl":
            curve([(-head_w,Y(1.60)),(-head_w*.5,Y(1.55)),(0,Y(1.54)),
                   (head_w*.5,Y(1.55)),(head_w,Y(1.60))],.035,hair)

    # Face expression blend aur deterministic blink.
    expr = st.get("expr","smile")
    if expr not in EXPRS:
        expr = "smile"
    blend = clamp(st.get("expr_blend",1.0))
    look = st.get("look",(0,0))
    lx,ly = float(look[0]),float(look[1])
    eye_y = Y(1.49)
    blink_period = 3.1 + (seed%4)*.43
    blink_phase = (t + seed*.37) % blink_period
    blink = smooth(0.0,.045,blink_phase) * (1-smooth(.11,.17,blink_phase))
    eye_open = 1.0-blink

    brow_raise = .0
    if expr in ("shock","scared"):
        brow_raise=.045
    elif expr in ("angry","confused"):
        brow_raise=-.015

    for side in (-1,1):
        ex = side*.078
        if expr == "happy":
            curve([(ex-.035,eye_y),(ex,eye_y+.025),(ex+.035,eye_y)],.014)
        else:
            shape_ellipse(ex,eye_y,.030,.040*max(.08,eye_open),(250,250,245),False)
            shape_ellipse(ex+clamp(lx,-1,1)*.018,
                          eye_y+clamp(ly,-1,1)*.018,.013,.020*max(.08,eye_open),OUT,False)

        by=eye_y+.075+brow_raise
        if expr == "angry":
            line((ex-.04,by+.025),(ex+.04,by-.01),.016)
        elif expr == "sad" or expr == "cry":
            line((ex-.04,by-.01),(ex+.04,by+.025),.014)
        elif expr == "confused":
            line((ex-.04,by),(ex+.04,by+.025*side),.014)
        else:
            line((ex-.035,by),(ex+.035,by),.012)

    if ch.get("glasses"):
        for side in (-1,1):
            cv2.circle(canvas,xy(side*.078,eye_y),width(.063),OUT,width(.012),cv2.LINE_AA)
            cv2.circle(alpha,xy(side*.078,eye_y),width(.063),255,width(.012),cv2.LINE_AA)
        line((-.015,eye_y),(.015,eye_y),.012)

    if expr == "cry":
        shape_ellipse(.105,Y(1.40),.018,.038,(190,210,250),False)
        shape_ellipse(-.105,Y(1.40),.018,.038,(190,210,250),False)

    if ch.get("mustache"):
        polygon([(-.075,Y(1.38)),(0,Y(1.35)),(.075,Y(1.38)),
                 (.045,Y(1.34)),(-.045,Y(1.34))],hair)

    # Munh ki expressions aur viseme amplitude.
    talk = clamp(st.get("talk",0))
    visemes = ("closed","A","O","E","M/B","F/V","grin","shock","smile","frown")
    viseme = st.get("viseme")
    if viseme not in visemes:
        idx = int((t*7.0+seed*1.7) % len(visemes))
        viseme = visemes[idx] if talk > .05 else "closed"

    my=Y(1.355)
    if expr in ("grin","laugh") or viseme == "grin":
        polygon([(-.105,my+.035),(.105,my+.035),(.09,my-.045),
                 (-.09,my-.045)],(250,250,245))
        line((-.105,my),(.105,my),.008)
    elif expr == "shock" or viseme == "shock":
        shape_ellipse(0,my,.055,.075,(45,25,55))
    elif expr == "sad" or expr == "cry" or viseme == "frown":
        curve([(-.065,my-.015),(0,my-.045),(.065,my-.015)],.014)
    elif viseme == "A":
        shape_ellipse(0,my,.055,.050,(45,25,55),True)
    elif viseme == "O":
        shape_ellipse(0,my,.030,.050,(45,25,55),True)
    elif viseme == "E":
        shape_ellipse(0,my,.060,.018,(45,25,55),True)
    elif talk > .05:
        shape_ellipse(0,my,.045,.012+.035*talk,(45,25,55),True)
    elif expr in ("angry","embarrassed","tired"):
        line((-.065,my),(.065,my),.014)
    else:
        curve([(-.07,my+.025),(0,my-.025),(.07,my+.025)],.015)

    # Character ke gird halki rim light.
    tint,power,direction = light_for(r,Pw)
    if power > .08:
        rim = tuple(int(min(255,x*.55+125)) for x in tint)
        cv2.ellipse(canvas,xy(0,Y(1.48)),
                    (width(head_w*1.03),width(head_h*1.03)),
                    0,-65,65,rim,width(.009),cv2.LINE_AA)
        cv2.ellipse(alpha,xy(0,Y(1.48)),
                    (width(head_w*1.03),width(head_h*1.03)),
                    0,-65,65,255,width(.009),cv2.LINE_AA)

    # High resolution image aur alpha ko final size par smooth karo.
    small = cv2.resize(canvas,(W,H),interpolation=cv2.INTER_AREA)
    mask = cv2.resize(alpha,(W,H),interpolation=cv2.INTER_AREA)
    a = mask.astype(np.float32)[...,None]/255.0
    r.fr[:] = np.clip(
        r.fr.astype(np.float32)*(1.0-a)+small.astype(np.float32)*a,
        0,255
    ).astype(np.uint8)


def draw_char(r, P_world, ch, st):
    """Required public signature."""
    _draw(r, P_world, ch, st)
