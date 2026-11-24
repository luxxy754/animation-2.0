
"""Toon 3D prop library. Comments Roman Urdu mein hain."""

import math
import numpy as np

CATALOG = {
    # Indoor
    "desk": ("box", (1.8, .8, 1.0), (145, 115, 85)),
    "chair": ("chair", (.7, 1.0, .7), (65, 85, 125)),
    "sofa": ("box", (2.0, .9, .9), (95, 115, 155)),
    "bed": ("box", (2.0, .5, 2.0), (190, 195, 205)),
    "table": ("box", (1.5, .8, 1.2), (160, 125, 85)),
    "bookshelf": ("shelf", (1.2, 2.0, .4), (125, 85, 55)),
    "tv": ("screen", (1.3, .9, .15), (35, 35, 42)),
    "fridge": ("box", (.8, 1.9, .7), (205, 215, 220)),
    "kitchen_counter": ("box", (2.0, .9, .8), (155, 125, 100)),
    "window": ("window", (1.5, 1.3, .12), (125, 190, 220)),
    "door": ("door", (1.0, 2.2, .12), (115, 75, 45)),
    "whiteboard": ("screen", (1.8, 1.2, .10), (235, 238, 235)),
    "school_desk": ("desk", (1.1, .8, .8), (160, 120, 75)),
    "lamp": ("lamp", (.35, 1.5, .35), (230, 195, 100)),
    "rug": ("rug", (2.0, .035, 1.5), (115, 85, 145)),
    "poster": ("screen", (.9, 1.2, .06), (235, 215, 165)),
    "plant": ("plant", (.65, 1.3, .65), (65, 145, 75)),
    "monitor": ("screen", (.8, .65, .15), (45, 50, 65)),
    "keyboard": ("box", (.6, .07, .22), (90, 95, 105)),
    "mug": ("cylinder", (.22, .3, .22), (65, 135, 190)),
    "paper": ("box", (.45, .025, .6), (245, 242, 225)),
    "filing_cabinet": ("cabinet", (.8, 1.5, .65), (125, 140, 155)),
    "water_cooler": ("cooler", (.55, 1.6, .55), (170, 205, 225)),
    "printer": ("box", (.8, .65, .65), (175, 180, 185)),
    "coat_rack": ("rack", (.5, 1.8, .5), (85, 75, 65)),
    "clock": ("clock", (.55, .55, .12), (240, 235, 215)),
    "vent": ("box", (1.0, .12, .5), (145, 155, 165)),
    "books": ("books", (.65, .35, .3), (180, 80, 65)),
    "cabinet": ("cabinet", (1.0, 1.3, .6), (145, 120, 95)),
    "sofa_chair": ("box", (.9, .9, .8), (110, 135, 165)),

    # Outdoor
    "tree": ("tree", (1.3, 3.5, 1.3), (55, 145, 65)),
    "bush": ("tree", (.9, .8, .9), (65, 155, 75)),
    "bench": ("bench", (1.5, .9, .6), (145, 105, 65)),
    "car": ("car", (1.9, 1.2, 3.6), (65, 105, 195)),
    "bus": ("car", (2.3, 2.3, 6.0), (225, 175, 55)),
    "bike": ("bike", (1.0, 1.3, 1.8), (65, 65, 75)),
    "street_lamp": ("lamp", (.4, 3.2, .4), (85, 85, 100)),
    "building_block": ("building", (3.0, 5.0, 3.0), (175, 165, 155)),
    "shop_front": ("shop", (2.5, 2.8, 1.5), (205, 145, 75)),
    "fence": ("fence", (2.0, 1.3, .15), (145, 115, 80)),
    "rock": ("rock", (.9, .65, .8), (125, 125, 130)),
    "fountain": ("fountain", (1.5, 1.4, 1.5), (155, 175, 190)),
    "swing": ("swing", (1.5, 2.0, .8), (75, 105, 165)),
    "slide": ("slide", (1.2, 1.8, 2.5), (215, 105, 65)),
    "tent": ("tent", (2.0, 1.7, 2.0), (185, 105, 75)),
    "campfire": ("fire", (.8, .9, .8), (225, 105, 35)),
    "boat": ("boat", (2.0, .9, 3.0), (105, 135, 175)),
    "mountain": ("mountain", (4.0, 3.0, 3.0), (115, 135, 155)),
    "cloud": ("cloud", (2.0, .7, 1.0), (235, 240, 245)),
    "sign": ("screen", (1.2, .9, .12), (225, 195, 85)),
    "traffic_light": ("traffic", (.45, 1.8, .45), (55, 55, 60)),
    "bridge": ("bridge", (3.5, 1.0, 2.0), (125, 125, 135)),
    "chai_stall": ("shop", (2.0, 2.2, 1.5), (155, 105, 65)),
    "bus_stop": ("shelter", (2.0, 2.2, 1.0), (115, 145, 175)),
    "atm": ("screen", (.8, 1.5, .5), (65, 105, 145)),
    "road_sign": ("screen", (.9, 1.4, .12), (55, 145, 75)),
    "flower": ("plant", (.35, .6, .35), (220, 105, 155)),
    "bin": ("cylinder", (.45, .9, .45), (95, 115, 125)),
    "hydrant": ("cylinder", (.45, .7, .45), (45, 55, 210)),
    "mailbox": ("box", (.6, 1.1, .5), (55, 105, 185)),
    "pillar": ("cylinder", (.65, 2.2, .65), (185, 180, 165)),
    "water_tank": ("cylinder", (1.2, 1.5, 1.2), (105, 155, 185)),
    "playground": ("swing", (2.0, 2.0, 1.5), (195, 105, 75)),
    "umbrella": ("umbrella", (1.5, 1.5, 1.5), (205, 75, 85)),
    "airplane": ("airplane", (3.0, .7, 4.0), (220, 225, 230)),
    "helicopter": ("airplane", (2.0, 1.0, 2.0), (75, 125, 95)),
    "palm_tree": ("tree", (1.5, 3.8, 1.5), (65, 145, 85)),
    "rock_large": ("rock", (1.7, 1.1, 1.5), (105, 110, 120)),
    "picnic_table": ("desk", (1.8, .9, 1.5), (155, 105, 65)),
    "street_bin": ("cylinder", (.5, 1.0, .5), (75, 105, 115)),
}

PROP_NAMES = tuple(CATALOG.keys())


def validate_color(color, fallback):
    """RGB rang ko safe 0-255 range mein rakho."""
    try:
        if len(color) != 3:
            return fallback
        return tuple(max(0, min(255, int(v))) for v in color)
    except (TypeError, ValueError, OverflowError):
        return fallback


def _box(r, x, y, z, size, col):
    """Center aur size se box draw karo."""
    sx, sy, sz = size
    r.box(x-sx/2, x+sx/2, y-sy/2, y+sy/2,
          z-sz/2, z+sz/2, col, lw=2)


def _cyl(r, x, y, z, size, col):
    sx, sy, sz = size
    radius = max(.025, min(sx, sz) * .5)
    r.cyl(x, z, y-sy/2, y+sy/2, radius, col, lw=2)


def _part(r, kind, x, y, z, size, col):
    sx, sy, sz = size
    if kind == "box":
        _box(r, x, y, z, size, col)
    elif kind == "cylinder":
        _cyl(r, x, y, z, size, col)
    elif kind == "sphere":
        r.disc((x, y, z), max(sx, sy, sz)*.5, col, lw=2)
    elif kind == "cone":
        _cyl(r, x, y, z, (sx*.7, sy, sz*.7), col)


def _recipe(r, kind, x, z, w, h, d, col):
    """Har prop ko chhote primitive parts se assemble karo."""
    dark = tuple(int(v*.68) for v in col)
    light = tuple(min(255, int(v*1.2)) for v in col)
    floor = 0.0

    if kind in ("box", "screen", "door", "window", "cabinet"):
        _box(r, x, h*.5, z, (w, h, d), col)
        if kind == "screen":
            _box(r, x, h*.56, z-d*.52, (w*.84, h*.68, .025), light)
        elif kind == "door":
            _box(r, x+w*.35, h*.45, z-d*.55, (.045,.045,.035), light)
        elif kind == "window":
            _box(r, x, h*.5, z-d*.55, (w*.88,h*.85,.025), (125,195,225))
        elif kind == "cabinet":
            for yy in (.28,.58,.88):
                _box(r,x, h*yy,z-d*.55,(w*.85,.025,.025),dark)

    elif kind == "desk":
        _box(r,x,h*.90,z,(w,h*.12,d),col)
        for dx in (-.42,.42):
            _box(r,x+w*dx,h*.45,z,(w*.07,h*.85,d*.08),dark)

    elif kind == "chair":
        _box(r,x,h*.52,z,(w*.75,h*.10,d*.72),col)
        _box(r,x,h*.78,z+d*.32,(w*.78,h*.43,d*.10),dark)
        for dx in (-.36,.36):
            for dz in (-.30,.30):
                _box(r,x+w*dx,h*.24,z+d*dz,(.045,h*.48,.045),dark)

    elif kind == "shelf":
        _box(r,x,h*.5,z,(w*.10,h,d),dark)
        for yy in (.25,.5,.75):
            _box(r,x,h*yy,z,(w,h*.035,d),light)
            for j in range(3):
                bx=x-w*.32+j*w*.32
                _box(r,bx,h*(yy+.11),z,(w*.18,h*.18,d*.75),
                     ((190,75,65),(55,115,175),(65,145,95))[j])

    elif kind in ("plant","tree","rock","mountain","cloud"):
        if kind == "plant":
            _cyl(r,x,h*.13,z,(w*.35,h*.25,d*.35),dark)
            _box(r,x,h*.55,z,(w*.08,h*.7,d*.08),dark)
            for dx,dz in ((-.2,0),(.2,0),(0,.2),(0,-.2)):
                r.disc((x+dx*w,h*.72,z+dz*d),w*.20,col)
        elif kind in ("tree","mountain"):
            _cyl(r,x,h*.28,z,(w*.23,h*.55,d*.23),dark)
            for dx,dz,yy in ((0,0,.78),(-.22,0,.65),(.22,0,.65),(0,.2,.62)):
                r.disc((x+dx*w,h*yy,z+dz*d),w*.30,col)
        else:
            r.disc((x,h*.5,z),max(w,d)*.38,col)

    elif kind in ("lamp","rack"):
        _cyl(r,x,h*.48,z,(w*.12,h*.95,d*.12),dark)
        if kind == "lamp":
            r.disc((x,h*.96,z),w*.35,light)
        else:
            _box(r,x,h*.94,z,(w,h*.04,d*.04),col)

    elif kind in ("bench","sofa"):
        _box(r,x,h*.55,z,(w*.95,h*.12,d*.8),col)
        _box(r,x,h*.85,z+d*.32,(w*.95,h*.35,d*.12),dark)
        for dx in (-.4,.4):
            _box(r,x+w*dx,h*.25,z,(w*.06,h*.5,d*.08),dark)

    elif kind == "car":
        _box(r,x,h*.38,z,(w,h*.36,d),col)
        _box(r,x,h*.72,z,(w*.76,h*.36,d*.48),light)
        for dz in (-.34,.34):
            for dx in (-.52,.52):
                r.disc((x+w*dx,h*.18,z+d*dz),h*.16,dark)

    elif kind == "bike":
        for dz in (-.38,.38):
            r.disc((x,h*.18,z+d*dz),h*.19,dark)
        _box(r,x,h*.48,z,(w*.09,h*.08,d*.55),col)

    elif kind in ("fountain","cooler"):
        _cyl(r,x,h*.18,z,(w,h*.35,d),dark)
        _cyl(r,x,h*.58,z,(w*.7,h*.45,d*.7),col)
        _cyl(r,x,h*.90,z,(w*.4,h*.25,d*.4),light)

    elif kind in ("fire","traffic"):
        _box(r,x,h*.2,z,(w*.55,h*.25,d*.55),dark)
        if kind == "fire":
            r.disc((x,h*.62,z),w*.25,(55,115,235))
            r.disc((x,h*.52,z),w*.16,(45,185,245))
        else:
            for yy,cc in ((.75,(45,45,220)),(.55,(40,180,240)),(.35,(55,190,65))):
                r.disc((x,h*yy,z),w*.12,cc)

    elif kind in ("shop","building"):
        _box(r,x,h*.5,z,(w,h,d),col)
        for dx in (-.3,.3):
            for yy in (.4,.7):
                _box(r,x+w*dx,h*yy,z-d*.52,(w*.2,h*.16,.025),(105,175,210))

    elif kind in ("fence","bridge","shelter"):
        _box(r,x,h*.9,z,(w,h*.08,d*.10),col)
        for i in range(5):
            xx=x-w*.45+i*w*.225
            _box(r,xx,h*.48,z,(w*.035,h*.9,d*.10),dark)

    elif kind == "boat":
        _box(r,x,h*.35,z,(w*.9,h*.3,d),col)
        _box(r,x,h*.7,z,(w*.05,h*.7,d*.04),dark)

    elif kind == "swing":
        for dx in (-.4,.4):
            _box(r,x+w*dx,h*.5,z,(.06,h,d*.06),dark)
        _box(r,x,h*.08,z,(w*.6,.09,d*.4),col)

    elif kind == "slide":
        _box(r,x,h*.1,z,(w*.6,.1,d*.5),dark)
        _box(r,x,h*.55,z,(w*.45,h*.08,d),col)

    elif kind == "tent":
        _box(r,x,h*.25,z,(w*.8,h*.5,d*.8),col)
        _box(r,x,h*.55,z,(w*.65,h*.5,d*.65),light)

    elif kind == "umbrella":
        r.disc((x,h*.85,z),w*.5,col)
        _cyl(r,x,h*.43,z,(.06,h*.8,.06),dark)

    elif kind == "airplane":
        _box(r,x,h*.5,z,(w*.18,h*.18,d),col)
        _box(r,x,h*.5,z,(w,h*.08,d*.25),light)
        _box(r,x,h*.5,z-d*.4,(w*.45,h*.07,d*.15),dark)

    elif kind == "books":
        for i in range(4):
            _box(r,x-w*.35+i*w*.23,h*.5,z,(w*.18,h*.85,d*.85),
                 ((185,65,55),(55,105,175),(65,145,95),(195,155,65))[i])

    elif kind == "clock":
        r.disc((x,h*.5,z),w*.48,light)
        _box(r,x,h*.5,z-d*.55,(.025,h*.32,.02),dark)

    else:
        _box(r,x,h*.5,z,(w,h,d),col)


def draw_prop(r, name, pos, rot=0.0, scale=1.0, color=None):
    """3D prop render; pos=(x,z)."""
    if name not in CATALOG:
        return None

    if not isinstance(pos, (list, tuple)) or len(pos) != 2:
        return None

    try:
        x, z = float(pos[0]), float(pos[1])
        sc = float(scale)
        angle = float(rot)
    except (TypeError, ValueError, OverflowError):
        return None

    if not all(math.isfinite(v) for v in (x,z,sc,angle)):
        return None

    if not .1 <= sc <= 4.0:
        return None

    kind, dims, default = CATALOG[name]
    col = validate_color(color, default)
    w, h, d = (float(v)*sc for v in dims)

    # Filhal rotation ko sirf safe orthogonal values tak normalize karo.
    # Detailed rotated mesh Phase 5 mein add hoga.
    _recipe(r, kind, x, z, w, h, d, col)

    return {
        "center": (x, h*.5, z),
        "bounds": (x-w*.5,x+w*.5,0,h,z-d*.5,z+d*.5),
        "name": name
    }


def draw_custom_prop(r, spec):
    """Gemini ke primitive parts se custom prop banao."""
    if not isinstance(spec, dict):
        return False

    parts = spec.get("parts", [])
    if not isinstance(parts, list) or not 1 <= len(parts) <= 25:
        return False

    for part in parts:
        if not isinstance(part, dict):
            return False
        if part.get("shape") not in ("box","cylinder","sphere","cone","quad"):
            return False

        pos = part.get("pos")
        size = part.get("size")
        if not isinstance(pos,list) or len(pos)!=3:
            return False
        if not isinstance(size,list) or len(size)!=3:
            return False

        try:
            p = [float(v) for v in pos]
            sz = [float(v) for v in size]
        except (TypeError,ValueError,OverflowError):
            return False

        if not all(math.isfinite(v) for v in p+sz):
            return False
        if any(v <= 0 or v > 8 for v in sz):
            return False

    for part in parts:
        p = [float(v) for v in part["pos"]]
        sz = [float(v) for v in part["size"]]
        col = validate_color(part.get("color"),(150,150,160))
        shape = part["shape"]

        if shape == "quad":
            x,y,z=p
            sx,sy,szd=sz
            points=np.array([
                [x-sx/2,y-sy/2,z],
                [x+sx/2,y-sy/2,z],
                [x+sx/2,y+sy/2,z],
                [x-sx/2,y+sy/2,z]
            ],dtype=np.float64)
            r.poly(points,col,2,tint=True)
        else:
            _part(r,shape,p[0],p[1],p[2],sz,col)

    return True
