"""toon/lighting.py — lights se shading (toon bands), glow/bloom, volumetric shafts, sunlight patches, vignette.
Lights: [{"pos":(x,y,z),"color":(b,g,r),"intensity":f,"kind":"sun|panel|lamp"}]"""
import math
import numpy as np, cv2
from .core import W, H, SC, NEAR, clamp, _ic


def light_k(lights, ctr, n=None, q=False, amb0=0.74):
    """Point ctr par roshni ka (b,g,r) multiplier. q=True -> 3 toon bands. Garam/thande rang light ke color se aate hain."""
    kb = kg = kr = amb0
    for L in lights:
        lp = L["pos"]; dx, dy, dz = lp[0] - ctr[0], lp[1] - ctr[1], lp[2] - ctr[2]
        dist = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
        lam = 0.7 if n is None else max(0.0, (dx * n[0] + dy * n[1] + dz * n[2]) / dist)
        if q: lam = round(lam * 3) / 3.0
        kind = L.get("kind", "panel")
        att = 1.0 if kind == "sun" else 1.0 / (1.0 + (dist / (2.5 if kind == "lamp" else 5.0)) ** 2)
        w = L["intensity"] * att * lam; c = L["color"]
        kb += w * c[0] / 255.0; kg += w * c[1] / 255.0; kr += w * c[2] / 255.0
    return np.clip(np.array([kb, kg, kr]), 0.55, 1.4)


def _vig():
    yy, xx = np.mgrid[0:H, 0:W]; d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    return (1 - .30 * np.clip(d - .55, 0, 1) ** 1.3)[..., None].astype(np.float32)


VIG = _vig(); _TONE = {}


def proj_dir(P, D, plane, val):
    """Point P se direction D mein chalo jab tak plane ('x','y','z') = val na ho."""
    i = "xyz".index(plane); P = np.array(P, float); D = np.array(D, float)
    if abs(D[i]) < 1e-6: return None
    return P + D * ((val - P[i]) / D[i])


class LightFX:
    def lk(s, ctr, n=None, q=False): return light_k(s.lights, ctr, n, q, s.amb0)

    def shade(s, base, ctr, n=None, q=False, k=1.0):
        kk = s.lk(ctr, n, q) * k
        return tuple(int(clamp(base[i] * kk[i], 0, 255)) for i in range(3))

    def L_hard(s):
        if s._ah is None: s._ah = np.zeros((H, W, 3), np.uint8)
        return s._ah

    def L_soft(s):                       # quarter-res layer, baad mein blur hota hai (soft glow/shafts)
        if s._as is None: s._as = np.zeros((H // 4, W // 4, 3), np.uint8)
        return s._as

    def fx_poly(s, P3, col, hard=False):
        p = s.pts(P3)
        if p is None: return
        if hard: cv2.fillPoly(s.L_hard(), [p], _ic(col), cv2.LINE_AA)
        else: cv2.fillPoly(s.L_soft(), [(p // 4).astype(np.int32)], _ic(col))

    def glow(s, pos, rad, col, k=1.0):
        pc = s.c.cam(np.array(pos, float))
        if pc[0, 2] < NEAR: return
        u, v = s.c.proj(pc)[0]; rp = s.c.f * rad / pc[0, 2]
        if u < -3 * rp or u > W + 3 * rp or v < -3 * rp or v > H + 3 * rp: return
        L = s.L_soft()
        for fr_, m in ((1.0, .22), (.7, .4), (.45, .65), (.25, 1.0)):
            cv2.circle(L, (int(u / 4), int(v / 4)), max(1, int(rp * fr_ / 4)), tuple(int(clamp(x * k * m, 0, 255)) for x in col), -1)
        cv2.circle(s.L_hard(), (int(u), int(v)), max(1, int(rp * .12)), tuple(int(clamp(x * .8 * k, 0, 255)) for x in col), -1, cv2.LINE_AA)

    def shaft(s, quad, D, ln, col, n=6):
        """Khidki se volumetric shaft: quad ko D direction mein kheench kar slices (door se door, halka hota hua)."""
        quad = np.array(quad, float); D = np.array(D, float)
        for k in range(n - 1, -1, -1):
            a = quad + D * (ln * k / n); b = quad + D * (ln * (k + 1) / n)
            P = np.vstack([a, b]); pc = s.c.cam(P)
            if (pc[:, 2] < NEAR).any(): continue
            hull = cv2.convexHull(np.round(s.c.proj(pc)).astype(np.int32))
            f = (1 - k / n) ** 1.1
            cv2.fillPoly(s.L_soft(), [(hull // 4).astype(np.int32)], tuple(int(clamp(x * f, 0, 255)) for x in col))

    def run_fx(s, fx):
        for e in fx:
            t = e["t"]
            if t == "glow": s.glow(e["pos"], e["rad"], e["col"], e.get("k", 1.0))
            elif t == "panel": s.fx_poly(e["quad"], e["col"], False)
            elif t == "hard": s.fx_poly(e["quad"], e["col"], True)
            elif t == "shaft": s.shaft(e["quad"], e["dir"], e["len"], e["col"], e.get("n", 6))
            elif t == "pool":      # farsh/desk par roshni ka dhabba (ellipse)
                cx, cz = e["c"]; y = e.get("y", 0.01)
                ring = [(cx + e["rx"] * math.cos(a), y, cz + e["rz"] * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 18, endpoint=False)]
                s.fx_poly(ring, e["col"], False)

    def finish(s, tone=(1, 1, 1)):
        """Glow/shafts add, tone + vignette."""
        f = s.fr
        if s._ah is not None: cv2.add(f, s._ah, dst=f)
        if s._as is not None:
            b = cv2.GaussianBlur(s._as, (0, 0), 1.6)
            cv2.add(f, cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR), dst=f)
        key = tuple(tone)
        if key not in _TONE: _TONE[key] = (np.array(key, np.float32) * VIG).astype(np.float32)
        g = f.astype(np.float32) * _TONE[key]
        np.clip(g, 0, 255, out=g); f[:] = g.astype(np.uint8)
