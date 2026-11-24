"""toon/shadows.py — flat, thode neele dark, multiply shadows + banded ambient occlusion.
Sab kuch ek 'multiply layer' (sh) mein banta hai; apply_shadows() shell ke baad ek hi baar lagata hai."""
import math
import numpy as np, cv2
from .core import W, H, clamp

SHCOL = (214, 176, 150)                       # (b,g,r) multiply: neela-sa dark
AO_VALS = ((246, 243, 242), (236, 231, 229), (222, 214, 210))


class Shadows:
    def L_sh(s):
        if s._sh is None: s._sh = np.full((H, W, 3), 255, np.uint8)
        return s._sh

    def key_light(s):
        k = s.room.get("key") if s.room else None
        return None if k is None else np.array(s.lights[k]["pos"], float)

    def _fill(s, P3, col=SHCOL):
        p = s.pts(P3)
        if p is not None: cv2.fillPoly(s.L_sh(), [p], col, cv2.LINE_AA)

    def _floor_pts(s, L, Q):
        if Q[1] >= L[1] - 0.05: return None
        t = L[1] / (L[1] - Q[1]); X = L + (Q - L) * t
        return X

    def _clampd(s, x, z):
        if not s.room: return x, z
        x0, x1, zm, _ = s.room["dims"]; return clamp(x, x0, x1), clamp(z, 0.0, zm)

    def shadow_box(s, x0, x1, y0, y1, z0, z1):
        """Box ko light direction ke saath y=0 par project kar ke floor shadow."""
        L = s.key_light()
        if L is None: return
        pc = s.c.cam(np.array([(x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2]))
        if pc[0, 2] < -3.0: return
        pts = []
        for x in (x0, x1):
            for y in (y0, y1):
                for z in (z0, z1):
                    X = s._floor_pts(L, np.array([x, y, z], float))
                    if X is not None: pts.append(s._clampd(X[0], X[2]))
                    if y0 <= 0.01 and y == y0: pts.append(s._clampd(x, z))
        if len(pts) < 3: return
        h = cv2.convexHull(np.array(pts, np.float32)).reshape(-1, 2)
        if len(h) >= 3: s._fill([(a, 0.0, b) for a, b in h])

    def shadow_char(s, P, light=None, sit=False):
        """Contact shadow + light direction ke hisaab se lambi cast shadow (farsh), deewar ke qareeb ho to deewar par bhi."""
        P = np.array(P, float).reshape(3); x, z = P[0], P[2]
        ring = [(x + .30 * math.cos(a), 0.0, z + .13 * math.sin(a)) for a in np.linspace(0, 2 * math.pi, 14, endpoint=False)]
        s._fill(ring)
        L = np.array(light, float) if light is not None else s.key_light()
        if L is None: return
        k = .78 if sit else 1.0
        body = [(x - .16, 0, z), (x + .16, 0, z), (x - .22, 1.05 * k, z), (x + .22, 1.05 * k, z), (x - .2, 1.5 * k, z), (x + .2, 1.5 * k, z), (x, 1.78 * k, z)]
        fl = []
        for q in body:
            X = s._floor_pts(L, np.array(q, float))
            if X is not None: fl.append(s._clampd(X[0], X[2]))
        fl += [s._clampd(x - .16, z), s._clampd(x + .16, z)]
        if len(fl) >= 3:
            h = cv2.convexHull(np.array(fl, np.float32)).reshape(-1, 2)
            if len(h) >= 3: s._fill([(a, 0.0, b) for a, b in h])
        if not s.room: return
        x0, x1, zm, Hh = s.room["dims"]
        for ax, val, near in (("x", x0, x - x0), ("x", x1, x1 - x), ("z", zm, zm - z)):
            if near > 1.4: continue
            pts = []
            for q in body:
                Q = np.array(q, float); i = 0 if ax == "x" else 2
                den = Q[i] - L[i]
                if abs(den) < 1e-6: continue
                t = (val - L[i]) / den
                if t < 1.0 or t > 4.0: continue
                Y = L[1] + (Q[1] - L[1]) * t; o = L[2 if ax == "x" else 0] + (Q[2 if ax == "x" else 0] - L[2 if ax == "x" else 0]) * t
                Y = clamp(Y, 0.0, Hh)
                pts.append((Y, o))
            if len(pts) >= 3:
                h = cv2.convexHull(np.array(pts, np.float32)).reshape(-1, 2)
                if len(h) >= 3:
                    off = .01 if ax == "z" else .01
                    if ax == "x": poly = [((val + off) if val == x0 else (val - off), a, b) for a, b in h]
                    else: poly = [(b, a, val - off) for a, b in h]
                    s._fill(poly)

    def ao_edge(s, A, B, inw, ws=(.55, .3, .12), vals=AO_VALS):
        """Kone/edge ke saath banded ambient occlusion strips."""
        A = np.array(A, float); B = np.array(B, float); inw = np.array(inw, float)
        for w, v in zip(ws, vals): s._fill([A, B, B + inw * w, A + inw * w], v)

    def ao_room(s):
        x0, x1, zm, Hh = s.room["dims"]; P = lambda *a: np.array(a, float)
        e = [((x0, 0, 0), (x0, 0, zm), (1, 0, 0)), ((x1, 0, 0), (x1, 0, zm), (-1, 0, 0)), ((x0, 0, zm), (x1, 0, zm), (0, 0, -1)),
             ((x0, Hh, 0), (x0, Hh, zm), (0, -1, 0)), ((x1, Hh, 0), (x1, Hh, zm), (0, -1, 0)), ((x0, Hh, zm), (x1, Hh, zm), (0, -1, 0)),
             ((x0, 0, zm), (x0, Hh, zm), (1, 0, 0)), ((x1, 0, zm), (x1, Hh, zm), (-1, 0, 0)),
             ((x0, 0, zm), (x0, Hh, zm), (0, 0, -1)), ((x1, 0, zm), (x1, Hh, zm), (0, 0, -1))]
        ws = (.5, .28, .1)
        for A, B, inw in e: s.ao_edge(P(*A), P(*B), P(*inw), ws)

    def apply_shadows(s):
        if s._sh is not None:
            cv2.multiply(s.fr, s._sh, dst=s.fr, scale=1 / 255.0); s._sh = None
