"""toon/core.py — Cam, math helpers aur basic geometry renderer R.
Static dunya (deewar/farsh/furniture) mein koi jitter nahi; boil sirf boil>0 par (characters/chhote props)."""
import math, os
import numpy as np, cv2

W = int(os.environ.get("TOON_W", 720)); H = int(os.environ.get("TOON_H", 1280))
SC = W / 540.0                       # line width scale (resolution ke hisaab se)
OUT = (22, 18, 26); NEAR = 0.1
FONT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "Poppins-Bold.ttf")


def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(t): t = clamp(t); return t * t * (3 - 2 * t)                       # smoothstep
def ease5(t): t = clamp(t); return t * t * t * (t * (t * 6 - 15) + 10)     # smootherstep (camera ke liye)
def lerp(a, b, t): return a + (b - a) * t
def lerpv(a, b, t): a = np.array(a, float); return a + (np.array(b, float) - a) * t
def _w(lw): return max(1, int(round(lw * SC)))
def _ic(c): return tuple(int(v) for v in c)


class Cam:
    def __init__(s, pos, yaw, pitch, roll=0.0, hfov=70):
        s.pos = np.array(pos, float); s.roll = roll; s.yaw = yaw; s.pitch = pitch
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


FACES = [((0, 1, 3, 2), (-1, 0, 0)), ((4, 6, 7, 5), (1, 0, 0)), ((0, 4, 5, 1), (0, -1, 0)),
         ((2, 3, 7, 6), (0, 1, 0)), ((0, 2, 6, 4), (0, 0, -1)), ((1, 5, 7, 3), (0, 0, 1))]


class R:
    """Basic renderer. shade()/lights wala hissa lighting.LightFX mixin deta hai, shadows Shadows mixin."""

    def __init__(s, fr, cam, fi, room=None):
        s.fr, s.c, s.fi, s.room = fr, cam, fi, room
        s.ph = (fi // 3) * 1.9                              # sirf characters ke line-boil ke liye phase
        s.amb = np.array((1.0, 1.0, 1.0), float)
        s.lights = room["lights"] if room else []
        s.amb0 = room.get("amb0", 0.74) if room else 0.74
        s._sh = s._ah = s._as = None

    def col(s, c, k=1.0): return tuple(int(clamp(v * a * k, 0, 255)) for v, a in zip(c, s.amb))

    def jit(s, p, amp=0.8):    # boil: har 3 frame par halka hilna (sirf boil>0 par)
        return p + amp * SC * np.stack([np.sin(0.07 * p[:, 0] + 0.11 * p[:, 1] + s.ph), np.cos(0.09 * p[:, 0] - 0.06 * p[:, 1] + s.ph * 1.3)], 1)

    def pts(s, P, boil=0.0):
        pc = s.c.cam(np.asarray(P, float))
        if not (pc[:, 2] >= NEAR).all():
            if (pc[:, 2] < NEAR).all(): return None
            pc = clip_poly(pc)
            if len(pc) < 3: return None
        p = s.c.proj(pc)
        if boil: p = s.jit(p, boil)
        return np.round(np.clip(p, -20000, 20000)).astype(np.int32)

    def poly(s, P, fill, lw=2, k=1.0, tint=True, boil=0.0):
        p = s.pts(P, boil)
        if p is None: return
        if p[:, 0].max() < 0 or p[:, 0].min() > W or p[:, 1].max() < 0 or p[:, 1].min() > H: return
        if fill is not None: cv2.fillPoly(s.fr, [p], s.col(fill, k) if tint else _ic(fill), cv2.LINE_AA)
        if lw: cv2.polylines(s.fr, [p], True, OUT, _w(lw), cv2.LINE_AA)

    def spoly(s, P, color, lw=2, n=None, boil=0.0):
        """Light ke hisaab se shaded polygon (centroid par)."""
        P = np.asarray(P, float); s.poly(P, s.shade(color, P.mean(0), n, True), lw, tint=False, boil=boil)

    def segs(s, A, B, color, lw=1):
        """Bohot saari lines ek saath (vectorised, near-plane clip) — bricks/tiles ke liye."""
        a = s.c.cam(A); b = s.c.cam(B); za, zb = a[:, 2].copy(), b[:, 2].copy()
        keep = (za >= NEAR) | (zb >= NEAR)
        a, b, za, zb = a[keep], b[keep], za[keep], zb[keep]
        if not len(a): return
        m = za < NEAR
        if m.any(): t = (NEAR - za[m]) / (zb[m] - za[m]); a[m] = a[m] + (b[m] - a[m]) * t[:, None]
        m = zb < NEAR
        if m.any(): t = (NEAR - zb[m]) / (za[m] - zb[m]); b[m] = b[m] + (a[m] - b[m]) * t[:, None]
        pa, pb = s.c.proj(a), s.c.proj(b)
        ok = (np.maximum(pa[:, 0], pb[:, 0]) >= 0) & (np.minimum(pa[:, 0], pb[:, 0]) <= W) & \
             (np.maximum(pa[:, 1], pb[:, 1]) >= 0) & (np.minimum(pa[:, 1], pb[:, 1]) <= H)
        if not ok.any(): return
        arr = np.round(np.clip(np.stack([pa[ok], pb[ok]], 1), -20000, 20000)).astype(np.int32)
        cv2.polylines(s.fr, list(arr), False, _ic(color), _w(lw), cv2.LINE_AA)

    def fill_quads(s, Q, T, pal):
        """Quads (N,4,3) tone index T se palette pal mein; tone ke hisaab se grouped fill (tez)."""
        n = len(Q)
        pc = s.c.cam(Q.reshape(-1, 3)).reshape(n, 4, 3)
        zok = pc[:, :, 2] >= NEAR; full = zok.all(1); part = zok.any(1) & ~full
        polys = {}
        if full.any():
            idx = np.nonzero(full)[0]
            sp = s.c.proj(pc[idx].reshape(-1, 3)).reshape(-1, 4, 2)
            ok = (sp[:, :, 0].max(1) >= 0) & (sp[:, :, 0].min(1) <= W) & (sp[:, :, 1].max(1) >= 0) & (sp[:, :, 1].min(1) <= H)
            sp = np.round(np.clip(sp, -20000, 20000)).astype(np.int32)
            for i, pq, o in zip(idx, sp, ok):
                if o: polys.setdefault(int(T[i]), []).append(pq)
        for i in np.nonzero(part)[0]:
            pq = s.pts(Q[i])
            if pq is not None: polys.setdefault(int(T[i]), []).append(pq)
        for t, lst in polys.items(): cv2.fillPoly(s.fr, lst, _ic(pal[t]))

    def box(s, x0, x1, y0, y1, z0, z1, color, lw=2, boil=0.0, k=1.0):
        V = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], float)
        for idx, n in FACES:
            q = V[list(idx)]; ctr = q.mean(0)
            if np.dot(n, s.c.pos - ctr) > 0:
                s.poly(q, s.shade(color, ctr, n, True, k), lw, tint=False, boil=boil)

    def disc(s, C, rad, fill, lw=1):
        pc = s.c.cam(np.array(C, float)); z = pc[0, 2]
        if z < NEAR: return
        u, v = s.c.proj(pc)[0]; r_ = max(1, int(round(s.c.f * rad / z)))
        cv2.circle(s.fr, (int(u), int(v)), r_, _ic(fill), -1, cv2.LINE_AA)
        if lw: cv2.circle(s.fr, (int(u), int(v)), r_, OUT, _w(lw), cv2.LINE_AA)

    def cyl(s, x, z, y0, y1, rad, color, lw=2):
        """Billboard cylinder (bottle, bin, pot)."""
        b = s.c.cam(np.array([x, y0, z], float)); t = s.c.cam(np.array([x, y1, z], float))
        if b[0, 2] < NEAR or t[0, 2] < NEAR: return
        pb = s.c.proj(b)[0]; pt = s.c.proj(t)[0]; rb = s.c.f * rad / b[0, 2]; rt = s.c.f * rad / t[0, 2]
        col = s.shade(color, (x, (y0 + y1) / 2, z), None, True); lit = tuple(min(255, int(v * 1.15) + 8) for v in col)
        q = np.round(np.array([(pb[0] - rb, pb[1]), (pb[0] + rb, pb[1]), (pt[0] + rt, pt[1]), (pt[0] - rt, pt[1])])).astype(np.int32)
        cv2.ellipse(s.fr, (int(pb[0]), int(pb[1])), (max(1, int(rb)), max(1, int(rb * .35))), 0, 0, 360, col, -1, cv2.LINE_AA)
        cv2.fillPoly(s.fr, [q], col, cv2.LINE_AA)
        cv2.ellipse(s.fr, (int(pt[0]), int(pt[1])), (max(1, int(rt)), max(1, int(rt * .35))), 0, 0, 360, lit, -1, cv2.LINE_AA)
        w = _w(lw)
        cv2.ellipse(s.fr, (int(pb[0]), int(pb[1])), (max(1, int(rb)), max(1, int(rb * .35))), 0, 0, 180, OUT, w, cv2.LINE_AA)
        cv2.line(s.fr, tuple(q[0]), tuple(q[3]), OUT, w, cv2.LINE_AA); cv2.line(s.fr, tuple(q[1]), tuple(q[2]), OUT, w, cv2.LINE_AA)
        cv2.ellipse(s.fr, (int(pt[0]), int(pt[1])), (max(1, int(rt)), max(1, int(rt * .35))), 0, 0, 360, OUT, w, cv2.LINE_AA)

    def tex(s, img, P, k=1.0):
        """Chhoti flat texture (sign/poster) — sirf bounding box warp hota hai."""
        pc = s.c.cam(np.array(P, float))
        if (pc[:, 2] < NEAR).any(): return
        dst = s.c.proj(pc).astype(np.float32)
        x0, y0 = np.floor(dst.min(0)).astype(int); x1, y1 = np.ceil(dst.max(0)).astype(int)
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 < 2 or y1 - y0 < 2: return
        h, w = img.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]), dst - np.float32([x0, y0]))
        wp = cv2.warpPerspective(img, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR)
        mk = cv2.warpPerspective(np.full((h, w), 255, np.uint8), M, (x1 - x0, y1 - y0)) > 127
        roi = s.fr[y0:y1, x0:x1]; roi[mk] = np.clip(wp[mk].astype(np.float32) * k, 0, 255).astype(np.uint8)

    def sky(s, stops):
        """Aasman gradient (camera pitch ke hisaab se). stops = [(elevation_rad, (b,g,r)), ...] barhte hue."""
        el = np.array([a for a, _ in stops]); cols = np.array([c for _, c in stops], float)
        ang = s.c.pitch + np.arctan((H / 2 - np.arange(H)) / s.c.f)
        lut = np.stack([np.interp(ang, el, cols[:, i]) for i in range(3)], 1)
        s.fr[:] = lut[:, None, :].astype(np.uint8)
