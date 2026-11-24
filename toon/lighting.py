
"""toon/lighting.py — Toon lighting, glow, shafts aur vignette."""

import math
import numpy as np
import cv2

from .core import W, H, SC, NEAR, clamp, _ic


def light_k(lights, ctr, n=None, q=False, amb0=0.74):
    """Qareebi lights se RGB toon multiplier nikalo."""
    ctr = np.asarray(ctr, dtype=np.float64)
    result = np.full(3, float(amb0), dtype=np.float64)

    for light in lights or []:
        lp = np.asarray(light.get("pos", (0, 2, 0)), dtype=np.float64)
        delta = lp - ctr
        dist = float(np.linalg.norm(delta)) + 1e-6

        if n is None:
            lam = 0.7
        else:
            normal = np.asarray(n, dtype=np.float64)
            nl = float(np.linalg.norm(normal))
            if nl < 1e-8:
                lam = 0.7
            else:
                normal /= nl
                lam = max(0.0, float(np.dot(delta / dist, normal)))

        if q:
            lam = round(lam * 3.0) / 3.0

        kind = light.get("kind", "panel")
        if kind == "sun":
            attenuation = 1.0
        elif kind == "fire":
            attenuation = 1.0 / (1.0 + (dist / 2.0) ** 2)
        elif kind == "lamp" or kind == "street_lamp":
            attenuation = 1.0 / (1.0 + (dist / 2.5) ** 2)
        else:
            attenuation = 1.0 / (1.0 + (dist / 5.0) ** 2)

        intensity = max(0.0, float(light.get("intensity", 1.0)))
        tint = np.asarray(light.get("color", (255, 255, 255)), dtype=np.float64)
        result += intensity * attenuation * lam * tint / 255.0

    return np.clip(result, 0.48, 1.45)


def _vig():
    """Ek martaba halka vignette texture banao."""
    yy, xx = np.mgrid[0:H, 0:W]
    dx = (xx - W / 2.0) / max(1.0, W / 2.0)
    dy = (yy - H / 2.0) / max(1.0, H / 2.0)
    distance = np.sqrt(dx * dx + dy * dy)
    return (1.0 - 0.25 * np.clip(distance - 0.55, 0, 1) ** 1.3)[..., None].astype(np.float32)


VIG = _vig()
_TONE = {}


def proj_dir(P, D, plane, val):
    """Ray ko diye gaye plane tak project karo."""
    axis = "xyz".index(plane)
    P = np.asarray(P, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)

    if abs(D[axis]) < 1e-7:
        return None

    return P + D * ((float(val) - P[axis]) / D[axis])


def _project_polygon(r, points):
    """3D polygon ko camera space mein clip karke 2D points do."""
    points = np.asarray(points, dtype=np.float64)
    pc = r.c.cam(points)

    if len(pc) < 3:
        return None

    from .core import clip_poly

    if (pc[:, 2] < NEAR).any():
        pc = clip_poly(pc)

    if len(pc) < 3:
        return None

    projected = r.c.proj(pc)
    if not np.isfinite(projected).all():
        return None

    return np.round(np.clip(projected, -20000, 20000)).astype(np.int32)


class LightFX:

    def lk(s, ctr, n=None, q=False):
        return light_k(s.lights, ctr, n, q, s.amb0)

    def shade(s, base, ctr, n=None, q=False, k=1.0):
        """Flat cel-shading ke liye lighting multiplier."""
        kk = s.lk(ctr, n, q) * float(k)
        return tuple(
            int(clamp(float(base[i]) * kk[i], 0, 255))
            for i in range(3)
        )

    def L_hard(s):
        if s._ah is None:
            s._ah = np.zeros((H, W, 3), dtype=np.uint8)
        return s._ah

    def L_soft(s):
        if s._as is None:
            s._as = np.zeros(
                (max(1, H // 4), max(1, W // 4), 3),
                dtype=np.uint8
            )
        return s._as

    def fx_poly(s, P3, col, hard=False):
        p = _project_polygon(s, P3)
        if p is None:
            return

        if hard:
            cv2.fillPoly(s.L_hard(), [p], _ic(col), cv2.LINE_AA)
        else:
            small = np.round(p.astype(np.float32) / 4.0).astype(np.int32)
            cv2.fillPoly(s.L_soft(), [small], _ic(col), cv2.LINE_AA)

    def glow(s, pos, rad, col, k=1.0):
        """Lamp aur bulb ka soft bloom."""
        pc = s.c.cam(np.asarray(pos, dtype=np.float64))
        if pc[0, 2] < NEAR:
            return

        u, v = s.c.proj(pc)[0]
        radius = float(s.c.f) * float(rad) / float(pc[0, 2])

        if radius <= 0 or not np.isfinite(radius):
            return

        if (u < -3 * radius or u > W + 3 * radius or
                v < -3 * radius or v > H + 3 * radius):
            return

        layer = s.L_soft()
        cx, cy = int(round(u / 4)), int(round(v / 4))

        for fraction, strength in (
            (1.0, 0.16),
            (0.72, 0.28),
            (0.48, 0.45),
            (0.26, 0.72),
            (0.12, 1.0),
        ):
            rr = max(1, int(round(radius * fraction / 4)))
            c = tuple(
                int(clamp(float(vv) * float(k) * strength, 0, 255))
                for vv in col
            )
            cv2.circle(layer, (cx, cy), rr, c, -1, cv2.LINE_AA)

        hard_col = tuple(
            int(clamp(float(vv) * 0.65 * float(k), 0, 255))
            for vv in col
        )
        cv2.circle(
            s.L_hard(),
            (int(round(u)), int(round(v))),
            max(1, int(round(radius * 0.10))),
            hard_col,
            -1,
            cv2.LINE_AA
        )

    def shaft(s, quad, D, ln, col, n=7):
        """Khidki se aati hui additive volumetric roshni."""
        quad = np.asarray(quad, dtype=np.float64)
        direction = np.asarray(D, dtype=np.float64)
        length = max(0.0, float(ln))
        count = max(2, min(16, int(n)))

        if quad.shape != (4, 3) or not np.isfinite(quad).all():
            return

        if float(np.linalg.norm(direction)) < 1e-8:
            return

        direction /= np.linalg.norm(direction)

        for i in range(count - 1, -1, -1):
            a = quad + direction * (length * i / count)
            b = quad + direction * (length * (i + 1) / count)
            points = np.vstack((a, b))

            p = _project_polygon(s, points)
            if p is None:
                continue

            hull = cv2.convexHull(p)
            small = np.round(hull.astype(np.float32) / 4.0).astype(np.int32)

            strength = (1.0 - i / count) ** 1.15
            c = tuple(
                int(clamp(float(v) * strength, 0, 255))
                for v in col
            )
            cv2.fillPoly(s.L_soft(), [small], c, cv2.LINE_AA)

    def run_fx(s, fx):
        """Room ke tamam lighting effects render karo."""
        for effect in fx or []:
            kind = effect.get("t")

            if kind == "glow":
                s.glow(
                    effect["pos"],
                    effect["rad"],
                    effect["col"],
                    effect.get("k", 1.0)
                )

            elif kind == "panel":
                s.fx_poly(effect["quad"], effect["col"], False)

            elif kind == "hard":
                s.fx_poly(effect["quad"], effect["col"], True)

            elif kind == "shaft":
                s.shaft(
                    effect["quad"],
                    effect["dir"],
                    effect["len"],
                    effect["col"],
                    effect.get("n", 7)
                )

            elif kind == "pool":
                cx, cz = effect["c"]
                y = float(effect.get("y", 0.012))
                rx = max(0.01, float(effect["rx"]))
                rz = max(0.01, float(effect["rz"]))

                ring = [
                    (
                        cx + rx * math.cos(a),
                        y,
                        cz + rz * math.sin(a)
                    )
                    for a in np.linspace(0, 2 * math.pi, 28, endpoint=False)
                ]
                s.fx_poly(ring, effect["col"], False)

    def finish(s, tone=(1.0, 1.0, 1.0)):
        """Glow, light shafts, color tone aur vignette apply karo."""
        frame = s.fr

        if s._ah is not None:
            cv2.add(frame, s._ah, dst=frame)

        if s._as is not None:
            blurred = cv2.GaussianBlur(s._as, (0, 0), 1.7)
            resized = cv2.resize(
                blurred,
                (W, H),
                interpolation=cv2.INTER_LINEAR
            )
            cv2.add(frame, resized, dst=frame)

        key = tuple(float(v) for v in tone)
        if key not in _TONE:
            tint = np.asarray(key, dtype=np.float32).reshape(1, 1, 3)
            _TONE[key] = tint * VIG

        result = frame.astype(np.float32) * _TONE[key]
        np.clip(result, 0, 255, out=result)
        frame[:] = result.astype(np.uint8)
