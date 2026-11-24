
"""toon/shadows.py — Furniture, character shadows aur ambient occlusion."""

import math
import numpy as np
import cv2

from .core import W, H, clamp

SHCOL = (214, 176, 150)

AO_VALS = (
    (246, 243, 242),
    (236, 231, 229),
    (222, 214, 210)
)


class Shadows:

    def L_sh(s):
        """Multiply shadow layer lazmi tor par white se shuru hoti hai."""
        if s._sh is None:
            s._sh = np.full((H, W, 3), 255, dtype=np.uint8)
        return s._sh

    def key_light(s):
        """Room ki main light ki position."""
        if not s.room:
            return None

        key = s.room.get("key")
        if key is None:
            return None

        lights = s.lights or []
        if not isinstance(key, int) or key < 0 or key >= len(lights):
            return None

        return np.asarray(
            lights[key].get("pos", (0, 3, 0)),
            dtype=np.float64
        )

    def _fill(s, P3, col=SHCOL):
        """3D shadow polygon ko multiply layer par banao."""
        p = s.pts(np.asarray(P3, dtype=np.float64))
        if p is None or len(p) < 3:
            return

        if not np.isfinite(p).all():
            return

        cv2.fillPoly(s.L_sh(), [p], tuple(int(v) for v in col), cv2.LINE_AA)

    def _floor_pts(s, L, Q):
        """Light ray ko floor y=0 tak project karo."""
        if Q[1] >= L[1] - 0.05:
            return None

        denominator = L[1] - Q[1]
        if abs(denominator) < 1e-8:
            return None

        t = L[1] / denominator
        if t < 0:
            return None

        return L + (Q - L) * t

    def _clampd(s, x, z):
        """Shadow ko room ki boundaries ke andar rakho."""
        if not s.room:
            return float(x), float(z)

        dims = s.room.get("dims")
        if not dims:
            return float(x), float(z)

        x0, x1, zm, _ = dims
        return (
            clamp(x, x0, x1),
            clamp(z, 0.0, zm)
        )

    def shadow_box(s, x0, x1, y0, y1, z0, z1):
        """Furniture box ko light direction mein floor par project karo."""
        L = s.key_light()
        if L is None:
            return

        center = np.array([
            (x0 + x1) * 0.5,
            (y0 + y1) * 0.5,
            (z0 + z1) * 0.5
        ], dtype=np.float64)

        pc = s.c.cam(center.reshape(1, 3))
        if pc[0, 2] < -3.0:
            return

        projected = []

        for x in (x0, x1):
            for y in (y0, y1):
                for z in (z0, z1):
                    point = np.array([x, y, z], dtype=np.float64)
                    floor_point = s._floor_pts(L, point)

                    if floor_point is not None:
                        projected.append(
                            s._clampd(floor_point[0], floor_point[2])
                        )

                    if y <= 0.01:
                        projected.append(s._clampd(x, z))

        if len(projected) < 3:
            return

        points = np.asarray(projected, dtype=np.float32)
        hull = cv2.convexHull(points).reshape(-1, 2)

        if len(hull) < 3:
            return

        polygon = [(float(x), 0.005, float(z)) for x, z in hull]
        s._fill(polygon)

    def shadow_char(s, P, light=None, sit=False):
        """Character ke contact aur directional cast shadows."""
        P = np.asarray(P, dtype=np.float64).reshape(3)
        x, y, z = P

        # Pairon ke neeche soft contact shadow.
        radius_x = 0.30 if not sit else 0.25
        radius_z = 0.14 if not sit else 0.12

        ring = [
            (
                x + radius_x * math.cos(a),
                0.006,
                z + radius_z * math.sin(a)
            )
            for a in np.linspace(0, 2 * math.pi, 24, endpoint=False)
        ]

        s._fill(ring, (220, 190, 170))

        L = (
            np.asarray(light, dtype=np.float64)
            if light is not None
            else s.key_light()
        )

        if L is None:
            return

        L = np.asarray(L, dtype=np.float64).reshape(3)

        # Character ke body points se lamba cast shadow banao.
        factor = 0.78 if sit else 1.0

        body = [
            (x - .16, y, z),
            (x + .16, y, z),
            (x - .22, y + 1.05 * factor, z),
            (x + .22, y + 1.05 * factor, z),
            (x - .19, y + 1.48 * factor, z),
            (x + .19, y + 1.48 * factor, z),
            (x, y + 1.78 * factor, z)
        ]

        floor_points = []

        for point in body:
            projected = s._floor_pts(L, np.asarray(point, dtype=np.float64))
            if projected is not None:
                floor_points.append(
                    s._clampd(projected[0], projected[2])
                )

        floor_points.extend([
            s._clampd(x - .16, z),
            s._clampd(x + .16, z)
        ])

        if len(floor_points) >= 3:
            hull = cv2.convexHull(
                np.asarray(floor_points, dtype=np.float32)
            ).reshape(-1, 2)

            if len(hull) >= 3:
                poly = [
                    (float(a), 0.008, float(b))
                    for a, b in hull
                ]
                s._fill(poly, (220, 185, 165))

        # Qareebi deewar par bhi cast shadow project karo.
        if not s.room:
            return

        dims = s.room.get("dims")
        if not dims:
            return

        x0, x1, zm, room_h = dims

        walls = (
            ("x", x0, x - x0),
            ("x", x1, x1 - x),
            ("z", zm, zm - z)
        )

        for axis, value, distance in walls:
            if distance < 0 or distance > 1.4:
                continue

            projected = []

            for point in body:
                Q = np.asarray(point, dtype=np.float64)
                index = 0 if axis == "x" else 2

                denominator = Q[index] - L[index]
                if abs(denominator) < 1e-7:
                    continue

                t = (value - L[index]) / denominator

                if t < 1.0 or t > 4.0:
                    continue

                hit = L + (Q - L) * t
                height = clamp(hit[1], 0.0, room_h)

                if axis == "x":
                    projected.append((height, hit[2]))
                else:
                    projected.append((height, hit[0]))

            if len(projected) < 3:
                continue

            hull = cv2.convexHull(
                np.asarray(projected, dtype=np.float32)
            ).reshape(-1, 2)

            if len(hull) < 3:
                continue

            if axis == "x":
                wall_x = value + (0.01 if value == x0 else -0.01)
                polygon = [
                    (wall_x, float(a), float(b))
                    for a, b in hull
                ]
            else:
                wall_z = value - 0.01
                polygon = [
                    (float(b), float(a), wall_z)
                    for a, b in hull
                ]

            s._fill(polygon, (225, 190, 175))

    def ao_edge(s, A, B, inw, ws=(0.55, 0.30, 0.12), vals=AO_VALS):
        """Room ke corners mein banded ambient occlusion."""
        A = np.asarray(A, dtype=np.float64)
        B = np.asarray(B, dtype=np.float64)
        inward = np.asarray(inw, dtype=np.float64)

        for width, value in zip(ws, vals):
            w = float(width)

            polygon = [
                A,
                B,
                B + inward * w,
                A + inward * w
            ]

            s._fill(polygon, value)

    def ao_room(s):
        """Deewar aur floor ke joints par static AO."""
        if not s.room:
            return

        dims = s.room.get("dims")
        if not dims:
            return

        x0, x1, zm, height = dims

        edges = [
            ((x0, 0, 0), (x0, 0, zm), (1, 0, 0)),
            ((x1, 0, 0), (x1, 0, zm), (-1, 0, 0)),
            ((x0, 0, zm), (x1, 0, zm), (0, 0, -1)),

            ((x0, height, 0), (x0, height, zm), (0, -1, 0)),
            ((x1, height, 0), (x1, height, zm), (0, -1, 0)),
            ((x0, height, zm), (x1, height, zm), (0, -1, 0)),

            ((x0, 0, zm), (x0, height, zm), (1, 0, 0)),
            ((x1, 0, zm), (x1, height, zm), (-1, 0, 0)),
            ((x0, 0, zm), (x0, height, zm), (0, 0, -1)),
            ((x1, 0, zm), (x1, height, zm), (0, 0, -1))
        ]

        for A, B, inward in edges:
            s.ao_edge(A, B, inward)

    def apply_shadows(s):
        """Scene ke akhir mein multiply shadow layer apply karo."""
        if s._sh is None:
            return

        if s._sh.shape != s.fr.shape:
            s._sh = cv2.resize(
                s._sh,
                (s.fr.shape[1], s.fr.shape[0]),
                interpolation=cv2.INTER_LINEAR
            )

        cv2.multiply(
            s.fr,
            s._sh,
            dst=s.fr,
            scale=1.0 / 255.0
        )

        s._sh = None
