
"""toon/cameras.py
Camera moves: har function ka contract (u: 0..1, room, target_xz) -> Cam.
Indoor aur outdoor dono ke liye safe. Sab moves smooth smootherstep easing use karte hain.
Sirf closeup_shake mein bohot halka deterministic handheld shake hai.
"""
import math
import numpy as np

from .core import Cam, ease5, lerp, lerpv, look_at, clamp


def _target(txz, y=1.25):
    return np.array([float(txz[0]), float(y), float(txz[1])], dtype=float)


def _room_dims(room):
    """Room dimensions ko safe values mein normalize karta hai."""
    dims = room.get("dims", (-5.0, 5.0, 12.0, 4.0))
    try:
        x0, x1, zmax, height = [float(v) for v in dims]
    except (TypeError, ValueError):
        x0, x1, zmax, height = -5.0, 5.0, 12.0, 4.0

    if x1 <= x0:
        x0, x1 = -5.0, 5.0
    if zmax <= 1.0:
        zmax = 12.0
    if height <= 1.0:
        height = 4.0

    return x0, x1, zmax, height


def _inside(room, p, margin=0.3):
    x0, x1, zmax, height = _room_dims(room)

    # Chhote outdoor spaces mein margin boundaries ko cross na kare.
    mx = min(margin, max(0.0, (x1 - x0) * 0.2))
    mz = min(margin, max(0.0, (zmax - 0.3) * 0.2))

    return np.array([
        clamp(float(p[0]), x0 + mx, x1 - mx),
        clamp(float(p[1]), 0.15, max(0.2, height - 0.2)),
        clamp(float(p[2]), 0.15, max(0.3, zmax - mz))
    ], dtype=float)


def _make_cam(room, eye, target, roll=0.0, fov=64):
    eye = _inside(room, eye)
    yaw, pitch = look_at(eye, target)
    return Cam(eye, yaw, pitch, roll, fov)


def _ease(u):
    return ease5(clamp(float(u), 0.0, 1.0))


def push_in(u, room, target_xz):
    """Character ki taraf smooth camera push."""
    e = _ease(u)
    t = _target(target_xz)

    eye = np.array([
        t[0] + lerp(0.7, 0.1, e),
        lerp(1.65, 1.45, e),
        t[2] - lerp(3.4, 1.5, e)
    ])

    return _make_cam(room, eye, t, 0.0, lerp(68, 56, e))


def pull_out(u, room, target_xz):
    """Close-up se wide room view tak smooth pull."""
    e = _ease(u)
    t = _target(target_xz)

    eye = np.array([
        t[0] + lerp(0.1, 0.7, e),
        lerp(1.45, 1.7, e),
        t[2] - lerp(1.4, 3.8, e)
    ])

    return _make_cam(room, eye, t, 0.0, lerp(54, 68, e))


def orbit(u, room, target_xz):
    """Character ke gird controlled half-orbit."""
    e = _ease(u)
    t = _target(target_xz)
    angle = lerp(-0.55, 0.55, e)
    radius = 2.7

    eye = np.array([
        t[0] + radius * math.sin(angle),
        1.65,
        t[2] - radius * math.cos(angle)
    ])

    return _make_cam(room, eye, t, 0.0, 60)


def pan(u, room, target_xz):
    """Camera position stable rakh kar left-to-right pan."""
    e = _ease(u)
    t = _target(target_xz, 1.3)

    eye = np.array([t[0] * 0.3, 1.55, t[2] - 3.4])
    look = np.array([
        t[0] + lerp(-1.7, 1.7, e),
        1.3,
        t[2]
    ])

    return _make_cam(room, eye, look, 0.0, 62)


def tilt(u, room, target_xz):
    """Character ke lower body se face tak tilt."""
    e = _ease(u)
    t = _target(target_xz)

    eye = np.array([t[0] + 0.3, 1.4, t[2] - 3.0])
    look = np.array([
        t[0],
        lerp(0.25, 2.5, e),
        t[2]
    ])

    return _make_cam(room, eye, look, 0.0, 64)


def walk_through(u, room, target_xz):
    """Outdoor ya door-less world mein character ke paas se aage guzarti camera."""
    e = _ease(u)
    t = _target(target_xz, 1.3)

    x0, x1, zmax, height = _room_dims(room)

    # Camera target ke peeche se shuru ho kar uske aage ki taraf move karti hai.
    start_z = min(zmax - 0.5, t[2] - 4.0)
    end_z = max(0.4, t[2] + 2.0)

    # Agar space chhoti ho to camera ko safe range mein rakho.
    if end_z < start_z:
        start_z, end_z = end_z, start_z

    eye = np.array([
        clamp(t[0] + lerp(-0.35, 0.25, e), x0 + 0.3, x1 - 0.3),
        lerp(1.55, 1.7, e),
        lerp(start_z, end_z, e)
    ])

    look = np.array([
        t[0],
        lerp(1.3, 1.45, e),
        t[2] + lerp(0.2, 1.0, e)
    ])

    return _make_cam(room, eye, look, 0.0, lerp(66, 74, e))


def through_door(u, room, target_xz):
    """Door ho to us se guzarti camera; door na ho to walk_through fallback."""
    door = room.get("door")

    if not isinstance(door, dict) or not isinstance(door.get("pos"), (tuple, list)):
        return walk_through(u, room, target_xz)

    try:
        dx, dz = float(door["pos"][0]), float(door["pos"][1])
    except (TypeError, ValueError, IndexError):
        return walk_through(u, room, target_xz)

    e = _ease(u) ** 1.4

    start = np.array([dx * 0.3, 1.55, 0.8], dtype=float)
    end = np.array([dx, 1.5, dz - 0.45], dtype=float)
    eye = lerpv(start, end, e)

    look = np.array([dx, 1.4, dz], dtype=float)
    roll = 0.02 * math.sin(float(u) * 6.0)
    fov = lerp(66, 92, e)

    return _make_cam(room, eye, look, roll, fov)


def low_angle(u, room, target_xz):
    """Low-angle hero shot with a very subtle dutch roll."""
    e = _ease(u)
    t = _target(target_xz)

    eye = np.array([
        t[0] + lerp(0.9, 0.4, e),
        lerp(0.3, 0.5, e),
        t[2] - lerp(2.3, 1.9, e)
    ])

    look = np.array([t[0], 1.55, t[2]])
    roll = 0.06 - 0.02 * e

    return _make_cam(room, eye, look, roll, 64)


def closeup_shake(u, room, target_xz):
    """Face close-up with extremely subtle deterministic handheld motion."""
    t = _target(target_xz, 1.42)
    phase = float(u) * 110.0

    shake = np.array([
        0.006 * math.sin(phase * 1.3),
        0.006 * math.sin(phase * 1.7 + 1.0),
        0.004 * math.sin(phase * 0.9 + 2.0)
    ])

    eye = np.array([t[0] + 0.05, 1.45, t[2] - 1.15]) + shake
    look = t + shake * 0.5
    roll = 0.012 * math.sin(phase * 1.1)

    return _make_cam(room, eye, look, roll, lerp(46, 44, _ease(u)))


CAMERA_MOVES = {
    "push_in": push_in,
    "pull_out": pull_out,
    "orbit": orbit,
    "pan": pan,
    "tilt": tilt,
    "through_door": through_door,
    "walk_through": walk_through,
    "low_angle": low_angle,
    "closeup_shake": closeup_shake
}
