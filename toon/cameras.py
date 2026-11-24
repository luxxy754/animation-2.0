"""toon/cameras.py — CAMERA_MOVES: har ek function(u: 0..1, room, target_xz) -> Cam.
Sab moves smootherstep easing ke saath; sirf closeup_shake mein halka handheld shake."""
import math
import numpy as np
from .core import Cam, ease5, lerp, lerpv, look_at, clamp


def _T(txz, y=1.25): return np.array([txz[0], y, txz[1]], float)


def _in(room, p, m=0.3):
    x0, x1, zm, Hh = room["dims"]
    return np.array([clamp(p[0], x0 + m, x1 - m), clamp(p[1], 0.15, Hh - 0.2), clamp(p[2], 0.3, zm - 0.3)])


def _cam(room, eye, tgt, roll=0.0, fov=64):
    eye = _in(room, eye); yaw, pit = look_at(eye, tgt); return Cam(eye, yaw, pit, roll, fov)


def push_in(u, room, txz):          # characters ki taraf aahista aage
    e = ease5(u); T = _T(txz)
    return _cam(room, (T[0] + lerp(.7, .1, e), lerp(1.65, 1.45, e), T[2] - lerp(3.4, 1.5, e)), T, 0, lerp(68, 56, e))


def pull_out(u, room, txz):         # close se peeche hat kar poora room
    e = ease5(u); T = _T(txz)
    return _cam(room, (T[0] + lerp(.1, .7, e), lerp(1.45, 1.7, e), T[2] - lerp(1.4, 3.8, e)), T, 0, lerp(54, 68, e))


def orbit(u, room, txz):            # character ke gird aadha chakkar
    e = ease5(u); T = _T(txz); a = lerp(-.55, .55, e); rad = 2.7
    return _cam(room, (T[0] + rad * math.sin(a), 1.65, T[2] - rad * math.cos(a)), T, 0, 60)


def pan(u, room, txz):              # camera jagah par, left se right ghoomti hai
    e = ease5(u); T = _T(txz, 1.3)
    tgt = np.array([T[0] + lerp(-1.7, 1.7, e), 1.3, T[2]])
    return _cam(room, (T[0] * .3, 1.55, T[2] - 3.4), tgt, 0, 62)


def tilt(u, room, txz):             # joote se sar tak (neeche se upar)
    e = ease5(u); T = _T(txz)
    return _cam(room, (T[0] + .3, 1.4, T[2] - 3.0), np.array([T[0], lerp(.25, 2.5, e), T[2]]), 0, 64)


def through_door(u, room, txz):     # darwaze ki taraf tez rush, fov khulta hai
    e = ease5(u) ** 1.4; dx, dz = room["door"]["pos"]
    eye = lerpv((dx * .3, 1.55, .8), (dx, 1.5, dz - .45), e)
    return _cam(room, eye, np.array([dx, 1.4, dz]), .02 * math.sin(u * 6), lerp(66, 92, e))


def low_angle(u, room, txz):        # neeche se upar dekhna, halka dutch roll
    e = ease5(u); T = _T(txz)
    return _cam(room, (T[0] + lerp(.9, .4, e), lerp(.3, .5, e), T[2] - lerp(2.3, 1.9, e)), np.array([T[0], 1.55, T[2]]), .06 - .02 * e, 64)


def closeup_shake(u, room, txz):    # chehre ke bilkul qareeb + handheld shake
    T = _T(txz, 1.42); ph = u * 110.0
    sh = np.array([.006 * math.sin(ph * 1.3), .006 * math.sin(ph * 1.7 + 1), .004 * math.sin(ph * 0.9 + 2)])
    return _cam(room, np.array([T[0] + .05, 1.45, T[2] - 1.15]) + sh, T + sh * .5, .012 * math.sin(ph * 1.1), lerp(46, 44, ease5(u)))


CAMERA_MOVES = {"push_in": push_in, "pull_out": pull_out, "orbit": orbit, "pan": pan, "tilt": tilt,
                "through_door": through_door, "low_angle": low_angle, "closeup_shake": closeup_shake}
