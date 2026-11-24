"""Lipsync (PART B): maujooda voice.py se har character ki line ki awaaz banata hai aur amplitude -> talk_lookup(char_name, t).
Dhyan: t SCENE-LOCAL seconds hai (render_scene(frame_idx, ...) mein frame_idx bhi scene ke andar ka hai)."""
import math
import os

import numpy as np

import voice
from toon.characters import get_char


class TalkMap:
    """talk_lookup: __call__(char_name, t) -> 0..1. Picklable (multiprocessing ke liye)."""

    def __init__(self, lines, fps):
        self.lines, self.fps = lines, fps

    def __call__(self, name, t):
        n = str(name).lower()
        for l in self.lines:
            if l["name"] == n and l["t0"] <= t < l["t1"]:
                e = l["env"]
                if len(e) == 0:
                    return 0.0
                return float(e[min(max(int(round((t - l["t0"]) * self.fps)), 0), len(e) - 1)])
        return 0.0


def build(scenes, tmp, fps=30):
    """Poori video ek hi engine se (kokoro fail ho to saari lines edge-tts se dobara). Returns (audio@voice.SR, infos, total_sec).
    infos[i] = dict(start, hold, lines=[{name,text,t0,t1,env,words}], talk=TalkMap). t0/t1/words scene-local hain."""
    errs = []
    for eng in voice.engines():
        try:
            return _build(eng, scenes, tmp, fps)
        except Exception as e:
            errs.append(str(e)[:160])
            print(f"[lipsync] {eng} fail -> agla engine poori video ke liye ({errs[-1]})", flush=True)
    raise RuntimeError("voice fail: " + " | ".join(errs))


def _build(eng, scenes, tmp, fps, gap=0.18, lead=0.30, tail=0.35, extra=1.2):
    one = os.environ.get("SINGLE_VOICE") == "1"                 # 1 ho to sab characters ek hi awaaz (workflow wali) mein
    orig = (voice.VOICE, voice.EDGE_VOICE)
    chunks, infos, t, n = [], [], 0.0, 0
    try:
        for si, sc in enumerate(scenes):
            cur, lines = (lead if si == 0 else 0.15), []
            for ch in sc["characters"]:
                text = (ch.get("line") or "").strip()
                if not text:
                    continue
                cd = get_char(ch["name"])
                if not one:
                    voice.VOICE, voice.EDGE_VOICE = cd["voice"], cd["edge"]
                p = os.path.join(tmp, f"v{n}.wav")
                n += 1
                voice.synth(text, p, eng)
                x, sr = voice.read(p)
                if sr != voice.SR:
                    x = np.interp(np.linspace(0, len(x), int(len(x) * voice.SR / sr), endpoint=False), np.arange(len(x)), x).astype(np.float32)
                x, _, _ = voice.trim(x, voice.SR)
                x = voice.squeeze_silences(x, voice.SR)
                d = len(x) / voice.SR
                env, _ = voice.analyze(x, voice.SR, fps)
                words = [(w, cur + a, cur + b) for w, a, b in voice.word_times(text, d, env, fps)]
                lines.append(dict(name=str(ch["name"]).lower(), text=text, t0=cur, t1=cur + d, env=env, words=words, x=x))
                cur += d + gap
            dur = float(sc.get("duration", 3.0))
            if lines:
                end = cur - gap
                hold = max(end + tail, min(dur, end + extra))     # awaaz ke baad zyada se zyada 'extra' sec ka reaction
            else:
                hold = min(max(dur, 1.2), 4.0)
            hold = math.ceil(hold * fps) / fps                    # poore frames
            for l in lines:
                chunks.append((t + l["t0"], l.pop("x")))
            infos.append(dict(start=t, hold=hold, lines=lines, talk=TalkMap(lines, fps)))
            t += hold
    finally:
        voice.VOICE, voice.EDGE_VOICE = orig
    out = np.zeros(int(t * voice.SR) + voice.SR, np.float32)
    for st, x in chunks:
        a = int(st * voice.SR)
        out[a:a + len(x)] += x
    print(f"[lipsync] engine={eng}, {n} lines, total {t:.1f}s", flush=True)
    return out[: int(t * voice.SR)], infos, t
