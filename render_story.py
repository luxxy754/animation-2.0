"""script.json (scene JSON) -> output.mp4: har scene toon.world.render_scene se (multiprocessing), voice (lipsync.py), music, captions."""
import faulthandler
import json
import multiprocessing as mp
import os
import subprocess
import sys
import tempfile
import time
import zlib

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

import music
import voice
from toon import lipsync

if os.environ.get("STUB_WORLD") == "1":                      # PART A ka world.py na ho to test ke liye dev_stub
    from dev_stub import stub_world as world
else:
    from toon import world

HERE = os.path.dirname(os.path.abspath(__file__))
FPS, OW, OH = 30, 1080, 1920
MAX_SEC = 59.5                                               # Short 60 second se kam
YELLOW, WHITE = (40, 214, 255), (255, 255, 255)              # BGR
SUB_SECONDS = 3.0
HOOK_SECONDS = 2.4
FONTS = [os.path.join(HERE, "fonts", "Poppins-Bold.ttf"), "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "DejaVuSans-Bold.ttf"]
_fonts, _spr = {}, {}


def font(sz):
    if sz not in _fonts:
        for f in FONTS:
            try:
                _fonts[sz] = ImageFont.truetype(f, sz)
                break
            except Exception:
                pass
        else:
            _fonts[sz] = ImageFont.load_default()
    return _fonts[sz]


def sprite(words, active=-1, size=112):
    """Caption chunk: har lafz safed, jo lafz abhi bol raha hai wo peela + mota kaala stroke + soft shadow -> BGRA float32 (cache)."""
    key = (tuple(words), active, size)
    if key in _spr:
        return _spr[key]
    if len(_spr) > 400:
        _spr.clear()
    f, stroke, pad = font(size), max(7, size // 9), 36
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    space = probe.textlength(" ", font=f) + stroke
    widths = [probe.textlength(w, font=f) for w in words]
    total = int(sum(widths) + space * (len(words) - 1)) + 2 * pad + 2 * stroke
    asc, desc = f.getmetrics()
    h = asc + desc + 2 * pad + 2 * stroke
    sh = Image.new("RGBA", (total, h), (0, 0, 0, 0))
    im = Image.new("RGBA", (total, h), (0, 0, 0, 0))
    x = pad + stroke
    for i, w in enumerate(words):
        fill = (255, 214, 40, 255) if i == active else (255, 255, 255, 255)
        ImageDraw.Draw(sh).text((x, pad + size // 14), w, font=f, fill=(0, 0, 0, 170), stroke_width=stroke, stroke_fill=(0, 0, 0, 170))
        ImageDraw.Draw(im).text((x, pad), w, font=f, fill=fill, stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
        x += widths[i] + space
    sh = sh.filter(ImageFilter.GaussianBlur(size // 18))
    a = np.asarray(Image.alpha_composite(sh, im)).astype(np.float32)
    _spr[key] = a[..., [2, 1, 0, 3]]                         # RGBA -> BGRA
    return _spr[key]


def card(lines, size, bg, fg, w=900, pad=34, radius=44, border=(255, 255, 255)):
    """Rounded text card (hook banner / lesson card) -> BGRA float32."""
    f = font(size)
    probe = ImageDraw.Draw(Image.new("RGBA", (10, 10)))
    rows = []
    for para in lines:                                       # har entry nayi line se shuru; andar word-wrap
        cur = ""
        for wd in para.split():
            t = (cur + " " + wd).strip()
            if probe.textlength(t, font=f) > w - 2 * pad and cur:
                rows.append(cur)
                cur = wd
            else:
                cur = t
        rows.append(cur)
    lh = int(size * 1.22)
    h = lh * len(rows) + 2 * pad
    im = Image.new("RGBA", (w + 20, h + 20), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([14, 16, w + 14, h + 16], radius=radius, fill=(0, 0, 0, 120))
    d.rounded_rectangle([6, 6, w + 6, h + 6], radius=radius, fill=bg + (255,), outline=border + (255,), width=6)
    for i, row in enumerate(rows):
        d.text(((w + 12) / 2, 6 + pad + lh * i + lh / 2), row, font=f, fill=fg + (255,), anchor="mm")
    return np.asarray(im).astype(np.float32)[..., [2, 1, 0, 3]]


def blit(frame, spr, cx, cy, scale=1.0, alpha=1.0):
    if scale != 1.0:
        spr = cv2.resize(spr, (max(2, int(spr.shape[1] * scale)), max(2, int(spr.shape[0] * scale))), interpolation=cv2.INTER_LINEAR)
    h, w = spr.shape[:2]
    x0, y0 = int(cx - w / 2), int(cy - h / 2)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(OW, x0 + w), min(OH, y0 + h)
    if xb <= xa or yb <= ya:
        return
    s = spr[ya - y0:yb - y0, xa - x0:xb - x0]
    a = s[..., 3:4] / 255.0 * alpha
    reg = frame[ya:yb, xa:xb].astype(np.float32)
    frame[ya:yb, xa:xb] = (reg * (1 - a) + s[..., :3] * a).astype(np.uint8)


def make_subscribe():
    im = Image.new("RGBA", (780, 230), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([10, 10, 770, 150], radius=70, fill=(215, 20, 25, 255), outline=(255, 255, 255, 255), width=6)
    d.polygon([(70, 45), (70, 115), (130, 80)], fill=(255, 255, 255, 255))
    d.text((165, 80), "SUBSCRIBE", font=font(66), fill=(255, 255, 255, 255), anchor="lm")
    d.rounded_rectangle([150, 168, 630, 220], radius=24, fill=(8, 8, 14, 215))
    d.text((390, 195), "FOLLOW FOR MORE STORIES", font=font(26), fill=(255, 214, 40, 255), anchor="mm")
    return np.asarray(im).astype(np.float32)[..., [2, 1, 0, 3]]


def caption_groups(infos):
    """Chunks of 1-3 lafz (<= 20 haroof), har lafz ka apna waqt -> [dict(start, end, words=[(txt, s, e)])] global seconds."""
    g = []
    for nf in infos:
        for ln in nf["lines"]:
            cur = []
            for w, a, b in ln["words"]:
                txt = w.upper()
                if cur and (len(cur) >= 3 or sum(len(x[0]) for x in cur) + len(txt) > 20 or cur[-1][0][-1:] in ".,!?;:"):
                    g.append(cur)
                    cur = []
                cur.append((txt, nf["start"] + a, nf["start"] + b))
            if cur:
                g.append(cur)
    out = [dict(start=c[0][1], end=c[-1][2], words=c) for c in g]
    for k in range(len(out) - 1):
        out[k]["end"] = min(out[k]["end"] + 0.18, out[k + 1]["start"] + 0.02) if out[k]["end"] + 0.18 > out[k + 1]["start"] else out[k]["end"] + 0.18
    return out


_S = {}


def _init(scenes, talks):
    cv2.setNumThreads(1)
    _S["s"], _S["t"] = scenes, talks


def _work(a):
    si, lf = a
    try:
        return np.ascontiguousarray(world.render_scene(lf, FPS, _S["s"][si], _S["t"][si]))
    except Exception:
        import traceback
        traceback.print_exc()
        return None


def fix_slots(scenes):
    """Room mein agar slot na ho (PART A ke ROOMS ke hisaab se) to 'center'."""
    rooms = getattr(world, "ROOMS", None)
    if not rooms:
        return
    for sc in scenes:
        rm = rooms.get(sc.get("room", "office")) or {}
        available = rm.get("slots", {})
        for c in sc["characters"]:
            if available and c["slot"] not in available:
                c["slot"] = "center"


def create(data, out_path):
    import story_gen
    t_start = time.time()
    try:
        scenes = story_gen.clean(data)["scenes"]                 # dobara validation (galat naam -> fallback)
    except Exception as e:
        print(f"[render] validation warning: {e} (script.json jaisa hai waisa use ho raha hai)")
        scenes = data["scenes"]
    fix_slots(scenes)
    tmp = tempfile.mkdtemp()
    seed = zlib.crc32(data.get("topic", "x").encode()) % 100000
    vox, infos, total = lipsync.build(scenes, tmp, FPS)
    total = min(total, MAX_SEC)
    sc2 = [dict(s, duration=nf["hold"]) for s, nf in zip(scenes, infos)]
    counts = [int(round(nf["hold"] * FPS)) for nf in infos]
    starts_f = np.concatenate([[0], np.cumsum(counts)])
    n_frames = min(int(starts_f[-1]), int(total * FPS))
    groups = caption_groups(infos)
    gstart = [g["start"] for g in groups]
    # ---- audio mix
    sub_t0 = max(0.0, total - SUB_SECONDS)
    v44 = np.interp(np.linspace(0, len(vox), int(len(vox) * music.SR / voice.SR), endpoint=False), np.arange(len(vox)), vox).astype(np.float32)
    ev = []
    for i, (s, nf) in enumerate(zip(scenes, infos)):
        if i:
            ev.append((nf["start"] + 0.02, "whoosh", 0.18))
        if s["camera_move"] == "closeup_shake":
            ev.append((nf["start"] + 0.05, "thud", 0.8))
    ev.append((sub_t0 + 0.05, "ding", 0.6))
    mixed, credit = music.mix(v44, total, data.get("music_mood", "quirky"), seed, ev)
    audio_path = os.path.join(tmp, "mix.wav")
    music.write_wav(audio_path, mixed)
    print(f"[render] audio ok ({total:.1f}s)", flush=True)
    # ---- frames (scenes ke frames parallel processes mein)
    tasks = [(si, lf) for si, c in enumerate(counts) for lf in range(c)][:n_frames]
    procs = max(1, int(os.environ.get("RENDER_PROCS", os.cpu_count() or 2)))
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OW}x{OH}", "-r", str(FPS), "-i", "-",
           "-i", audio_path, "-c:v", "libx264", "-preset", "veryfast", "-crf", "19", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
           "-t", f"{n_frames / FPS:.3f}", "-movflags", "+faststart", os.path.abspath(out_path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    sub = make_subscribe()
    hook_text = (data.get("hook_text") or "").strip()
    hook_spr = card([hook_text.upper()], 78, (40, 214, 255), (20, 14, 30), w=920) if hook_text else None
    fact = (data.get("fact") or "").strip()
    lesson_spr = card(["THE LESSON", fact], 52, (30, 24, 40), (255, 255, 255), w=900, border=(40, 214, 255)) if fact else None
    lesson_t0 = max(0.0, total - 5.2)
    faulthandler.dump_traceback_later(300, repeat=True)
    last, f = None, 0
    pool = mp.get_context("fork").Pool(procs, _init, (sc2, [nf["talk"] for nf in infos]))
    try:
        for fr in pool.imap(_work, tasks, chunksize=3):
            if fr is None:
                fr = last if last is not None else np.zeros((OH, OW, 3), np.uint8)
            last = fr
            if fr.shape[1] != OW or fr.shape[0] != OH:
                fr = cv2.resize(fr, (OW, OH), interpolation=cv2.INTER_CUBIC)
            else:
                fr = fr.copy()
            tg = f / FPS
            si = int(np.searchsorted(starts_f, f, side="right") - 1)
            lt = (f - starts_f[si]) / FPS
            if lt < 0.17:                                        # scene badalne par halka safed flash
                q = 1 - lt / 0.17
                fr = cv2.addWeighted(fr, 1 - 0.45 * q * q, np.full_like(fr, 255), 0.45 * q * q, 0)
            k = bisect_right(gstart, tg) - 1
            if k >= 0 and tg < groups[k]["end"] + 0.05:
                ws = groups[k]["words"]
                act = max(i for i, (_, s0, _) in enumerate(ws) if s0 <= tg + 1e-6) if any(s0 <= tg for _, s0, _ in ws) else 0
                spr = sprite([x[0] for x in ws], act)
                pop = (1.0 + 0.14 * max(0.0, 1 - (tg - ws[act][1]) / 0.11)) * min(1.0, 980 / spr.shape[1])
                blit(fr, spr, OW / 2, 0.80 * OH, pop)
            if tg < HOOK_SECONDS and hook_spr is not None:                 # hook banner: upar se slide-in, phir bahar
                pin = min(1.0, tg / 0.35)
                pout = min(1.0, max(0.0, (HOOK_SECONDS - tg) / 0.30))
                yy = (-0.12 + (0.2 + 0.12) * (1 - (1 - pin) ** 3)) * OH
                blit(fr, hook_spr, OW / 2, yy, 1.0 + 0.02 * np.sin(tg * 9), pout)
            if tg >= lesson_t0 and lesson_spr is not None and tg < sub_t0 + SUB_SECONDS:      # aakhri lesson card
                p = min(1.0, (tg - lesson_t0) / 0.40)
                blit(fr, lesson_spr, OW / 2, 0.40 * OH, (0.6 + 0.4 * (1 - (1 - p) ** 3)), min(1.0, p * 3))
            bar = int(OW * min(1.0, tg / max(total, 1e-6)))               # progress bar (neeche)
            if bar > 0:
                fr[OH - 14:OH - 4, 0:OW] = (fr[OH - 14:OH - 4, 0:OW] * 0.55).astype(np.uint8)
                fr[OH - 14:OH - 4, 0:bar] = (40, 214, 255)
            if tg >= sub_t0:
                p = (tg - sub_t0) / 0.45
                blit(fr, sub, OW / 2, 0.14 * OH, max(0.05, min(1.1, p)) * (1 + 0.025 * np.sin(tg * 6)), min(1.0, p * 3))
            try:
                proc.stdin.write(fr.tobytes())
            except BrokenPipeError:
                break
            f += 1
            if f % 90 == 0:
                print(f"[render] frame {f}/{n_frames} ({time.time() - t_start:.0f}s)", flush=True)
        proc.stdin.close()
    finally:
        pool.terminate()
        faulthandler.cancel_dump_traceback_later()
        try:
            proc.stdin.close()
        except Exception:
            pass
        rc = proc.wait()
    if rc != 0:
        sys.exit(f"ERROR: ffmpeg fail (code {rc})")
    print(f"Video created: {out_path} ({n_frames / FPS:.1f}s, {len(scenes)} scenes, {time.time() - t_start:.0f}s lagay)")
    return credit


def bisect_right(a, x):
    import bisect
    return bisect.bisect_right(a, x)


def main():
    if not os.path.exists("script.json"):
        sys.exit("script.json nahi mila (pehle story_gen.py chalayein)")
    data = json.load(open("script.json", encoding="utf-8"))
    credit = create(data, "output.mp4")
    if credit:
        data["description"] = (data.get("description", "") + "\n\n" + credit).strip()
        json.dump(data, open("script.json", "w", encoding="utf-8"), ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
