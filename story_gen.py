"""Gemini se naya relatable office-comedy short ka scene JSON (CONTRACT format) banata hai -> script.json.
Sirf catalog ke naam allowed hain; galat naam validation mein fallback se theek hote hain."""
import datetime
import json
import os
import random
import re
import sys
import time

import requests

from toon.characters import CHARS
from toon.props import PROP_NAMES
from toon.world_builder import validate_world_spec, nearest_fixed_room
from toon.rig import EXPRS as RIG_EXPRS

API_KEY = os.environ.get("GEMINI_API_KEY")
MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"] if m]
HISTORY, OUT = "history.json", "script.json"

# ---- CATALOG (CONTRACT): inke ilawa koi naam nahi chalta
ROOMS = ("office", "corridor", "break_room", "night_exterior")
MOVES = ("push_in", "pull_out", "orbit", "pan", "tilt", "through_door", "walk_through", "low_angle", "closeup_shake")
SLOTS = ("left", "center", "right", "far", "seat")
EXPRS = tuple(RIG_EXPRS)
GESTURES = ("idle", "wave", "point", "shrug", "cheer", "clap", "facepalm", "sad", "type", "talk")
FACING = ("front", "left", "right", "three_quarter")
LOCATIONS = tuple(ROOMS) + ("custom",)
NAMES = tuple(CHARS)
SEAT_ROOMS = {"office"}                 # 'seat' slot sirf in rooms mein (PART A ke ROOMS se match rakho)
MOODS = ("quirky", "chill")
MAX_SEC, MAX_WORDS, WPS = 58.0, 125, 2.5   # short 60s se kam; WPS = bolne ki andazan raftaar (words/sec)

SITUATIONS = [
    "the lift stops between floors with two strangers inside", "a chai order gets mixed up at a busy stall",
    "a student brings the wrong project to class", "a neighbour alarm rings at 5 AM every day",
    "a family tries to take one perfect group photo", "a bus arrives just as someone sits down",
    "a shopkeeper recognises a customer who forgot their wallet", "a picnic gets interrupted by curious birds",
    "a person waves back at someone who was waving behind them", "a rooftop kite gets tangled in laundry",
    "a restaurant menu has one item nobody can pronounce", "a tourist confidently walks the wrong way",
    "a phone battery reaches one percent during an important call", "a dog steals the only sandwich at the park",
    "a rainstorm starts seconds after laundry is hung outside", "a school group project has one invisible teammate",
    "a person tries to look calm while carrying too many grocery bags", "a queue moves faster in the other line",
    "a cafe runs out of the one pastry everyone came for", "a beach umbrella blows away at the worst moment",
    "a wedding guest mistakes the family photographer for a waiter", "a delivery arrives at the wrong door",
    "a person gets locked out while holding the keys inside", "a street vendor gives the exact wrong change",
    "a kid teaches an adult how to use a smartphone", "a friend says five minutes but arrives an hour later",
    "a bus conductor and passenger debate the last seat", "a power cut interrupts a dramatic story",
    "a person practises a speech and the audience is a cat", "a supermarket trolley has one stubborn wheel",
    "someone tries to quietly open a noisy snack packet", "a family argues over the air conditioner remote",
    "a person forgets why they entered a room", "a shop sign has a hilarious spelling mistake",
    "a park jogger gets followed by an overenthusiastic puppy", "a school bell rings during the wrong class",
    "a customer asks for a tiny discount and starts a negotiation", "a person tries to catch a falling hat in the wind",
    "a neighbour borrows a ladder and returns with a strange story", "a cafe table wobbles during a serious conversation"
]
STRUCTURES = ["escalation: each exchange makes it worse", "rule of three with a twist on the third", "misunderstanding that snowballs",
              "calm start, absurd reveal, deadpan reaction", "false victory then instant reversal", "the boss is the one who gets outsmarted at the end"]
TONES = ["dry deadpan", "chaotic and panicky", "sarcastic", "wholesome but awkward", "dark office humour (clean)"]
HOOKS = ["a shocking one-line statement", "a weird question to the viewer", "a scene already mid-disaster", "a bold 'you know this feeling' line"]

RULES = f"""Write a funny, relatable 45-58 second vertical comedy Short. The story may happen ANYWHERE: home, school, street, park, rooftop, cafe, airport, beach, jungle, shop, office, or another everyday place. Do not default to office.
Tell the story through character dialogue only; no narrator. Output ONLY JSON.
FIXED CAST (exact names): {', '.join(CHARS.keys())}
FIXED LOCATIONS: {', '.join(ROOMS)}. Use these for ordinary recurring places.
For a new location use location="custom" and include world_spec with env indoor/outdoor; size w,d,h; time day/night/sunset/overcast; weather clear/rain/snow/fog; ground type carpet/tile/wood/grass/road/sand/snow/water/concrete and RGB color; walls pattern plain/brick/tile/wood and RGB color; lights (kind and pos); props (exact catalog name, pos [x,z], rot, scale); custom_props (name and 1-25 primitive parts); slots.
PROP CATALOG (exact names only): {', '.join(PROP_NAMES)}
CAMERA MOVES: {', '.join(MOVES)}. Vary moves; outdoor scenes prefer walk_through.
EXPRESSIONS: {', '.join(EXPRS)}. GESTURES: {', '.join(GESTURES)}. FACING: {', '.join(FACING)}.
Slots: left, center, right, far, seat. Never put two characters in the same slot.
Create 10-14 scenes. Scene 1 must hook within 2 seconds; last scene is a short punchline.
Each scene has 1-3 characters; each speaks at most one short line (max 12 words). At most two silent reaction scenes. Total dialogue 95-125 words; duration 2-5 seconds.
Use 2-4 recurring characters and keep roles consistent. Title 35-70 chars, description one hook sentence plus 3-5 relevant hashtags, tags 6-10. Topic must be specific to prevent repetition. music_mood quirky or chill.
Scene fields: location, optional world_spec, camera_move, duration, characters. Character fields: name, slot, expr, optional gesture, facing, props, line.
"""


def schema(strict=True):
    S = lambda **k: {"type": "STRING", **{a:b for a,b in k.items() if strict or a != "enum"}}
    ch = {"type":"OBJECT","properties":{
        "name":S(enum=list(NAMES)),"slot":S(enum=list(SLOTS)),"expr":S(enum=list(EXPRS)),
        "gesture":S(enum=list(GESTURES)),"facing":S(enum=list(FACING)),
        "props":{"type":"ARRAY","items":S(enum=["phone","key","mug","paper","laptop","bag","umbrella","food"])},"line":S()
    },"required":["name","slot","expr","line"]}
    ws = {"type":"OBJECT","properties":{
        "env":S(enum=["indoor","outdoor"]),
        "size":{"type":"OBJECT","properties":{"w":{"type":"NUMBER"},"d":{"type":"NUMBER"},"h":{"type":"NUMBER"}},"required":["w","d","h"]},
        "time":S(enum=["day","night","sunset","overcast"]),"weather":S(enum=["clear","rain","snow","fog"]),
        "ground":{"type":"OBJECT","properties":{"type":S(enum=["carpet","tile","wood","grass","road","sand","snow","water","concrete"]),"color":{"type":"ARRAY","items":{"type":"INTEGER"}}}},
        "walls":{"type":"OBJECT","properties":{"pattern":S(enum=["plain","brick","tile","wood"]),"color":{"type":"ARRAY","items":{"type":"INTEGER"}}}},
        "sky":{"type":"OBJECT","properties":{"gradient":{"type":"ARRAY","items":{"type":"ARRAY","items":{"type":"INTEGER"}}}}},
        "lights":{"type":"ARRAY","items":{"type":"OBJECT","properties":{"kind":S(enum=["sun","panel","lamp","street_lamp","fire"]),"pos":{"type":"ARRAY","items":{"type":"NUMBER"}},"color":{"type":"ARRAY","items":{"type":"INTEGER"}},"intensity":{"type":"NUMBER"}},"required":["kind","pos"]}},
        "props":{"type":"ARRAY","items":{"type":"OBJECT","properties":{"prop":S(enum=list(PROP_NAMES)),"pos":{"type":"ARRAY","items":{"type":"NUMBER"}},"rot":{"type":"NUMBER"},"scale":{"type":"NUMBER"},"color":{"type":"ARRAY","items":{"type":"INTEGER"}}},"required":["prop","pos"]}},
        "custom_props":{"type":"ARRAY","items":{"type":"OBJECT","properties":{"name":S(),"parts":{"type":"ARRAY","items":{"type":"OBJECT","properties":{"shape":S(enum=["box","cylinder","sphere","cone","quad"]),"pos":{"type":"ARRAY","items":{"type":"NUMBER"}},"size":{"type":"ARRAY","items":{"type":"NUMBER"}},"color":{"type":"ARRAY","items":{"type":"INTEGER"}}},"required":["shape","pos","size","color"]}}},"required":["name","parts"]}},
        "slots":{"type":"ARRAY","items":S(enum=list(SLOTS))}
    },"required":["env","size","time","weather","ground","walls"]}
    sc={"type":"OBJECT","properties":{"location":S(enum=list(LOCATIONS)),"world_spec":ws,"camera_move":S(enum=list(MOVES)),
        "duration":{"type":"NUMBER"},"characters":{"type":"ARRAY","items":ch}},
        "required":["location","camera_move","duration","characters"]}
    return {"type":"OBJECT","properties":{"topic":S(),"title":S(),"description":S(),"tags":{"type":"ARRAY","items":S()},
        "music_mood":S(enum=list(MOODS)),"scenes":{"type":"ARRAY","items":sc}},
        "required":["topic","title","description","tags","music_mood","scenes"]}



def ask(prompt, sch, max_out=8000, think=1024):
    last = None
    for model in MODELS:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        cfg = {"temperature": 1.0, "maxOutputTokens": max_out, "responseMimeType": "application/json", "responseSchema": sch}
        if "2.5-flash" in model:
            cfg["thinkingConfig"] = {"thinkingBudget": think}
            cfg["maxOutputTokens"] = max_out + think
        for attempt in range(3):
            try:
                r = requests.post(url, headers={"x-goog-api-key": API_KEY}, json={"contents": [{"parts": [{"text": prompt}]}], "generationConfig": cfg}, timeout=240)
                if r.status_code == 200:
                    cands = r.json().get("candidates") or []
                    parts = (cands[0].get("content") or {}).get("parts") or [] if cands else []
                    txt = "".join(p.get("text", "") for p in parts).strip()
                    if not txt:
                        last = f"{model}: empty answer"
                        continue
                    return json.loads(re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip())
                last = f"{model}: HTTP {r.status_code} {r.text[:160]}"
                if r.status_code in (400, 401, 403, 404):
                    break
                time.sleep(20 if r.status_code == 429 else 5 * (attempt + 1))
            except Exception as e:
                last = f"{model}: {e}"
                time.sleep(4)
    raise RuntimeError(last)


def _words(s):
    return {re.sub(r"(ed|s)$", "", w) for w in re.findall(r"[a-z0-9]+", s.lower()) if len(w) > 3}


def too_similar(topic, past, thr=0.4):
    """Same idea naye alfaaz mein dobara na aaye (exact match ke ilawa overlap bhi dekhta hai)."""
    a = _words(topic)
    for p in past:
        b = _words(p)
        if a and b and len(a & b) / len(a | b) >= thr:
            return True
    return False


def load_history():
    try:
        return json.load(open(HISTORY, encoding="utf-8"))
    except Exception:
        return []


def _pick(v, allowed, default):
    return v if v in allowed else default


def _name(n):
    """Galat/ajeeb naam -> sabse milta catalog naam (prefix/substring), warna None."""
    n = re.sub(r"[^a-z]", "", str(n or "").lower())
    if n in CHARS:
        return n
    return next((k for k in CHARS if n and (k.startswith(n) or n.startswith(k) or CHARS[k]["name"].lower() == n)), None)


def _line(s):
    s = re.sub(r"[^\w\s.,!?'\-]", " ", str(s or "").replace("\n", " "))
    s = re.sub(r"\s+", " ", s).strip()
    w = s.split()
    if len(w) > 14:
        s = " ".join(w[:14]).rstrip(",;:-") + "."
    return s


def est_sec(sc):
    """Scene ki andazan lambai (lipsync.py jaisi): bolne ka waqt + thoda reaction."""
    ls = [c["line"] for c in sc["characters"] if c["line"]]
    if not ls:
        return min(max(sc["duration"], 1.2), 4.0)
    voiced = sum(len(l.split()) / WPS + 0.18 for l in ls) + 0.3
    return max(voiced + 0.35, min(sc["duration"], voiced + 1.2))


def clean(d):
    """Gemini response ko safe renderer contract mein normalize karo."""
    scenes, prev_move = [], None
    for raw in (d.get("scenes") or [])[:18]:
        raw_spec = raw.get("world_spec")
        location = raw.get("location", raw.get("room", "office"))
        validated_spec = validate_world_spec(raw_spec) if isinstance(raw_spec, dict) else None
        world_spec = raw_spec if validated_spec is not None else None
        if isinstance(raw_spec, dict) and world_spec is None:
            location = nearest_fixed_room(raw_spec)
        if location not in ROOMS and location != "custom":
            location = "office"
        if location == "custom" and world_spec is None:
            location = "office"
        room = location if location in ROOMS else nearest_fixed_room(world_spec)
        move = _pick(raw.get("camera_move"), MOVES, "push_in")
        if move == prev_move:
            move = MOVES[(MOVES.index(move) + 3) % len(MOVES)]
        prev_move = move
        try:
            dur = min(max(float(raw.get("duration",3)),1.5),6.0)
        except (TypeError,ValueError):
            dur = 3.0
        chars, seen, used = [], set(), set()
        for c in (raw.get("characters") or [])[:3]:
            nm = _name(c.get("name"))
            if not nm or nm in seen:
                continue
            seen.add(nm)
            slot = _pick(c.get("slot"),SLOTS,"center")
            if slot == "seat" and (world_spec is not None or room not in SEAT_ROOMS):
                slot = "center"
            hand_props = [v for v in (c.get("props") or []) if v in ("phone","key","mug","paper","laptop","bag","umbrella","food")][:4]
            chars.append({"name":nm,"slot":slot,"expr":_pick(c.get("expr"),EXPRS,"smile"),
                "gesture":_pick(c.get("gesture"),GESTURES,"idle"),"facing":_pick(c.get("facing"),FACING,"front"),
                "props":hand_props,"line":_line(c.get("line"))})
        for c in chars:
            if c["slot"] in used:
                c["slot"] = next((v for v in ("center","left","right","far") if v not in used),"far")
            used.add(c["slot"])
        if chars:
            item={"room":room,"location":location,"camera_move":move,"duration":round(dur,1),"characters":chars}
            if world_spec is not None:
                item["world_spec"]=world_spec
            scenes.append(item)
    talk=[sc for sc in scenes if any(c["line"] for c in sc["characters"])]
    if len(talk)<5 or len({c["name"] for sc in scenes for c in sc["characters"]})<2:
        raise ValueError("story too short/one character only")
    words=sum(len(c["line"].split()) for sc in scenes for c in sc["characters"])
    total=sum(est_sec(sc) for sc in scenes)
    if total>MAX_SEC+8 or words>MAX_WORDS+25:
        raise ValueError(f"too long: ~{total:.0f}s, {words} words (max {MAX_SEC:.0f}s / {MAX_WORDS} words)")
    while total>MAX_SEC and len(scenes)>6:
        k=max(range(1,len(scenes)-1),key=lambda i:est_sec(scenes[i]) if not any(c["line"] for c in scenes[i]["characters"]) else est_sec(scenes[i])-2)
        total-=est_sec(scenes.pop(k))
    tags=[re.sub(r"[^\w]","",str(tag))[:30] for tag in (d.get("tags") or [])][:10]
    return {"topic":str(d.get("topic") or "relatable comedy")[:140],"title":str(d.get("title") or "Everyday Life Is Complicated")[:95],
        "description":str(d.get("description") or "#relatable #comedy #shorts")[:900],
        "tags":[tag for tag in tags if tag] or ["comedy","relatable","shorts"],
        "music_mood":_pick(d.get("music_mood"),MOODS,"quirky"),"scenes":scenes}



def _S(room, move, dur, *cs):
    return {"room": room, "camera_move": move, "duration": dur, "characters": [dict(name=n, slot=s, expr=e, line=l) for n, s, e, l in cs]}


FALLBACK = {"topic": "a surprise problem at an everyday place", "title": "When The Boss Says 'Quick Chat?'", "description": "You know that walk. #office #relatable #funny #Shorts",
            "tags": ["office", "funny", "relatable", "boss", "monday", "work"], "music_mood": "quirky", "scenes": [
    _S("office", "push_in", 3, ("arjun", "seat", "smile", "Everything is fine. Totally normal Monday.")),
    _S("office", "pan", 3, ("zoya", "left", "smile", "Did you see the boss's calendar?"), ("arjun", "right", "shock", "Why would I look at that?")),
    _S("corridor", "orbit", 3, ("boss", "center", "grin", "Arjun! My office. Now. Quick chat!")),
    _S("office", "closeup_shake", 2.5, ("arjun", "center", "shock", "Quick chat means I'm fired.")),
    _S("corridor", "through_door", 3, ("arjun", "left", "sad", "Any last words, Zoya?"), ("zoya", "right", "smile", "Can I have your chair?")),
    _S("office", "low_angle", 3, ("boss", "center", "grin", "Great news! You're getting a promotion.")),
    _S("office", "push_in", 3, ("arjun", "left", "happy", "Seriously?"), ("boss", "right", "grin", "Yes. Same salary, bigger title.")),
    _S("break_room", "pull_out", 3, ("arjun", "center", "sad", "I'm now Senior Vice Intern."))]}


def pick_ingredients(used):
    """Har run bilkul naya combo: situation (aaj tak use na hui) + structure + tone + hook + protagonist; 25% wildcard."""
    rnd = random.SystemRandom()
    forced = os.environ.get("FORCE_SEED", "").strip()
    pool = [forced] if forced else ([s for s in SITUATIONS if s not in used] or SITUATIONS)
    wild = (not forced) and rnd.random() < 0.25
    seed = "wildcard" if wild else rnd.choice(pool)
    idea = ("Situation: invent a FRESH relatable everyday-life situation yourself (not a cliche like 'meeting could be an email')." if wild
            else f"Situation (sharpen it, make it specific and surprising): {seed}.")
    cast = rnd.sample(NAMES, rnd.choice((2, 3)))
    return seed, (f"TODAY'S INGREDIENTS. {idea} Comedy structure: {rnd.choice(STRUCTURES)}. Tone: {rnd.choice(TONES)}. Opening: {rnd.choice(HOOKS)}. "
                  f"Suggested cast (you may add one): {', '.join(cast)}; protagonist = {cast[0]}.")


def main():
    if os.path.exists(OUT):
        os.remove(OUT)
    hist = load_history()
    recent = [h.get("topic", "") for h in hist]
    used = {h.get("seed") for h in hist if h.get("seed")}
    d, fb, seed_used, err = None, False, "", ""
    if API_KEY:
        for attempt in range(5):
            seed, spec = pick_ingredients(used)
            prompt = (f"{RULES}\n\n{spec}\nDo NOT repeat or resemble these past sketches: {recent[-50:]}\n"
                      f"{('Your previous answer failed: ' + err + '. Fix it. ') if err else ''}Return JSON only.")
            try:
                cand = clean(ask(prompt, schema(strict="HTTP 400" not in err)))
                if too_similar(cand["topic"] + " " + cand["title"], recent):
                    recent.append(cand["topic"])
                    err = "topic too similar to a past one"
                    print(f"attempt {attempt + 1}: {err} -> retry")
                    continue
                d, seed_used = cand, seed
                break
            except Exception as e:
                err = str(e)
                print(f"attempt {attempt + 1} failed: {e}")
                time.sleep(3)
    if not d:
        if not os.environ.get("ALLOW_FALLBACK"):
            sys.exit("ERROR: Gemini se nayi story nahi mili (GEMINI_API_KEY/quota dekho). Duplicate video upload na ho isliye ruk gaya.")
        print("ALLOW_FALLBACK -> offline story (history mein save nahi hogi)")
        d, fb = clean(FALLBACK), True
    json.dump(d, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("SCRIPT:", d["title"], f"({len(d['scenes'])} scenes, {sum(len(c['line'].split()) for s in d['scenes'] for c in s['characters'])} words, ~{sum(est_sec(s) for s in d['scenes']):.0f}s)")
    if not fb and not any(h.get("topic", "").lower() == d["topic"].lower() for h in hist):
        hist.append({"topic": d["topic"], "title": d["title"], "seed": seed_used, "date": datetime.date.today().isoformat()})
        json.dump(hist[-1000:], open(HISTORY, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
