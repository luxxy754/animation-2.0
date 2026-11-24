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

API_KEY = os.environ.get("GEMINI_API_KEY")
MODELS = [m for m in [os.environ.get("GEMINI_MODEL"), "gemini-2.5-flash", "gemini-flash-latest", "gemini-2.5-flash-lite"] if m]
HISTORY, OUT = "history.json", "script.json"

# ---- CATALOG (CONTRACT): inke ilawa koi naam nahi chalta
ROOMS = ("office", "corridor", "break_room", "night_exterior")
MOVES = ("push_in", "pull_out", "orbit", "pan", "tilt", "through_door", "low_angle", "closeup_shake")
SLOTS = ("left", "center", "right", "far", "seat")
EXPRS = ("smile", "grin", "shock", "happy", "sad")
NAMES = tuple(CHARS)
SEAT_ROOMS = {"office"}                 # 'seat' slot sirf in rooms mein (PART A ke ROOMS se match rakho)
MOODS = ("quirky", "chill")
MAX_SEC, MAX_WORDS, WPS = 58.0, 125, 2.5   # short 60s se kam; WPS = bolne ki andazan raftaar (words/sec)

SITUATIONS = [
    "a meeting that could have been an email", "'quick sync?' invite at 5:55 pm on Friday", "the office fridge lunch thief",
    "someone microwaving fish in the break room", "the printer only works when the boss is watching", "'per my last email' passive aggression",
    "new hire accidentally replies-all to everyone", "Monday morning 'how was your weekend' small talk", "the thermostat war: freezing vs boiling",
    "the chair that slowly sinks all day", "pretending to be busy when the boss walks by", "the mystery smell near the cubicles",
    "wifi dies exactly during the big presentation", "the coworker who says 'as I was saying' after interrupting", "appraisal meeting where the raise is 'exposure'",
    "the one light that flickers over your desk", "leaving at 5:01 and everyone stares", "the secret santa gift nobody wanted",
    "someone eats your clearly labelled sandwich", "camera on / camera off video call chaos", "the elevator small talk that never ends",
    "a 'we're a family' company speech before layoffs rumour", "forgot to unmute and said it out loud", "the stapler that vanished from your desk",
    "the intern who fixes in 5 minutes what took you 2 days", "bring-your-own-snack day gone wrong", "the new open-plan seating chart drama",
    "the colleague who is always on a 'very important call'", "the fire drill during a deadline", "'we need to talk' message from the boss with no context",
    "the office plant that is thriving only because of one person", "password expires again for the 5th time", "the long hallway walk after being called by the boss",
    "the vending machine that eats your coins", "the 'optional' team outing that is not optional", "asking for leave and the boss says 'let me think'",
]
STRUCTURES = ["escalation: each exchange makes it worse", "rule of three with a twist on the third", "misunderstanding that snowballs",
              "calm start, absurd reveal, deadpan reaction", "false victory then instant reversal", "the boss is the one who gets outsmarted at the end"]
TONES = ["dry deadpan", "chaotic and panicky", "sarcastic", "wholesome but awkward", "dark office humour (clean)"]
HOOKS = ["a shocking one-line statement", "a weird question to the viewer", "a scene already mid-disaster", "a bold 'you know this feeling' line"]

RULES = f"""You write a 45-58 second vertical animated comedy sketch (YouTube Short) set mostly in ONE recurring office, with recurring characters. Relatable, funny, never boring.
Everything is told by dialogue between characters (no narrator). Output ONLY the JSON that matches the schema.
CATALOG (use these exact names only):
- room: {', '.join(ROOMS)}. 'office' is the main hub (use it for ~60% of scenes); corridor = hallway chats/walks; break_room = coffee/snack talk; night_exterior = night view of the building, good for an opening/closing beat.
- camera_move: {', '.join(MOVES)}. Vary them; never the same move twice in a row. closeup_shake = shocked/panic moments, through_door = entering a room, push_in = punchlines.
- characters: {'; '.join(f"{k} ({v['bio']})" for k, v in CHARS.items())}
- slot (floor position in the room): left, center, right, far (back of room, small), seat (sitting at a desk; ONLY in office). Two characters in one scene: use left + right. Never two characters in the same slot.
- expr: {', '.join(EXPRS)}.
STRUCTURE: 10-14 scenes. Scene 1 = a hook that grabs in 2 seconds (a bold or weird line). The last scene = the punchline (short). 1-3 characters per scene, each character speaks AT MOST one short line per scene (max 12 words, spoken English, contractions, no emojis, no stage directions).
You may use a silent reaction scene (empty line) at most twice. Total spoken words across the whole sketch: {MAX_WORDS - 30}-{MAX_WORDS}. 'duration' in seconds (2-5) should roughly match the line length.
Use the SAME characters in recurring roles (the boss is the boss); pick 2-4 characters per sketch. Make the expr match the line (shock for a surprise, grin for the boss's fake-nice smile, sad for defeat, happy for a win).
TITLE: 35-70 chars, curiosity or relatable (no clickbait lies). DESCRIPTION: one hook sentence + 3-5 hashtags (#office #relatable #funny ...). TAGS: 6-10 short tags. TOPIC: one short specific premise (for repeat avoidance).
music_mood: quirky or chill."""


def schema(strict=True):
    # strict=False: enums hata deta hai (Gemini kabhi 'schema too complex' 400 deta hai); clean() phir bhi sab theek karta hai
    S = lambda **k: {"type": "STRING", **{a: b for a, b in k.items() if strict or a != "enum"}}
    ch = {"type": "OBJECT", "properties": {"name": S(enum=list(NAMES)), "slot": S(enum=list(SLOTS)), "expr": S(enum=list(EXPRS)), "line": S()},
          "required": ["name", "slot", "expr", "line"]}
    sc = {"type": "OBJECT", "properties": {"room": S(enum=list(ROOMS)), "camera_move": S(enum=list(MOVES)), "duration": {"type": "NUMBER"},
                                           "characters": {"type": "ARRAY", "items": ch}}, "required": ["room", "camera_move", "duration", "characters"]}
    return {"type": "OBJECT", "properties": {"topic": S(), "title": S(), "description": S(), "tags": {"type": "ARRAY", "items": S()},
            "music_mood": S(enum=list(MOODS)), "scenes": {"type": "ARRAY", "items": sc}},
            "required": ["topic", "title", "description", "tags", "music_mood", "scenes"]}


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
    """Gemini ka jawab -> CONTRACT scene JSON. Galat room/move/name/expr/slot par fallback; lambai 58s se andar."""
    scenes, prev_move = [], None
    for raw in (d.get("scenes") or []):
        room = _pick(raw.get("room"), ROOMS, "office")
        move = _pick(raw.get("camera_move"), MOVES, "push_in")
        if move == prev_move:                                        # lagatar ek hi camera move nahi
            move = MOVES[(MOVES.index(move) + 3) % len(MOVES)]
        prev_move = move
        try:
            dur = min(max(float(raw.get("duration", 3)), 1.5), 6.0)
        except (TypeError, ValueError):
            dur = 3.0
        chars, seen, used = [], set(), set()
        for c in (raw.get("characters") or [])[:3]:
            nm = _name(c.get("name"))
            if not nm or nm in seen:
                continue
            seen.add(nm)
            slot = _pick(c.get("slot"), SLOTS, "center")
            if slot == "seat" and room not in SEAT_ROOMS:
                slot = "center"
            chars.append({"name": nm, "slot": slot, "expr": _pick(c.get("expr"), EXPRS, "smile"), "line": _line(c.get("line"))})
        for c in chars:                                               # ek slot par do character nahi
            if c["slot"] in used:
                c["slot"] = next((s for s in ("center", "left", "right", "far") if s not in used), "far")
            used.add(c["slot"])
        if len(chars) == 2 and {chars[0]["slot"], chars[1]["slot"]} == {"center"}:
            chars[0]["slot"], chars[1]["slot"] = "left", "right"
        if chars:
            scenes.append({"room": room, "camera_move": move, "duration": round(dur, 1), "characters": chars})
    talk = [s for s in scenes if any(c["line"] for c in s["characters"])]
    if len(talk) < 5 or len({c["name"] for s in scenes for c in s["characters"]}) < 2:
        raise ValueError("story too short/one character only")
    words = sum(len(c["line"].split()) for s in scenes for c in s["characters"])
    total = sum(est_sec(s) for s in scenes)
    if total > MAX_SEC + 8 or words > MAX_WORDS + 25:
        raise ValueError(f"too long: ~{total:.0f}s, {words} words (max {MAX_SEC:.0f}s / {MAX_WORDS} words)")
    while total > MAX_SEC and len(scenes) > 6:                        # thodi si zyada ho to beech ka scene hata do (aakhri punchline bachta hai)
        k = max(range(1, len(scenes) - 1), key=lambda i: est_sec(scenes[i]) if not any(c["line"] for c in scenes[i]["characters"]) else est_sec(scenes[i]) - 2)
        total -= est_sec(scenes.pop(k))
    tags = [re.sub(r"[^\w]", "", str(t))[:30] for t in (d.get("tags") or [])][:10]
    return {"topic": str(d.get("topic") or "office comedy")[:140], "title": str(d.get("title") or "Every Office Ever")[:95],
            "description": str(d.get("description") or "#office #relatable #funny")[:900], "tags": [t for t in tags if t] or ["office", "funny", "relatable"],
            "music_mood": _pick(d.get("music_mood"), MOODS, "quirky"), "scenes": scenes}


def _S(room, move, dur, *cs):
    return {"room": room, "camera_move": move, "duration": dur, "characters": [dict(name=n, slot=s, expr=e, line=l) for n, s, e, l in cs]}


FALLBACK = {"topic": "boss calls you in on Monday", "title": "When The Boss Says 'Quick Chat?'", "description": "You know that walk. #office #relatable #funny #Shorts",
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
    idea = ("Situation: invent a FRESH relatable everyday work/office situation yourself (not a cliche like 'meeting could be an email')." if wild
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
