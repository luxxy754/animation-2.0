"""Gemini se naya PARABLE-EXPLAINER short (paisa / psychology / society / history ka ek idea, chhoti allegory duniya mein) ka scene JSON banata hai -> script.json.
Office-comedy nahi. Sirf catalog ke naam allowed hain; galat naam validation mein fallback se theek hote hain."""
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
GESTURES = ("idle", "wave", "point", "point_up", "shrug", "cheer", "clap", "facepalm", "sad", "type", "talk", "think", "cross", "hold_up", "bow", "jump")
HAND_PROPS = ("phone", "key", "mug", "paper", "laptop", "bag", "umbrella", "food", "coin", "money_bag", "bread", "fish", "apple")
FACING = ("front", "left", "right", "three_quarter")
LOCATIONS = tuple(ROOMS) + ("custom",)
NAMES = tuple(CHARS)
SEAT_ROOMS = {"office"}                 # parable places mein seat nahi
MOODS = ("quirky", "chill")
MAX_SEC, MAX_WORDS, WPS = 58.0, 135, 2.5   # short 60s se kam; WPS = bolne ki andazan raftaar (words/sec)

# ======================================================================================================== PLACES (ready-made parable duniyaen)
# Gemini sirf "place" ka naam chunta hai; world_spec code banata hai (isse background hamesha valid + rangeen rehta hai). RGB colours.
def _P(env, w, d, h, time, ground, gcol, wall, wcol, props, lights=None, weather="clear", sky=None):
    spec = {"env": env, "size": {"w": w, "d": d, "h": h}, "time": time, "weather": weather, "ground": {"type": ground, "color": gcol},
            "walls": {"pattern": wall, "color": wcol}, "lights": lights or [], "slots": ["left", "center", "right", "far"], "preset": True,
            "props": [{"prop": n, "pos": [round(x * .62, 2), z], "rot": rot, "scale": sc} for n, x, z, rot, sc in props], "custom_props": []}
    if sky:
        spec["sky"] = {"gradient": sky}
    return spec


SUN = [{"kind": "sun", "pos": [6, 12, 4], "color": [255, 240, 205], "intensity": 1.0}]
PRESETS = {
    "village_market": _P("outdoor", 16, 20, 6, "day", "road", [190, 150, 105], "plain", [230, 200, 160],
                         [("shop_front", -6.5, 15, 0, 1.4), ("chai_stall", 6.0, 14, 0, 1.2), ("tree", -7, 8, 0, 1.3), ("fountain", 0, 16, 0, 1.1),
                          ("bench", 6.8, 9, 0, 1), ("street_lamp", -3, 13, 0, 1), ("fence", 7, 18, 0, 1), ("bush", -7, 12, 0, 1)], SUN),
    "island": _P("outdoor", 18, 22, 6, "day", "sand", [238, 208, 140], "plain", [200, 200, 200],
                 [("palm_tree", -7, 10, 0, 1.4), ("palm_tree", 6.5, 12, 0, 1.3), ("palm_tree", -4, 17, 0, 1.2), ("rock_large", 7, 7.5, 0, 1),
                  ("boat", 0, 19, 0, 1.3), ("campfire", -6, 6.5, 0, 1), ("rock", 3.5, 15, 0, 1)], SUN),
    "farm": _P("outdoor", 18, 22, 6, "day", "grass", [110, 190, 80], "plain", [200, 200, 200],
               [("fence", -7, 14, 0, 1.2), ("fence", 0, 17, 0, 1.2), ("fence", 7, 14, 0, 1.2), ("tree", -7.5, 8, 0, 1.4), ("water_tank", 7, 10, 0, 1.2),
                ("bush", 5, 18, 0, 1), ("flower", -3, 16, 0, 1), ("mailbox", 7.5, 6.5, 0, 1)], SUN),
    "city_street": _P("outdoor", 18, 22, 8, "day", "road", [95, 100, 110], "plain", [200, 200, 200],
                      [("building_block", -7, 17, 0, 1.6), ("building_block", 0, 19, 0, 1.8), ("building_block", 7, 17, 0, 1.5), ("bus_stop", -7, 9, 0, 1.1),
                       ("traffic_light", 7.5, 8, 0, 1), ("street_lamp", 4, 12, 0, 1), ("atm", 6.5, 13.5, 0, 1), ("car", -2, 15, 0, 1)], SUN),
    "park": _P("outdoor", 18, 22, 6, "day", "grass", [120, 200, 95], "plain", [200, 200, 200],
               [("fountain", 0, 17, 0, 1.2), ("bench", -7, 10, 0, 1), ("tree", 7, 9, 0, 1.4), ("tree", -7.5, 15, 0, 1.3), ("bush", 6, 16, 0, 1),
                ("picnic_table", 7, 13, 0, 1), ("flower", -3, 14, 0, 1)], SUN),
    "school_yard": _P("outdoor", 18, 22, 6, "day", "concrete", [170, 175, 180], "plain", [200, 200, 200],
                      [("playground", -6.5, 15, 0, 1.2), ("swing", 6.5, 14, 0, 1.2), ("slide", 0, 18, 0, 1.2), ("bench", -7, 8, 0, 1), ("tree", 7.5, 8, 0, 1.3)], SUN),
    "campfire_night": _P("outdoor", 16, 20, 6, "night", "grass", [40, 80, 60], "plain", [60, 60, 80],
                         [("campfire", 0, 14, 0, 1.3), ("tent", -6, 14, 0, 1.3), ("tree", 7, 12, 0, 1.5), ("tree", -7.5, 8, 0, 1.4), ("rock", 5, 16, 0, 1)],
                         [{"kind": "fire", "pos": [0, 1.2, 14], "color": [255, 160, 70], "intensity": 1.4}]),
    "mountain_town": _P("outdoor", 18, 22, 8, "overcast", "snow", [235, 240, 250], "plain", [200, 200, 200],
                        [("mountain", -5, 20, 0, 1.8), ("mountain", 6, 20.5, 0, 1.6), ("tree", -7, 9, 0, 1.3), ("tent", 7, 12, 0, 1.2), ("sign", 5, 7.5, 0, 1)], SUN),
    "harbor": _P("outdoor", 18, 22, 6, "sunset", "concrete", [150, 150, 160], "plain", [200, 200, 200],
                 [("boat", -5, 19, 0, 1.5), ("boat", 5, 20, 0, 1.3), ("bridge", 0, 17, 0, 1.2), ("street_lamp", 7, 9, 0, 1), ("rock_large", -7, 8, 0, 1)],
                 [{"kind": "sun", "pos": [10, 5, 6], "color": [255, 190, 120], "intensity": 1.0}]),
    "classroom": _P("indoor", 12, 14, 4, "day", "wood", [190, 140, 90], "plain", [190, 225, 195],
                    [("whiteboard", 0, 13.7, 0, 1.6), ("school_desk", -3.5, 11, 0, 1), ("school_desk", 3.5, 11, 0, 1), ("bookshelf", -5.5, 13, 0, 1.1),
                     ("clock", 3.5, 13.7, 0, 1), ("plant", 5.2, 12.5, 0, 1)], [{"kind": "panel", "pos": [0, 3.6, 6], "color": [255, 250, 240], "intensity": 1.0}]),
    "living_room": _P("indoor", 12, 14, 4, "day", "wood", [175, 120, 80], "plain", [245, 215, 160],
                      [("sofa", 0, 13, 0, 1.3), ("tv", 4.2, 13.5, 0, 1.2), ("bookshelf", -5, 13, 0, 1.1), ("lamp", -3, 12.5, 0, 1), ("rug", 0, 7.5, 0, 1.6), ("plant", 5.3, 11, 0, 1)],
                      [{"kind": "lamp", "pos": [-3, 2.2, 12], "color": [255, 200, 130], "intensity": 1.0}, {"kind": "panel", "pos": [0, 3.6, 6], "color": [255, 240, 220], "intensity": .6}]),
    "kitchen": _P("indoor", 12, 14, 4, "day", "tile", [235, 230, 215], "tile", [250, 235, 190],
                  [("fridge", -5, 13, 0, 1.2), ("kitchen_counter", 0, 13.5, 0, 1.6), ("table", 4, 9.5, 0, 1.1), ("window", 5.5, 13.7, 0, 1.2), ("mug", 4, 9.5, 0, 1)],
                  [{"kind": "panel", "pos": [0, 3.6, 6], "color": [255, 250, 240], "intensity": 1.0}]),
    "shop_inside": _P("indoor", 12, 14, 4, "day", "wood", [160, 115, 80], "wood", [200, 150, 105],
                      [("cabinet", -4.5, 13, 0, 1.4), ("cabinet", 4.5, 13, 0, 1.4), ("table", 0, 12, 0, 1.5), ("sign", 0, 13.7, 0, 1.2), ("lamp", 5.3, 8, 0, 1)],
                      [{"kind": "lamp", "pos": [0, 3, 8], "color": [255, 215, 150], "intensity": 1.0}]),
}
for _k, _v in PRESETS.items():
    _v["theme"] = _k
FIXED_PLACES = {"hallway": "corridor", "night_city": "night_exterior", "break_room": "break_room"}   # purane rich fixed rooms (office jaan-boojh kar nahi)
PLACES = tuple(PRESETS) + tuple(FIXED_PLACES)
PLACE_HELP = ("village_market=busy bazaar with shops/fountain | island=sand, palm trees, boat | farm=fields and fences | city_street=buildings, bus stop, ATM | "
              "park=fountain, benches | school_yard=playground | campfire_night=night camp | mountain_town=snowy peaks | harbor=boats at sunset | "
              "classroom=whiteboard, desks | living_room=sofa, TV | kitchen=fridge, counter | shop_inside=wooden shop | hallway=long corridor | night_city=skyscraper at night | break_room=cafeteria")

# ======================================================================================================== TOPIC BANK (Rick-style explainers)
ROLES = {"harlan": "the rich owner / king / landlord who holds the money or the only resource", "milo": "the ordinary hard-working villager (audience stand-in)",
         "nina": "the clever curious one who asks the key 'but why?' question", "dex": "the impulsive young one who copies others and makes the classic mistake",
         "paula": "the rule-maker / banker / judge who sets the system", "gus": "the wise old storyteller who sums up the lesson",
         "ivy": "the sharp trader / merchant who spots opportunities", "bruno": "the strong worker / guard who does the real labour",
         "sultan": "the king / landlord who owns the land and makes the big decisions", "tara": "the little child who asks innocent but sharp questions",
         "sami": "the playful little boy who copies everyone", "rosa": "the village baker who sells the bread", "otto": "the farmer who grows the food",
         "kai": "the fisherman / sailor who knows the sea", "zara": "the scientist / teacher who explains facts", "ravi": "the friendly shopkeeper who sets the prices",
         "nora": "the kind grandmother who remembers old lessons", "felix": "the cunning banker / merchant who lends money"}

CONCEPTS = [  # (category, concept)
    ("money", "inflation: why the same coin buys less every year"), ("money", "compound interest: the tiny snowball that becomes an avalanche"),
    ("money", "opportunity cost: the price of the thing you did NOT choose"), ("money", "why the rich get richer: assets vs. paying for stuff"),
    ("money", "debt trap: borrowing to repay borrowing"), ("money", "bank run: why a healthy bank can collapse in a day"),
    ("money", "tulip mania and bubbles: when price has nothing to do with value"), ("money", "ponzi scheme: paying old investors with new investors' money"),
    ("money", "barter vs money: why coins were invented"), ("money", "scarcity and price: why water is cheap but diamonds are expensive"),
    ("money", "supply and demand: the village with one baker"), ("money", "sunk cost fallacy: throwing good money after bad"),
    ("money", "lifestyle inflation: earning more but saving the same"), ("money", "emergency fund: the rainy-day pot"), ("money", "time value of money: coin today vs coin tomorrow"),
    ("money", "monopoly: what happens when there is one seller"), ("money", "division of labour: why specializing makes everyone richer"),
    ("money", "comparative advantage: why even the best should trade"), ("money", "diversification: do not carry all eggs in one basket"),
    ("money", "taxes and public goods: who pays for the village well"), ("money", "automation and jobs: the village of infinite pots"),
    ("money", "cheap vs expensive: the true cost of buying cheap twice"), ("money", "side hustle vs. single income: one tree vs. an orchard"),
    ("society", "tragedy of the commons: the shared pasture that dies"), ("society", "network effects: why the last friend to join the app matters most"),
    ("society", "Goodhart's law: when a measure becomes a target it stops being good"), ("society", "cobra effect: a reward that made the problem bigger"),
    ("society", "free rider problem: the one who never chips in"), ("society", "broken windows and small rules that keep a town honest"),
    ("society", "information asymmetry: the seller knows what the buyer cannot see (lemons)"), ("society", "survivorship bias: the planes that never came back"),
    ("society", "prisoner's dilemma: why two sensible people choose badly"), ("society", "bystander effect: everyone watches, nobody helps"),
    ("psychology", "hedonic treadmill: why new things stop feeling new"), ("psychology", "loss aversion: losing hurts twice as much as winning feels good"),
    ("psychology", "anchoring: the first number you hear controls your decision"), ("psychology", "social proof: why we join the longest queue"),
    ("psychology", "Dunning-Kruger: why beginners feel like experts"), ("psychology", "confirmation bias: only finding what you already believe"),
    ("psychology", "procrastination: why the future self always gets the hard job"), ("psychology", "habit loop: cue, routine, reward"),
    ("psychology", "FOMO: the party you were never invited to"), ("psychology", "decision fatigue: why the 20th choice of the day is worse"),
    ("psychology", "imposter feelings: everyone is improvising"), ("psychology", "growth vs fixed mindset: 'not yet' instead of 'can't'"),
    ("psychology", "Parkinson's law: work expands to fill the time given"), ("psychology", "delayed gratification: one marshmallow now or two later"),
    ("psychology", "sunk time: why we stay in the wrong queue"), ("psychology", "paradox of choice: too many options make us unhappy"),
    ("life", "Pareto 80/20: a few things create most results"), ("life", "compounding habits: 1% better every day"), ("life", "Chesterton's fence: do not remove what you don't understand"),
    ("life", "the map is not the territory: plans vs reality"), ("life", "Murphy-proofing: why backups beat optimism"), ("life", "the two arrows: pain you cannot avoid vs the pain you add"),
    ("life", "specialist vs generalist: the fox and the hedgehog"), ("life", "the farmer and the horse: is it good luck or bad luck?"),
    ("history", "why the 24-hour clock / standard time was invented (railways)"), ("history", "how paper money began (Chinese merchants' receipts)"),
    ("history", "the real reason Q-tips carry a warning label"), ("history", "why zero took centuries to be accepted"),
]
FORMATS = ["PARABLE: a tiny allegory world where 2-4 characters live through events that show the idea (never name the idea until the last third)",
           "RICH vs POOR (or Two Paths): two characters make different choices with the same starting resources; time reveals the gap",
           "MYTH vs REALITY: a character confidently states a popular belief, events prove it wrong",
           "ORIGIN STORY: a real invention/rule/event retold with simple characters in a toy-sized setting",
           "WHAT IF: a village adopts a strange rule and we watch it play out day by day",
           "THE TRAP: a character walks step by step into the classic mistake while the storyteller warns us"]
TONES = ["calm storyteller with a dry twist", "slightly dark but clean irony", "wholesome and thought-provoking", "fast, punchy and witty", "mysterious, like an old fable"]
HOOKS = ["a shocking one-line statement about money or people", "a weird question to the viewer", "a scene already mid-disaster", "a bold 'nobody tells you this' line",
         "a tiny paradox ('the more X, the less Y')"]
PLACE_FOR = {"money": ["village_market", "city_street", "shop_inside", "harbor", "island"], "society": ["village_market", "farm", "island", "park", "mountain_town"],
             "psychology": ["park", "classroom", "living_room", "kitchen", "city_street"], "life": ["mountain_town", "farm", "campfire_night", "park", "harbor"],
             "history": ["harbor", "village_market", "classroom", "city_street", "shop_inside"]}

RULES = f"""You write 45-58 second vertical 'parable explainer' Shorts in the style of animated money/psychology/society explainers (like "The Difference Between Rich And Poor", "The Village of Infinite Pots").
NOT office comedy. Never set the story in an office. Each video teaches ONE real idea (given in TODAY'S INGREDIENTS) by acting it out in a small allegory world with 2-4 round-headed characters.
Tell the story only through character dialogue (no separate narrator): the wise-storyteller character may speak to the audience; everyone else speaks in-world. Lines must actually EXPLAIN what is happening, with concrete numbers/objects (coins, pots, fish, apples, days), not jokes about nothing.
Structure: 1) hook in 2 seconds, 2) setup of the tiny world and the starting situation, 3) 3-4 events that escalate the idea, 4) the reveal where the real concept is named in plain words, 5) one punchy last line the viewer remembers. Real-world facts must be accurate; if the idea is a real historical fact, state it correctly.
Output ONLY JSON.
CAST (exact names; roles are suggestions, you may reassign): {'; '.join(f"{k} = {v}" for k, v in ROLES.items())}
PLACES (pick one per scene in the field 'place', exact name): {', '.join(PLACES)}. Meaning: {PLACE_HELP}. Change place only when the story really moves; 1-3 different places per video.
CAMERA MOVES: {', '.join(MOVES)}. Vary moves; outdoor scenes prefer walk_through or orbit.
EXPRESSIONS: {', '.join(EXPRS)}. GESTURES: {', '.join(GESTURES)}. FACING: {', '.join(FACING)}. Hand props (use them to show the idea visually!): {', '.join(HAND_PROPS)}.
Slots: left, center, right, far. Never put two characters in the same slot (seat is not used).
Create 10-14 scenes of 2-5 seconds. Scene 1 hooks within 2 seconds; the last scene is a short punchline. Each scene has 1-3 characters; each speaks at most one line (max 12 words). At most two silent reaction scenes. Total dialogue 100-130 words.
Keep roles consistent for the whole video. Title 35-70 chars and curiosity-driven (e.g. 'The Village Where Everyone Got Poorer'). Description: one hook sentence + 3-5 hashtags like #money #psychology #storytime. Tags 6-10. 'fact' = the one-sentence real lesson. 'topic' must be specific so it never repeats. music_mood quirky or chill.
ACTING: change expressions between scenes (never the same expression twice in a row), use gestures (point, shrug, think, cheer, hold_up, jump) and hand props (coin, money_bag, bread, fish, apple, paper) so the idea is SHOWN, not only said. Use at least 4 different places OR 3 different camera moves.
HOOK: 'hook_text' = the on-screen hook shown in the first 2 seconds, 3-7 words, curiosity-driven, ALL real words (e.g. 'Why The Rich Get Richer', 'This Village Went Broke In 3 Days'). Scene 1's spoken line must match this hook and create a question the viewer wants answered.
TITLE: 35-70 chars, curiosity gap, no clickbait lies. DESCRIPTION: ONE strong hook sentence (max 140 chars) that makes people watch; do not add hashtags there (they are added automatically). 'fact' = the lesson in ONE punchy sentence (max 110 chars), shown on screen at the end.
Scene fields: place, camera_move, duration, characters. Character fields: name, slot, expr, optional gesture, facing, props, line.
"""

def schema(strict=True):
    S = lambda **k: {"type": "STRING", **{a:b for a,b in k.items() if strict or a != "enum"}}
    ch = {"type":"OBJECT","properties":{
        "name":S(enum=list(NAMES)),"slot":S(enum=list(SLOTS)),"expr":S(enum=list(EXPRS)),
        "gesture":S(enum=list(GESTURES)),"facing":S(enum=list(FACING)),
        "props":{"type":"ARRAY","items":S(enum=list(HAND_PROPS))},"line":S()
    },"required":["name","slot","expr","line"]}
    sc={"type":"OBJECT","properties":{"place":S(enum=list(PLACES)),"camera_move":S(enum=list(MOVES)),
        "duration":{"type":"NUMBER"},"characters":{"type":"ARRAY","items":ch}},
        "required":["place","camera_move","duration","characters"]}
    return {"type":"OBJECT","properties":{"topic":S(),"title":S(),"hook_text":S(),"description":S(),"fact":S(),"tags":{"type":"ARRAY","items":S()},
        "music_mood":S(enum=list(MOODS)),"scenes":{"type":"ARRAY","items":sc}},
        "required":["topic","title","hook_text","description","fact","tags","music_mood","scenes"]}



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
    """Gemini response ko safe renderer contract mein normalize karo (script.json par dobara chalane par bhi same nateeja)."""
    import copy
    scenes, prev_move = [], None
    for raw in (d.get("scenes") or [])[:18]:
        place = str(raw.get("place") or raw.get("location") or raw.get("room") or "park")
        raw_spec = raw.get("world_spec")
        world_spec, location = None, "custom"
        if isinstance(raw_spec, dict) and validate_world_spec(raw_spec) is not None:
            world_spec = raw_spec                                   # pehle se bana hua (preset ya valid custom)
        elif place in PRESETS:
            world_spec = copy.deepcopy(PRESETS[place])
        elif place in FIXED_PLACES:
            location = FIXED_PLACES[place]
        elif place in ROOMS:
            location = place
        elif isinstance(raw_spec, dict):
            location = nearest_fixed_room(raw_spec)
        else:
            world_spec = copy.deepcopy(PRESETS["park"])
        room = location if location in ROOMS else nearest_fixed_room(world_spec)
        move = _pick(raw.get("camera_move"), MOVES, "push_in")
        if move == prev_move:
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
            if slot == "seat" and (world_spec is not None or room not in SEAT_ROOMS):
                slot = "center"
            hand_props = [v for v in (c.get("props") or []) if v in HAND_PROPS][:3]
            chars.append({"name": nm, "slot": slot, "expr": _pick(c.get("expr"), EXPRS, "smile"),
                          "gesture": _pick(c.get("gesture"), GESTURES, "idle"), "facing": _pick(c.get("facing"), FACING, "front"),
                          "props": hand_props, "line": _line(c.get("line"))})
        for c in chars:
            if c["slot"] in used:
                c["slot"] = next((v for v in ("center", "left", "right", "far") if v not in used), "far")
            used.add(c["slot"])
        if chars:
            item = {"place": place if (place in PLACES or place in ROOMS) else "custom", "room": room, "location": location,
                    "camera_move": move, "duration": round(dur, 1), "characters": chars}
            if world_spec is not None:
                item["world_spec"] = world_spec
            scenes.append(item)
    talk = [sc for sc in scenes if any(c["line"] for c in sc["characters"])]
    if len(talk) < 5 or len({c["name"] for sc in scenes for c in sc["characters"]}) < 2:
        raise ValueError("story too short/one character only")
    words = sum(len(c["line"].split()) for sc in scenes for c in sc["characters"])
    total = sum(est_sec(sc) for sc in scenes)
    if total > MAX_SEC + 8 or words > MAX_WORDS + 25:
        raise ValueError(f"too long: ~{total:.0f}s, {words} words (max {MAX_SEC:.0f}s / {MAX_WORDS} words)")
    while total > MAX_SEC and len(scenes) > 6:
        k = max(range(1, len(scenes) - 1), key=lambda i: est_sec(scenes[i]) if not any(c["line"] for c in scenes[i]["characters"]) else est_sec(scenes[i]) - 2)
        total -= est_sec(scenes.pop(k))
    tags = [re.sub(r"[^\w]", "", str(tag))[:30] for tag in (d.get("tags") or [])][:10]
    title = re.sub(r"\s*#shorts\s*", " ", str(d.get("title") or "The Village That Learned The Hard Way"), flags=re.I).strip()[:92]
    fact = str(d.get("fact") or "").strip()[:140]
    hook_text = re.sub(r"[^\w\s'?!,.\-]", "", str(d.get("hook_text") or "")).strip()
    if len(hook_text.split()) > 8 or not hook_text:
        hook_text = " ".join(title.split()[:7])
    out = {"topic": str(d.get("topic") or "a small parable about money")[:140], "title": title, "hook_text": hook_text[:60],
           "description": build_description(d, title, fact), "fact": fact,
           "tags": [tag for tag in tags if tag] or ["money", "psychology", "storytime", "parable", "shorts"],
           "music_mood": _pick(d.get("music_mood"), MOODS, "chill"), "scenes": scenes}
    if not fact:
        out.pop("fact")
    return out


def build_description(d, title, fact):
    """Post ki description: hook line + lesson + CTA + hashtags (YouTube Shorts ke liye). Gemini ke hashtags/duplicate lines saaf."""
    raw = str(d.get("description") or "")
    first = re.sub(r"#\w+", "", raw).strip().split("\n")[0].strip()[:200] or title
    words = [re.sub(r"[^\w]", "", str(t_)).lower() for t_ in (d.get("tags") or [])][:5]
    tags = []
    for t_ in ["shorts", "storytime", "animation", "explained", "learnsomethingnew"] + words:
        if t_ and t_ not in tags and len(t_) < 25:
            tags.append(t_)
    parts = [first]
    if fact:
        parts.append(f"The lesson: {fact}")
    parts.append("Follow for a new animated story every day - every one teaches an idea that changes how you see money and people.")
    parts.append("Which character was YOU in this story? Tell us in the comments!")
    parts.append(" ".join("#" + t_ for t_ in tags[:8]))
    return "\n\n".join(parts)[:900]


def _S(place, move, dur, *cs):
    return {"place": place, "camera_move": move, "duration": dur, "characters": [dict(name=n, slot=sl, expr=e, line=l) for n, sl, e, l in cs]}


FALLBACK = {"topic": "inflation: why the same coin buys less every year", "title": "The Village Where Coins Got Lighter", "hook_text": "Why Everything Costs More",
            "description": "Why does everything cost more every year? A tiny village explains. #money #inflation #storytime #Shorts",
            "fact": "When more money is created than goods, each coin buys less: that is inflation.",
            "tags": ["money", "inflation", "economics", "storytime", "parable", "shorts"], "music_mood": "chill", "scenes": [
    _S("village_market", "push_in", 3, ("gus", "center", "smile", "One day, a village noticed bread getting expensive.")),
    _S("village_market", "orbit", 3, ("milo", "left", "sad", "Yesterday bread cost one coin. Today, two.")),
    _S("village_market", "pan", 3, ("ivy", "right", "smile", "Same bread. Same baker. Same flour. Why?")),
    _S("shop_inside", "push_in", 3, ("paula", "center", "grin", "The king printed a thousand new coins.")),
    _S("village_market", "walk_through", 3, ("harlan", "left", "grin", "Now everyone is richer! Look at all this money!"), ("nina", "right", "confused", "But the bakery still makes ten loaves.")),
    _S("village_market", "closeup_shake", 2.5, ("milo", "center", "shock", "More coins chasing the same bread.")),
    _S("village_market", "pull_out", 3, ("ivy", "left", "angry", "So bakers raise the price. Everyone does.")),
    _S("village_market", "low_angle", 3, ("nina", "center", "sad", "My savings are the same number but buy less.")),
    _S("park", "orbit", 3, ("gus", "center", "smile", "That is inflation. Coins did not shrink. Their power did.")),
    _S("park", "push_in", 3, ("gus", "center", "grin", "Money is only worth what it can buy."))]}


def pick_ingredients(used):
    """Har run naya combo: concept (aaj tak use na hua) + format + tone + hook + cast + places; 20% wildcard (Gemini khud naya idea chune)."""
    rnd = random.SystemRandom()
    forced = os.environ.get("FORCE_SEED", "").strip()
    pool = [c for c in CONCEPTS if c[1] not in used] or CONCEPTS
    wild = (not forced) and rnd.random() < 0.2
    cat, concept = (rnd.choice(list(PLACE_FOR)), "wildcard") if wild else (rnd.choice(pool) if not forced else ("money", forced))
    seed = "wildcard" if wild else concept
    idea = (f"Idea: pick a FRESH, specific, real idea from the category '{cat}' (economics, psychology, society, life or an accurate historical fact) that is NOT a cliche."
            if wild else f"Idea to teach (explain it accurately and concretely): {concept}.")
    cast = rnd.sample(NAMES, rnd.choice((3, 4)))
    places = ", ".join(rnd.sample(PLACE_FOR[cat], 3))
    return seed, (f"TODAY'S INGREDIENTS. {idea} Story format: {rnd.choice(FORMATS)}. Tone: {rnd.choice(TONES)}. Opening: {rnd.choice(HOOKS)}. "
                  f"Good places for this idea: {places}. Suggested cast: {', '.join(cast)} (use their roles; gus can be the storyteller).")


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
