"""toon/characters.py - 18 characters ka cast (v2: detailed rig ke liye). Tamam rang BGR mein.
Naya character = ek _c(...) line. Fields: shirt, pants, hair, style (HAIRSTYLES), hat (HATS), outfit (OUTFITS), skin, eye, tie, glasses, mustache, beard, lashes, freckles,
size (bachon ke liye <1), head_w/head_h. 'voice' = Kokoro voice, 'edge' = edge-tts backup voice. 'bio' = Gemini ko role-hint."""

CREAM, TAN, BROWN, DEEP = (232, 242, 250), (178, 212, 240), (125, 168, 205), (92, 128, 168)
SHOE = (22, 20, 28)


def _c(**data):
    base = dict(skin=CREAM, shoe=SHOE, pants=(70, 60, 58), eye=(80, 55, 40), head_w=.255, head_h=.275, body_w=.17, body_h=.29, limb_w=.034, size=1.0)
    base.update(data)
    return base


CHARS = {
    # ---------------- purana cast (ab zyada detail ke saath)
    "harlan": _c(name="Harlan", shirt=(105, 62, 30), pants=(70, 45, 28), hair=(150, 148, 150), style="side_part", outfit="suit", tie=True, tie_col=(40, 40, 215), seed=11,
                 voice="am_michael", edge="en-US-AndrewNeural", eye=(120, 90, 50), bio="Smiling grey-haired big boss in a navy suit and red tie."),
    "milo": _c(name="Milo", shirt=(225, 160, 80), pants=(85, 70, 60), hair=(40, 34, 40), style="side_part", outfit="shirt", seed=12, freckles=True,
               voice="am_adam", edge="en-US-GuyNeural", bio="Average hard-working villager, blue shirt."),
    "nina": _c(name="Nina", shirt=(110, 190, 70), pants=(70, 60, 58), hair=(28, 26, 34), style="bob" if False else "bowl", outfit="dress", glasses=True, lashes=True, seed=13,
               voice="af_heart", edge="en-US-JennyNeural", bio="Sharp curious one with a black bob and glasses."),
    "dex": _c(name="Dex", shirt=(60, 150, 245), pants=(110, 85, 70), hair=(30, 40, 90), style="spiky", outfit="hoodie", seed=14, skin=TAN,
              voice="am_eric", edge="en-US-BrianNeural", bio="Impulsive young one in an orange hoodie."),
    "paula": _c(name="Paula", shirt=(160, 95, 45), pants=(60, 45, 40), hair=(60, 70, 120), style="bun", outfit="suit", tie=False, lashes=True, seed=15, skin=TAN,
                voice="af_bella", edge="en-US-AriaNeural", bio="Calm, strict rule-maker / banker in a blue blazer."),
    "gus": _c(name="Gus", shirt=(120, 135, 150), pants=(90, 95, 105), hair=(160, 160, 165), style="bald_sides", outfit="vest", mustache=True, seed=16, glasses=True,
              voice="am_onyx", edge="en-US-ChristopherNeural", bio="Wise old storyteller with a big grey moustache."),
    "ivy": _c(name="Ivy", shirt=(70, 200, 245), pants=(95, 70, 60), hair=(70, 150, 215), style="long", outfit="blouse", lashes=True, seed=17, eye=(60, 120, 60),
              voice="af_nicole", edge="en-US-AriaNeural", bio="Sharp cheerful trader with long golden hair."),
    "bruno": _c(name="Bruno", shirt=(70, 70, 205), pants=(60, 55, 70), hair=(35, 30, 40), style="curly", outfit="uniform", seed=18, skin=BROWN, body_w=.20, head_w=.27,
                voice="am_michael", edge="en-US-ChristopherNeural", bio="Strong quiet worker / guard with curly hair."),
    # ---------------- naya cast
    "sultan": _c(name="Sultan", shirt=(60, 40, 170), pants=(50, 30, 120), hair=(190, 190, 195), style="crop", hat="crown", outfit="robe", beard=True, seed=19, size=1.05, body_w=.2,
                 voice="bm_george", edge="en-GB-RyanNeural", bio="The king / landlord / owner of the land, grey beard and golden crown."),
    "tara": _c(name="Tara", shirt=(70, 215, 250), pants=(160, 100, 70), hair=(40, 50, 100), style="ponytail", outfit="tee", lashes=True, seed=20, size=.74, head_w=.30, head_h=.32,
               body_w=.15, body_h=.25, skin=TAN, voice="af_sky", edge="en-US-AnaNeural", bio="Curious little girl who asks innocent but sharp questions."),
    "sami": _c(name="Sami", shirt=(95, 175, 90), pants=(120, 90, 70), hair=(30, 28, 38), style="messy", outfit="hoodie", seed=21, size=.76, head_w=.30, head_h=.31, body_w=.15, body_h=.25,
               skin=BROWN, voice="am_puck", edge="en-US-AnaNeural", bio="Playful little boy who copies others."),
    "rosa": _c(name="Rosa", shirt=(215, 190, 235), pants=(80, 70, 85), hair=(40, 60, 110), style="messy", hat="chef", outfit="apron", lashes=True, seed=22, skin=TAN,
               voice="af_jessica", edge="en-US-MichelleNeural", bio="Warm village baker with a chef hat and apron."),
    "otto": _c(name="Otto", shirt=(70, 100, 190), pants=(190, 120, 60), hair=(110, 150, 200), style="crop", hat="straw", outfit="overalls", mustache=True, seed=23, freckles=True,
               voice="am_liam", edge="en-US-RogerNeural", bio="Hard-working farmer with a straw hat and overalls."),
    "kai": _c(name="Kai", shirt=(120, 110, 60), pants=(70, 80, 95), hair=(60, 60, 70), style="messy", hat="beanie", outfit="jacket", beard=True, seed=24, skin=BROWN,
              voice="am_fenrir", edge="en-US-SteffanNeural", bio="Fisherman / sailor with a beanie and short beard."),
    "zara": _c(name="Zara", shirt=(245, 245, 240), pants=(75, 60, 70), hair=(30, 25, 40), style="bun", outfit="lab_coat", glasses=True, lashes=True, seed=25, skin=BROWN,
               voice="af_alloy", edge="en-GB-SoniaNeural", bio="Brilliant scientist / teacher in a lab coat and glasses."),
    "ravi": _c(name="Ravi", shirt=(235, 245, 250), pants=(215, 225, 230), hair=(30, 30, 38), style="topi", outfit="kurta", mustache=True, seed=26, skin=BROWN,
               voice="am_echo", edge="en-IN-PrabhatNeural", bio="Friendly shopkeeper in a white kurta and cap."),
    "nora": _c(name="Nora", shirt=(150, 120, 210), pants=(100, 70, 130), hair=(205, 205, 210), style="bun", outfit="dress", glasses=True, lashes=True, seed=27, size=.96, body_w=.19,
               voice="bf_emma", edge="en-GB-LibbyNeural", bio="Kind grandmother with grey bun, who remembers old lessons."),
    "felix": _c(name="Felix", shirt=(80, 80, 85), pants=(55, 55, 60), hair=(60, 55, 60), style="bald", outfit="suit", tie=True, tie_col=(50, 190, 235), glasses=True, seed=28,
                body_w=.2, voice="bm_lewis", edge="en-GB-ThomasNeural", bio="Cunning banker / merchant in a grey suit and gold tie."),
}

# purane naam -> naye character (purani scripts/fallback ke liye)
ALIASES = {"arjun": "milo", "boss": "harlan", "zoya": "nina", "kabir": "dex", "meera": "paula", "uncle": "gus", "sara": "ivy", "hamza": "bruno",
           "king": "sultan", "baker": "rosa", "farmer": "otto", "banker": "felix", "kid": "tara"}


def get_char(name):
    """Unknown/purane naam par safe fallback."""
    n = str(name).lower()
    return CHARS.get(n) or CHARS.get(ALIASES.get(n, "")) or CHARS["milo"]


def draw_char(r, P_world, ch, st):
    """Character drawing ka public contract."""
    from toon.rig import draw_char as renderer
    return renderer(r, P_world, ch, st)
