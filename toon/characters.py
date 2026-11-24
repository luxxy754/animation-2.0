"""toon/characters.py - NAYA cast (videoplayback.mp4 ke style ke characters, naye naam). Tamam rang BGR mein hain.
Style: bada gol anda-sar, cream-safed chehra, mota kaala outline, patla jism, mote stick haath-pair.
Naya character = ek _c(...) line. 'voice' = Kokoro voice, 'edge' = edge-tts backup voice."""

FACE = (232, 242, 250)          # cream-safed chehra (sab characters ka)
SHOE = (20, 18, 24)


def _c(**data):
    base = dict(skin=FACE, shoe=SHOE, head_w=.255, head_h=.275, body_w=.15, body_h=.27, limb_w=.032)
    base.update(data)
    return base


CHARS = {
    # --- bada boss: grey baal, navy suit, laal tie, chauri muskurahat
    "harlan": _c(name="Harlan", shirt=(95, 60, 30), hair=(150, 148, 150), style="side_part", outfit="suit", tie=True, tie_col=(40, 40, 215), seed=11,
                 voice="am_michael", edge="en-US-AndrewNeural", bio="Smiling grey-haired big boss in a navy suit and red tie."),
    # --- aam office worker: neeli shirt, hari tie
    "milo": _c(name="Milo", shirt=(225, 160, 80), hair=(40, 34, 40), style="side_part", outfit="suit", tie=True, tie_col=(70, 170, 60), seed=12,
               voice="am_adam", edge="en-US-GuyNeural", bio="Average tired office worker, blue shirt and green tie."),
    # --- tez-tarrar coworker: kaala bob, chashma, hara dress
    "nina": _c(name="Nina", shirt=(110, 190, 70), hair=(28, 26, 34), style="bob", outfit="dress", glasses=True, seed=13,
               voice="af_heart", edge="en-US-JennyNeural", bio="Sharp, sarcastic coworker with a black bob and glasses."),
    # --- junior intern: narangi hoodie, spiky baal
    "dex": _c(name="Dex", shirt=(60, 150, 245), hair=(30, 40, 90), style="spiky", outfit="hoodie", seed=14,
              voice="am_eric", edge="en-US-BrianNeural", bio="Overexcited intern in an orange hoodie."),
    # --- HR: neela blazer, joora
    "paula": _c(name="Paula", shirt=(150, 90, 40), hair=(60, 70, 120), style="bun", outfit="suit", tie=False, seed=15,
                voice="af_bella", edge="en-US-AriaNeural", bio="Calm and scary HR manager in a blue blazer."),
    # --- purana janitor / chacha: ganje sar, moonchh
    "gus": _c(name="Gus", shirt=(120, 135, 150), hair=(150, 150, 155), style="bald_sides", outfit="uniform", mustache=True, seed=16,
              voice="am_onyx", edge="en-US-ChristopherNeural", bio="Old janitor who knows every office secret."),
    # --- lambi baalon wali accountant
    "ivy": _c(name="Ivy", shirt=(70, 200, 245), hair=(70, 150, 215), style="long", outfit="blouse", seed=17,
              voice="af_nicole", edge="en-US-AriaNeural", bio="Cheerful accountant with long golden hair."),
    # --- bhaari-bharkam security guard
    "bruno": _c(name="Bruno", shirt=(60, 60, 200), hair=(35, 30, 40), style="curly", outfit="uniform", seed=18,
                voice="am_michael", edge="en-US-ChristopherNeural", bio="Big, quiet security guard with curly hair."),
}

# purane naam -> naye character (purani scripts/fallback ke liye)
ALIASES = {"arjun": "milo", "boss": "harlan", "zoya": "nina", "kabir": "dex", "meera": "paula", "uncle": "gus", "sara": "ivy", "hamza": "bruno"}


def get_char(name):
    """Unknown/purane naam par safe fallback."""
    n = str(name).lower()
    return CHARS.get(n) or CHARS.get(ALIASES.get(n, "")) or CHARS["milo"]


def draw_char(r, P_world, ch, st):
    """Character drawing ka public contract."""
    from toon.rig import draw_char as renderer
    return renderer(r, P_world, ch, st)
