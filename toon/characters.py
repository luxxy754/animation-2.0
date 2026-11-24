
"""Original fixed cast. Tamam rang BGR mein hain."""

def _c(**data):
    return data


CHARS = {
    "arjun": _c(name="Arjun", skin=(140,190,238), shirt=(150,170,40), hair=(38,58,100), style="messy", outfit="hoodie", seed=1, voice="am_adam", edge="en-US-GuyNeural", bio="Overthinking, sarcastic young employee."),
    "boss": _c(name="Boss", skin=(170,215,250), shirt=(55,65,115), hair=(178,172,168), style="side_part", outfit="suit", tie=True, tie_col=(40,40,210), seed=2, voice="am_michael", edge="en-US-AndrewNeural", bio="Grey-haired strict boss with a wide smile."),
    "zoya": _c(name="Zoya", skin=(150,200,242), shirt=(235,170,90), hair=(30,28,34), style="bob", outfit="blouse", glasses=True, seed=3, voice="af_heart", edge="en-US-JennyNeural", bio="Sharp, witty coworker with a black bob."),
    "kabir": _c(name="Kabir", skin=(105,150,205), shirt=(60,190,235), hair=(25,22,28), style="spiky", outfit="hoodie", seed=4, voice="am_eric", edge="en-US-BrianNeural", bio="Overenthusiastic young intern."),
    "meera": _c(name="Meera", skin=(165,210,248), shirt=(150,190,70), hair=(40,30,70), style="bun", outfit="dress", seed=5, voice="af_bella", edge="en-US-AriaNeural", bio="Calm and intimidating HR manager."),
    "uncle": _c(name="Uncle", skin=(120,170,220), shirt=(110,140,170), hair=(60,60,66), style="bald_sides", outfit="kurta", mustache=True, glasses=True, seed=6, voice="am_michael", edge="en-US-ChristopherNeural", bio="Wise older employee with a moustache."),
    "sara": _c(name="Sara", skin=(165,205,240), shirt=(170,100,190), hair=(35,30,65), style="long", outfit="lab_coat", seed=7, voice="af_nicole", edge="en-US-AriaNeural", bio="Curious researcher who loves experiments."),
    "hamza": _c(name="Hamza", skin=(115,165,220), shirt=(55,145,185), hair=(25,35,55), style="curly", outfit="uniform", seed=8, voice="am_onyx", edge="en-US-ChristopherNeural", bio="Quiet, clever problem solver."),
}


def get_char(name):
    """Unknown naam par safe fallback."""
    return CHARS.get(str(name).lower(), CHARS["arjun"])


def draw_char(r, P_world, ch, st):
    """Character drawing ka public contract."""
    from toon.rig import draw_char as renderer
    return renderer(r, P_world, ch, st)
