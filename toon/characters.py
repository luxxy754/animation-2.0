"""Original characters (PART B). Har video mein ek jaise: palette, hairstyle, kapde, chashma, ID card fixed hain.
Sab BGR rang. 'voice'/'edge' = us character ki awaaz (lipsync.py use karta hai). 'bio' sirf Gemini prompt ke liye."""

def _c(**k):
    return k


CHARS = {
    "arjun": _c(name="Arjun", skin=(140, 190, 238), shirt=(150, 170, 40), hair=(38, 58, 100), fy=.42, wave=1.6, style="messy", outfit="tee",
                id=(190, 120, 40), shoe=(40, 40, 60), seed=1, voice="am_adam", edge="en-US-GuyNeural",
                bio="Arjun, 20s, the relatable overthinking employee. Anxious, sarcastic inner voice, always on his phone or coffee."),
    "boss": _c(name="Boss", skin=(170, 215, 250), shirt=(72, 36, 26), hair=(178, 172, 168), fy=.55, style="comb", outfit="suit", tie=True,
               tie_col=(40, 40, 210), seed=2, voice="am_michael", edge="en-US-AndrewNeural",
               bio="Mr. Mehra, the boss, 50s, grey hair, navy suit, red tie. Smiles too wide with teeth when he has bad news. Loves buzzwords."),
    "zoya": _c(name="Zoya", skin=(150, 200, 242), shirt=(235, 170, 90), hair=(30, 28, 34), fy=.22, wave=0, style="bob", outfit="blouse", glasses=True,
               id=(60, 60, 200), shoe=(60, 40, 90), seed=3, voice="af_heart", edge="en-US-JennyNeural",
               bio="Zoya, the sharp coworker with glasses and a black bob. Deadpan, always knows the office gossip first."),
    "kabir": _c(name="Kabir", skin=(105, 150, 205), shirt=(60, 190, 235), hair=(25, 22, 28), fy=.35, style="spiky", outfit="hoodie",
                id=(160, 90, 150), shoe=(240, 240, 240), seed=4, voice="am_eric", edge="en-US-BrianNeural",
                bio="Kabir, the overenthusiastic intern in a mustard hoodie with spiky hair. Says yes to everything, understands nothing."),
    "meera": _c(name="Meera", skin=(165, 210, 248), shirt=(150, 190, 70), hair=(40, 30, 70), fy=.3, wave=.6, style="bun", outfit="blouse",
                id=(200, 130, 50), shoe=(50, 50, 120), seed=5, voice="af_bella", edge="en-US-AriaNeural",
                bio="Meera from HR, hair in a bun, teal blouse. Very polite, scary calm, sends 'quick sync?' invites."),
    "uncle": _c(name="Uncle", skin=(120, 170, 220), shirt=(110, 140, 170), hair=(60, 60, 66), fy=.0, style="bald", outfit="polo", mustache=True, glasses=True,
                id=(90, 160, 90), shoe=(30, 40, 60), seed=6, voice="am_michael", edge="en-US-ChristopherNeural",
                bio="Uncle, the bald moustached veteran who has been in the office since forever. Knows every shortcut and every rumor, calm and wise."),
}


def get_char(name):
    """Galat naam par pehla character (crash nahi)."""
    return CHARS.get(str(name).lower(), CHARS["arjun"])


def draw_char(r, Pw, ch, st):
    """Contract: draw_char(r, P_world, ch, st). Lazy import taake story_gen bina cv2 ke CHARS padh sake."""
    from toon.rig import draw_char as _d
    return _d(r, Pw, ch, st)
