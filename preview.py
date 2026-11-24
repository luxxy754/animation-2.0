"""Quick look: PRESETS ke scenes ka contact sheet (python preview.py [place ...])."""
import sys, copy, cv2, numpy as np
import story_gen
from toon.world import render_scene

places = sys.argv[1:] or ["village_market", "farm", "island", "city_street", "park", "harbor"]
cast = list(story_gen.CHARS)
tiles = []
for i, p in enumerate(places):
    two = [{"name": cast[i % len(cast)], "slot": "left", "expr": "smile", "gesture": "wave", "line": "hi"},
           {"name": cast[(i + 3) % len(cast)], "slot": "right", "expr": "shock", "line": ""}]
    sc = {"place": p, "camera_move": ["push_in", "orbit", "walk_through"][i % 3], "duration": 3.0, "characters": two}
    sc = story_gen.clean({"scenes": [sc] * 6, "topic": "t"}) if False else sc
    if p in story_gen.PRESETS:
        sc["world_spec"] = copy.deepcopy(story_gen.PRESETS[p]); sc["location"] = "custom"
    else:
        sc["location"] = story_gen.FIXED_PLACES.get(p, p)
    fr = render_scene(40, 30, sc)
    t = cv2.resize(fr, (360, 640), interpolation=cv2.INTER_AREA)
    cv2.putText(t, p, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2, cv2.LINE_AA)
    tiles.append(t)
rows = [np.hstack(tiles[i:i + 3] + [np.zeros_like(tiles[0])] * (3 - len(tiles[i:i + 3]))) for i in range(0, len(tiles), 3)]
cv2.imwrite("preview.png", np.vstack(rows)); print("preview.png", len(tiles))
