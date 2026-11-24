"""10 scenes ke render smoke-test aur contact sheet."""
import os
import sys
import cv2
import numpy as np

from toon.core import W, H
from toon.world import render_scene
from toon.rooms import ROOMS


def _scene(location, camera, name, expr, slot="center", world_spec=None):
    item = {
        "room": location if location in ROOMS else "office",
        "location": location,
        "camera_move": camera,
        "duration": 2.0,
        "characters": [{
            "name": name, "slot": slot, "expr": expr,
            "gesture": "idle", "facing": "front",
            "props": [], "line": "This is a render test."
        }]
    }
    if world_spec is not None:
        item["world_spec"] = world_spec
    return item


def main():
    scenes = [
        _scene("office", "push_in", "arjun", "smile", "seat"),
        _scene("corridor", "pan", "boss", "grin", "left"),
        _scene("break_room", "orbit", "zoya", "confused", "right"),
        _scene("night_exterior", "low_angle", "kabir", "shock"),
        _scene("office", "closeup_shake", "meera", "angry"),
        _scene("corridor", "pull_out", "uncle", "sad"),
        _scene("break_room", "tilt", "sara", "happy"),
        _scene("office", "through_door", "hamza", "tired"),
        _scene("custom", "walk_through", "zoya", "embarrassed", world_spec={
            "env":"outdoor","size":{"w":14,"d":18,"h":5},"time":"sunset","weather":"clear",
            "ground":{"type":"road","color":[75,78,85]},
            "walls":{"pattern":"plain","color":[190,190,190]},
            "sky":{"gradient":[[110,90,180],[240,170,110]]},
            "lights":[{"kind":"sun","pos":[-4,7,8],"color":[255,210,170],"intensity":1.2}],
            "props":[
                {"prop":"chai_stall","pos":[-3,7],"rot":0,"scale":1},
                {"prop":"street_lamp","pos":[4,10],"rot":0,"scale":1}
            ],
            "custom_props":[],"slots":["left","center","right","far"]
        }),
        _scene("custom", "push_in", "arjun", "grin", world_spec={
            "env":"indoor","size":{"w":12,"d":14,"h":4},"time":"day","weather":"clear",
            "ground":{"type":"tile","color":[185,185,180]},
            "walls":{"pattern":"brick","color":[215,220,225]},
            "sky":{"gradient":[[95,155,220],[220,230,245]]},
            "lights":[{"kind":"panel","pos":[0,3.5,5]}],
            "props":[{"prop":"desk","pos":[-3,5],"rot":0,"scale":1},
                     {"prop":"plant","pos":[3,8],"rot":0,"scale":1}],
            "custom_props":[],"slots":["left","center","right","far","seat"]
        })
    ]

    tiles = []
    for index, scene in enumerate(scenes):
        print(f"[dev_test] rendering {index+1}/{len(scenes)}: {scene['location']} / {scene['camera_move']}", flush=True)
        frame = render_scene(12, 30, scene)
        if not isinstance(frame, np.ndarray) or frame.shape != (H, W, 3) or frame.dtype != np.uint8:
            raise RuntimeError(f"Invalid frame {index}: shape/dtype={getattr(frame,'shape',None)}/{getattr(frame,'dtype',None)}")
        thumb = cv2.resize(frame, (360, 640), interpolation=cv2.INTER_AREA)
        cv2.rectangle(thumb, (0, 0), (359, 42), (15, 15, 20), -1)
        label = f"{index+1:02d} {scene['location']} | {scene['camera_move']}"
        cv2.putText(thumb, label[:42], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .48, (255,255,255), 1, cv2.LINE_AA)
        tiles.append(thumb)

    rows = []
    for i in range(0, len(tiles), 5):
        row = np.hstack(tiles[i:i+5])
        if row.shape[1] < 1800:
            row = cv2.copyMakeBorder(row,0,0,0,1800-row.shape[1],cv2.BORDER_CONSTANT,value=(30,30,30))
        rows.append(row)
    sheet = np.vstack(rows)
    if not cv2.imwrite("contact_sheet.png", sheet):
        raise RuntimeError("contact_sheet.png write failed")
    try:
        import subprocess
        subprocess.run([sys.executable, "preview.py", "village_market", "farm", "island", "city_street", "park", "harbor",
                        "school_yard", "campfire_night", "mountain_town"], check=True)
    except Exception as e:
        print("[dev_test] preview skip:", e)
    print(f"[dev_test] PASS: {len(tiles)} frames, contact_sheet.png {sheet.shape[1]}x{sheet.shape[0]}", flush=True)


if __name__ == "__main__":
    main()
