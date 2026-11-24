# RESUME — PART A (toon/ rooms + camera + lighting + shadows + world)

## Mukammal (smoke test pass, 720x1280 par ~0.06 s/frame; 1800 frame ~2 min)
- toon/core.py (Cam, R), lighting.py (light_k, glow/shafts/sun patches, vignette), shadows.py (furniture+character shadow, AO),
  cameras.py (8 moves), rooms.py (office/corridor/break_room/night_exterior), world.py (render_scene), _stub_char.py (PART B na ho to)
- Resolution env se: TOON_W / TOON_H (default 720x1280).

## Contract notes
- render_scene(frame_idx, fps, scene, talk_lookup=None, t_offset=0.0): scene = ek scene dict, frame_idx scene-local.
  Poora {"scenes":[...]} dene par frame_idx global maana jata hai. talk_lookup(name, t) ko t = t_offset + scene-local seconds milta hai.
- Extra char keys jo world padhta hai: slot, expr, line, walk_to (slot name), props (["phone","key"]).
- Camera-moves target = characters ki average (x,z); "through_door" room["door"]["pos"] ki taraf jati hai.
- toon.characters import fail ho to _stub_char use hota hai (PART B ki file aate hi automatic switch).

## Known issues
- Dhoop ke hard patches kabhi kabhi characters/desk par bhi additive padte hain (roshni overlay jaisa) — intensity _sun_fx ke pcol se kam karo.
- Characters sirf 2D billboard hain, furniture ke peeche chalne par order sirf centre-depth se tay hota hai (lambi cheezon par galat ho sakta hai).
- Character shadow sirf farsh/deewar par; furniture par nahi girta. Seat slot par chair ka rukh fixed (camera ki taraf).
- Night room mein seat bench (4.6,5.9) hai, dims clamp ki wajah se bohot door ki shadows clip hoti hain.

## Agla qadam
1. PART B ki toon/characters.py (CHARS, draw_char) rakho; contact sheet dobara dekho (proportions/scale).
2. render_story.py mein render_scene wire karo; bgimg.py (Pollinations) aur depth.py hatao; 45 min budget check.
3. Zaroorat ho to rooms.py mein pcol/ambient tune karo.
