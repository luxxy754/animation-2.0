# quiz-shorts-bot v5 ("parable explainer" Shorts)

Har run: Gemini ek **bilkul nayi parable-story** likhta hai ("The Difference Between Rich And Poor" jaisi: chhoti si allegory-duniya
jahan 2-4 gol-sar characters kisi money/society/psychology ke idea ko events se samjhate hain) -> narrator voice (Kokoro, free) ->
har location ka khaali **3D-diorama background** (Pollinations, free) -> **animated characters** code se (`rick.py`: chalna, ishare,
expressions, lipsync) -> camera moves + word-by-word captions -> YouTube Short.

Characters code se draw hote hain, isliye har scene mein bilkul **same** dikhte hain (AI images mein ye nahi hota).

Files: `story_gen.py` (story + cast + locations) | `voice.py` (awaaz + word timing + mouth data) | `bgimg.py` (backgrounds + fallback) |
`rick.py` (character engine) | `render_story.py` (actors, camera, captions, mix) | `music.py` (music + sfx) | `main.py` (upload).

Offline test (bina internet / bina API key):
`ALLOW_FALLBACK=1 python story_gen.py && BG_OFFLINE=1 VOICE_ENGINE=fake python render_story.py`

Tweak:
- Topics: `story_gen.py` -> `CONCEPTS`, `WORLDS`, `FORMATS`, `RULES`
- Awaaz: workflow `KOKORO_VOICE`, `VOICE_SPEED`
- Characters: `rick.py` (look) | `story_gen.py` -> `HATS`, `SHIRT_C` (options)
- Shot size / positions: `render_story.py` -> `SHOT_H`, `SHOT_TOP`, `SLOTS`
- Background look: `bgimg.py` -> `STYLE`
- Captions: `render_story.py` -> `word_sprite()` (size, colour, outline), `fonts/Poppins-Bold.ttf`

## v6: 2.5D parallax + flat-vector props
- `depth.py`: har location ka depth map (Depth Anything V2 Small, CPU; fail/`DEPTH_OFFLINE=1` par synthetic ground-plane depth). Camera dolly/orbit mein pixels depth ke hisaab se hilte hain; characters/props bhi usi parallax mein.
- `props.py`: murgi, barrel, chalkboard, sikke, ande, ghar, ped, sign, bori, graph, hath. Naya prop = ek function + `PROP_TYPES` mein naam.
- Offline test: `ALLOW_FALLBACK=1 python story_gen.py && BG_OFFLINE=1 VOICE_ENGINE=fake DEPTH_OFFLINE=1 python render_story.py`

## v7: detailed characters + rich backgrounds + better hook/captions
- `toon/rig.py` (v2): shaded rounded body, sleeves/pants, hands with thumbs, iris+highlight eyes, brows, nose, blush, teeth/tongue, hats (crown, straw, chef, beanie...), beards, head tilt, breathing, arm-swing walk, jump. Sirf character ROI par supersample (tez).
- `toon/characters.py`: 18 characters (8 purane + sultan, tara, sami, rosa, otto, kai, zara, ravi, nora, felix). Naya = ek `_c(...)` line.
- `toon/scenery.py`: har outdoor preset ka themed backdrop (sooraj, badal, parinde, pahariyan, skyline/barn/windmill/samundar), ground detail (ghaas, phool, raaste), set pieces (stalls, ghar, ped...), background NPC log, jaanwar (murghi, seagull, kekra, billi, titli), particles, vignette; indoor decor (khidki, frames, ghari, qaleen).
- `render_story.py`: highlighted word-by-word captions (bolta lafz peela), hook banner (pehle 2.4s), progress bar, aakhri "THE LESSON" card.
- `story_gen.py`: `hook_text`, naye gestures (think, hold_up, jump...) aur hand props (coin, money_bag, bread, fish, apple); description template: hook + lesson + CTA + hashtags.
- `python preview.py [place ...]` -> `preview.png` (backgrounds ka quick contact sheet).
