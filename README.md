# ChamberResonator

A desktop prototype for **rendering sound as it travels through drawn rooms**. Draw rooms of different shapes and sizes, set their temperature, humidity, materials and ceiling, optionally import a measured impulse response (IR) per room, draw a path through them, and render an audio file as if the source were walking that path and hearing itself in each room.

> **Status: working prototype.** The DSP core and the GUI have been smoke-tested (drawing, path editing, keyframes, mono and 16-channel rendering). Live audio preview has not been tested. Treat the acoustics as a *creative approximation*, not a prediction.

## Contents

- [Features](#features)
- [Install and run](#install-and-run)
- [Quick start](#quick-start)
- [Concepts](#concepts)
- [How the audio is rendered](#how-the-audio-is-rendered)
- [16-channel 8.1.7 export](#16-channel-817-export)
- [Reference: materials and passages](#reference-materials-and-passages)
- [Reference sheet: recording impulse responses](#reference-sheet-recording-impulse-responses)
- [Limitations](#limitations)
- [Ideas / roadmap](#ideas--roadmap)
- [Credits and licence](#credits-and-licence)

## Features

- **Room drawing:** rectangle, tapered (trapezoid) and circular floor plans; flat, angled or domed ceilings; width, length, height, temperature, humidity and material per room.
- **Per-room reverb engine:** statistical (octave-band decay), 16-line **FDN**, or **Bypass**. An imported IR overrides the engine.
- **Imported IRs:** one mono IR per room, adjusted for the room's temperature, humidity and decay settings.
- **Walker path:** place points inside rooms; the first point in each new room is an *entry point* with its own passage type (open, door, window, wall, thick wall).
- **Timed rendering:** keyframes on the waveform tie path points to times in the source audio; the room changes as the source crosses between rooms, with crossfades.
- **Early reflections from the room's real outline:** ray-cast from the walker's position and updated as it moves.
- **Cascade rendering:** an alternative model where the whole sound passes through every room in series.
- **Output:** mono WAV or **16-channel (8.1.7) float WAV**; optional playhead preview with `sounddevice`.

## Install and run

Requirements: Python 3.9+, `numpy`, `scipy`, and `tkinter` (included with most Python installs; on Debian/Ubuntu: `sudo apt install python3-tk`).

```bash
pip install numpy scipy
# optional, for the Preview button:
pip install sounddevice

python chamber_resonator.py
```

The whole program is the single file `chamber_resonator.py`. The DSP functions can also be imported without the GUI.

## Quick start

1. Choose **Draw room**, pick a shape in the dropdown next to it, and drag on the map (1 grid square = 1 m, snapped to 0.5 m, minimum 1 × 1 m). Edit its properties in the **Room** tab and press Enter or Apply.
2. Choose **Add path points** and click inside rooms in travel order. The first point you place in a new room becomes an **entry point** (drawn as a square); set its passage type in the **Path / keyframes** tab.
3. Click **Load input WAV…**. Two keyframes are created automatically (first point at 0 s, last point at the end).
4. Click the waveform to add a keyframe for the selected point. This is when the walker reaches that point. Points you skip are timed by path distance.
5. Pick **Timed (moving source)** and an output (**mono** or **16ch 8.1.7**), then **Render…** to export, or **Preview** to hear it with a moving marker.

The faint green lines on the map show the rays used for the early reflections, drawn from the selected point or, during preview, from the playhead.

### Controls at a glance

| Control | What it does |
| --- | --- |
| Draw room / Select / Add path points | Map interaction mode |
| Room tab | Name, shape, size, ceiling, temperature, humidity, material, decay multiplier (0.5–1.5), engine, material filter, early reflections, IR load/clear |
| Path / keyframes tab | Passage type of the selected entry point; delete/undo/clear path; keyframe list with time editing |
| Waveform | Click to add a keyframe for the selected point |
| Path model | **Timed** (walker) or **Cascade** (series) |
| Output | `mono` (16-bit WAV) or `16ch 8.1.7` (float32 WAV) |

## Concepts

- **Walker model (Timed):** the source and the listener are the same. As the source walks, the room it is in processes the audio, as when you hear your own footsteps change from room to room. The room just left keeps ringing, heard through the passage you went through.
- **Traversal vs. entry points:** ordinary points are traversal points with no passage setting. The first point in a new room is always an entry point, and only entry points carry a passage type.
- **Cascade model:** the full signal passes through each room in sequence, with delay, air absorption, level and passage filtering between rooms. Use it for "a sound travelling to a distant listener".

## How the audio is rendered

### Air and room acoustics

- **Speed of sound:** `c = 331.3 + 0.606 · T` m/s, so temperature changes delays and IR timing.
- **Air absorption:** ISO 9613-1, frequency-dependent, from temperature and humidity.
- **Decay:** Sabine reverb time per octave band (125 Hz–4 kHz) from the room's volume, surface area, material and air absorption. Shape and ceiling change volume and surface area (circle, trapezoid, angled or domed ceilings are accounted for).

### Reverb engines

| Engine | Description |
| --- | --- |
| Statistical | Noise with a separate exponential decay per octave band. |
| FDN | 16 delay lines, Hadamard feedback matrix, one-pole damping per line; delay lengths scale with room size and speed of sound. In 16-channel mode each line feeds its own channel. |
| Bypass | No reverb. |
| Imported IR | Used instead of the engine. Treated as recorded at **20 °C / 50 % RH**; the program applies the *difference* to the room's settings (time-axis scaling for speed of sound, air-absorption difference over a 10 m reference path, optional decay multiplier). Material is not applied unless you enable the optional material filter. |

### Early reflections (Timed model)

- 360 rays are cast from the walker's position against the room's outline (rectangle, trapezoid or circle). Rays near normal incidence return echoes, delayed by the round trip and arriving from the wall's direction.
- A ceiling echo uses the **local ceiling height** (so it changes under angled or domed ceilings) and a floor echo is added. Source/ear height is fixed at 1.6 m.
- Reflections are coloured by the room's material and loudness-capped near walls. They are computed in short overlapping blocks, so they change smoothly as the walker moves.
- Skipped for rooms with an imported IR (the measured IR contains its own), for the Bypass engine, when the room's checkbox is off, and in Cascade mode.

### Passages and crossfades

- Audio is gated into room branches with 30 ms crossfades. The room changes halfway in time between two keyframed points that lie in different rooms.
- When leaving a room, its tail (and reflections) are heard through the entry point's passage type, applied as a short crossfade into the filtered signal.

## 16-channel 8.1.7 export

Channel order (confirm against your own speaker routing; changing it is a small edit in `pan_gains`):

| Channels | Content |
| --- | --- |
| 1–8 | Ear-level ring, 45° steps. Channel 1 = front (canvas up), clockwise |
| 9 | LFE: low-passed (120 Hz) sum of the ear-level channels |
| 10–15 | Height ring at 30°, 90°, 150°, 210°, 270°, 330° |
| 16 | Zenith |

- The walker's own direct sound is non-directional (spread evenly around the ear-level ring).
- Early reflections arrive from the actual wall directions; the ceiling echo goes to the zenith and height speakers, the floor echo to the ear-level ring.
- Reverb tails are decorrelated across channels (FDN: one line per channel).
- Arrival directions are fixed to the plan, not to the walking direction.

## Reference: materials and passages

### Absorption coefficients (statistical and FDN engines)

Illustrative values per octave band. The experimental entries are invented for creative use.

| Material | 125 | 250 | 500 | 1k | 2k | 4k |
| --- | --- | --- | --- | --- | --- | --- |
| concrete | 0.01 | 0.01 | 0.02 | 0.02 | 0.02 | 0.03 |
| plaster | 0.01 | 0.02 | 0.02 | 0.03 | 0.04 | 0.05 |
| brick | 0.03 | 0.03 | 0.03 | 0.04 | 0.05 | 0.07 |
| glass | 0.35 | 0.25 | 0.18 | 0.12 | 0.07 | 0.04 |
| wood panel | 0.28 | 0.22 | 0.17 | 0.09 | 0.10 | 0.11 |
| carpet | 0.08 | 0.24 | 0.57 | 0.69 | 0.71 | 0.73 |
| curtains | 0.07 | 0.31 | 0.49 | 0.75 | 0.70 | 0.60 |
| acoustic tile | 0.50 | 0.70 | 0.60 | 0.70 | 0.70 | 0.50 |
| rubber (experimental) | 0.04 | 0.04 | 0.08 | 0.12 | 0.03 | 0.10 |
| water (experimental) | 0.008 | 0.008 | 0.013 | 0.015 | 0.020 | 0.025 |
| flesh (experimental) | 0.25 | 0.35 | 0.45 | 0.55 | 0.60 | 0.65 |

The optional **material filter** (gain + low-pass) can be added on top of any room, including imported-IR rooms. It may double-count effects already in a measured IR.

### Passage types

| Passage | Low-pass cutoff | Level |
| --- | --- | --- |
| open | none | 0 dB |
| door | 2.5 kHz | −6 dB |
| window | 5 kHz | −12 dB |
| wall | 400 Hz | −20 dB |
| thick wall | 200 Hz | −30 dB |

## Reference sheet: recording impulse responses

Recommendations for recording room IRs with an exponential sine-sweep tool such as the HISSTools Impulse Response Toolbox (HIRT) in Max. These are general swept-sine settings; map them onto the parameter names in your tool's helpfile.

### Settings

| Setting | Recommended | Notes |
| --- | --- | --- |
| Method | Exponential (log) sine sweep | Better signal-to-noise than a balloon or clap; speaker distortion lands before the IR and can be trimmed. |
| Sweep length | 15–20 s | About +3 dB signal-to-noise per doubling, but longer sweeps are more exposed to air movement and temperature drift. Go to 30–45 s only if the noise floor demands it. |
| Frequency range | ~30 Hz – 20 kHz | Start where the speaker actually produces output; stay well below Nyquist. |
| Fades | Fade-in ~50–100 ms, very short fade-out | Avoids clicks at the sweep ends. |
| Sample rate / depth | Your project rate (48 kHz is a good default), 24-bit | |
| Level | Loud but undistorted | Peaks around −12 to −6 dBFS at the input; sweep well above the room noise. |
| Tail capture | ≥ 1.5 × expected decay time after the sweep | About 3–5 s for a small room, 8–10 s for a large hall. |
| Repeats | 2–4 averaged runs | About −3 dB noise per doubling; pause between runs at least as long as the IR. |
| Latency | Record a loopback channel if available | Lets you remove interface latency. |

### Positions and gear

- Omni measurement mic and a small full-range speaker about 0.5–1 m apart, in a representative spot.
- Turn off HVAC, keep the room empty, keep the mic away from reflective surfaces.
- **Note the room's temperature and humidity.** ChamberResonator assumes imported IRs were recorded at 20 °C and 50 % RH.

### After recording

1. Trim the start to just before the direct sound (1–2 ms). Leading silence delays everything.
2. Remove content below about 20 Hz.
3. Apply a short fade-out at the tail.
4. Normalise so the **direct-sound peak is 1.0** (the built-in synthetic rooms use that level). Save as 32-bit float WAV.

### Mono now, multichannel later

- Record all channels at once, with the same sweep and the same deconvolution settings.
- Trim every channel by the same number of samples so inter-channel timing stays intact.
- For 8.1.7, a first-order Ambisonic mic (4 channels, decoded to your speakers) is a practical start.
- ChamberResonator currently imports **mono** IRs only (multi-channel files are averaged to mono).

## Limitations

- Creative approximation, not an acoustic prediction. Cascade mode is serial and does not model coupled-room decay.
- One material per room; imported IRs are mono only.
- Early reflections are first-order and horizontal-plane (plus flat ceiling/floor echoes); no dome focusing or ceiling tilt.
- Reflection directions are fixed to the plan, not to the walking direction.
- No distance cue between source and listener in Timed mode (they are the same point).
- A 16-channel render is memory- and time-hungry: roughly 6 s to render 4 s of audio in testing. Very long inputs need a lot of RAM.
- Resizing a room does not move existing path points.
- No project save/load yet.
- Preview needs `sounddevice` and renders in mono; it is untested.

## Ideas / roadmap

- Per-room "recorded at" temperature and humidity for imported IRs.
- Automatic direct-peak normalisation on IR import.
- Multichannel IR import (1/2/4/16 channels).
- Optional heading-relative reflection directions.
- Project save/load.
- Source-to-listener distance cues as an optional mode.

## Credits and licence

- Timed path, room shapes and extra materials were adapted from a separate React prototype.
- IR recording guidance refers to the [HISSTools Impulse Response Toolbox](https://github.com/HISSTools) for Max.
- **Licence:** not yet chosen. Add a `LICENSE` file before publishing.
