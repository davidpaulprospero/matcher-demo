"""Generate the synthetic stock-footage clip catalogue.

WHY SYNTHETIC
-------------
The demo needs a pool of candidate clips to match against, but no real stock
library was licensed for it. Downloading clips is not possible here anyway:
no Pexels/Unsplash API key, and a free host should not hold one. So the corpus
is *generated* from a fixed vocabulary and a seeded RNG.

That makes it reproducible: `python scripts/generate_corpus.py` writes the same
bytes every time, so anyone can check the data rather than trust it.

This is NOT scraped from a stock library, and the captions are not captions a
real catalogue produced. They are labelled `"synthetic": true` in the output
and the UI says so. See README.md.

Run:  python scripts/generate_corpus.py
Writes: data/catalog.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

SEED = 20261017
TARGET_CLIPS = 240

# Lighting/time qualifiers. These exist to make every caption unique rather
# than to add information: with only mood+shot+place the generator produced 269
# clips across just 200 distinct captions, so two different clips would show the
# same text in the UI and a "runner-up" could read as a duplicate of the pick.
# Real stock captions do carry this kind of detail.
LIGHTING = [
    "in flat morning light", "under overcast skies", "in warm late light",
    "in cool shade", "backlit", "in low winter light", "at dusk",
    "in hard sunlight", "under soft indoor light", "in blue hour",
    "in diffused window light", "lit from one side",
]

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "catalog.json"

# Each source video is a coherent "shoot" — a place plus a recurring subject.
# Keeping them coherent matters: if captions were random words, the whole
# source-diversity feature would be untestable, because every clip would look
# equally unrelated to its neighbours.
SHOOTS = [
    {
        "source": "synthetic/coast-dawn",
        "subject": "a rocky coastline at first light",
        "places": ["coast", "beach", "ocean", "cliff", "shoreline"],
        "moods": ["calm", "vast", "quiet"],
        "shots": [
            "wide shot of waves breaking over dark rocks",
            "slow push along a wet sand shoreline",
            "close-up of foam retreating from pebbles",
            "aerial drift over cliff edges and white water",
            "low angle on tide pooling between rocks",
        ],
        "has_face": False,
    },
    {
        "source": "synthetic/market-noon",
        "subject": "a crowded covered food market",
        "places": ["market", "street", "stall", "square"],
        "moods": ["busy", "lively", "warm"],
        "shots": [
            "handheld walk past produce stalls",
            "close-up of hands sorting vegetables",
            "overhead of spices in open sacks",
            "vendor weighing produce on a scale",
            "crowd moving through a narrow aisle",
        ],
        "has_face": True,
    },
    {
        "source": "synthetic/office-focus",
        "subject": "a quiet open-plan office",
        "places": ["office", "desk", "studio", "workspace"],
        "moods": ["focused", "calm", "methodical"],
        "shots": [
            "over-shoulder of someone typing at a desk",
            "slow dolly past rows of monitors",
            "close-up of a keyboard and a notebook",
            "empty meeting room with chairs stacked",
            "hand writing notes in a paper planner",
        ],
        "has_face": True,
    },
    {
        "source": "synthetic/city-night",
        "subject": "a downtown street after dark",
        "places": ["city", "street", "building", "bridge"],
        "moods": ["restless", "neon", "late"],
        "shots": [
            "long exposure of traffic light trails",
            "low angle up a lit tower at night",
            "rain on pavement reflecting shop signs",
            "pedestrians crossing an empty intersection",
            "slow push along a neon-lit storefront",
        ],
        "has_face": True,
    },
    {
        "source": "synthetic/workshop-hands",
        "subject": "a woodworker's bench",
        "places": ["workshop", "bench", "garage", "shop"],
        "moods": ["deliberate", "tactile", "patient"],
        "shots": [
            "close-up of a hand plane on oak",
            "sawdust catching light in a beam",
            "hands fitting a joint together",
            "macro of chisel edge and curled shavings",
            "wide of a bench with half-finished pieces",
        ],
        "has_face": False,
    },
    {
        "source": "synthetic/lab-bench",
        "subject": "a working laboratory bench",
        "places": ["lab", "bench", "room"],
        "moods": ["clinical", "precise", "measured"],
        "shots": [
            "gloved hands pipetting into a rack",
            "rack focus across glassware on a bench",
            "a monitor showing a flat trend line",
            "close-up of a labelled sample tray",
            "slow drift across a fume hood",
        ],
        "has_face": False,
    },
    {
        "source": "synthetic/hills-morning",
        "subject": "misty hills at sunrise",
        "places": ["mountain", "hill", "valley", "field", "path"],
        "moods": ["quiet", "wide", "still", "patient"],
        "shots": [
            "wide of fog sitting in a valley",
            "walking shot along a muddy field path",
            "close-up of wet grass in low light",
            "timelapse of light moving over a ridge",
            "pan across layered hills into haze",
        ],
        "has_face": False,
    },
    {
        "source": "synthetic/server-room",
        "subject": "a cold aisle of server racks",
        "places": ["corridor", "room", "building"],
        "moods": ["cold", "steady", "repeating"],
        "shots": [
            "slow dolly down a lit server aisle",
            "close-up of blinking drive LEDs",
            "rack of network cables in shadow",
            "indicator lights blinking in the dark",
            "wide of a cold aisle, condensation near the floor",
        ],
        "has_face": False,
    },
    {
        "source": "synthetic/kitchen-morning",
        "subject": "a home kitchen before breakfast",
        "places": ["kitchen", "table", "room"],
        "moods": ["domestic", "early", "ordinary"],
        "shots": [
            "close-up of coffee dripping into a cup",
            "morning light across a laid table",
            "hands opening curtains over a sink",
            "steam rising from a kettle",
            "quiet pan of an empty kitchen",
        ],
        "has_face": True,
    },
    {
        "source": "synthetic/commute-platform",
        "subject": "a railway platform in grey weather",
        "places": ["station", "platform", "track", "bridge"],
        "moods": ["waiting", "flat", "ordinary", "routine"],
        "shots": [
            "wide of a platform under grey sky",
            "commuters waiting under a departure board",
            "rails running into flat weather",
            "close-up of a station clock",
            "slow push along a wet platform edge",
        ],
        "has_face": True,
    },
]

CAMERAS = [
    "static wide", "slow push in", "slow pull back", "handheld follow",
    "locked-off medium", "low angle", "overhead", "rack focus",
    "slow lateral track", "handheld drift",
]


def build() -> dict:
    rng = random.Random(SEED)

    clips = []
    seen_captions: set[str] = set()

    for shoot in SHOOTS:
        # A shoot yields 20-30 clips, so the source-diversity cap has something
        # real to prevent: one source would otherwise own most of the timeline.
        cursor = 0.0
        for i in range(rng.randint(20, 30)):
            shot = rng.choice(shoot["shots"])
            place = rng.choice(shoot["places"])
            mood = rng.choice(shoot["moods"])

            caption = f"{mood} {shot} at the {place}"

            # Append a lighting qualifier until the caption is unique. Retrying
            # rather than sampling blindly: the qualifier pool is much smaller
            # than the space of all captions, so a plain `while` with a random
            # choice would not reliably terminate for a busy shoot.
            lighting = rng.choice(LIGHTING)
            attempts = 0
            while caption in seen_captions and attempts < len(LIGHTING):
                caption = f"{mood} {shot} at the {place}, {lighting}"
                attempts += 1
                if caption in seen_captions:
                    lighting = LIGHTING[(LIGHTING.index(lighting) + 1) % len(LIGHTING)]

            # A shot template repeated past the qualifier pool would loop
            # forever, so the shot itself is advanced before giving up.
            if caption in seen_captions:
                shot = rng.choice(shoot["shots"])
                caption = f"{mood} {shot} at the {place}, {lighting}"

            seen_captions.add(caption)
            title = caption.replace(" at the ", " — ").title()

            # Clips are laid end to end from a running cursor, not at a fixed
            # 4s stride. A fixed stride lets an 8s clip overlap the next one,
            # and an overlapping clip is unusable as a cut — the timeline
            # export would place two shots on top of each other.
            duration = rng.choice([3.0, 4.0, 5.0, 6.0, 8.0])
            start = round(cursor, 2)
            cursor += duration + rng.uniform(0.5, 2.5)  # a little dead air

            clips.append({
                "id": f"{shoot['source'].split('/')[-1]}-{i:03d}",
                "source": shoot["source"],
                "subject": shoot["subject"],
                "caption": caption,
                "title": title,
                "camera": rng.choice(CAMERAS),
                "lighting": lighting,
                "has_face": shoot["has_face"],
                "place": place,
                "mood": mood,
                "start_time": start,
                "duration": duration,
            })

    return {
        "synthetic": True,
        "note": (
            "Generated by scripts/generate_corpus.py with a fixed seed. Not "
            "scraped from any stock library; these captions were written for "
            "this demo. There are no real clips behind these IDs."
        ),
        "seed": SEED,
        "clip_count": len(clips),
        "clips": clips,
    }


def main() -> None:
    data = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=2), encoding="utf-8")

    sources = len({c["source"] for c in data["clips"]})
    print(f"wrote {OUT}")
    print(f"  {data['clip_count']} clips across {sources} synthetic sources")
    print(f"  seed={SEED} — re-running reproduces this file exactly")


if __name__ == "__main__":
    main()