"""Generate three example .tdance scene files in dance/examples/.

Run once:
    cd dance
    python make_examples.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from dancer import Dancer, DanceScene, generate_dancer
from presets import get_preset
from scene_io import save_scene

OUT = Path(__file__).parent / "examples"
OUT.mkdir(exist_ok=True)


def _dancer(name: str, preset: str, color: str,
            start: tuple = (0.0, 0.0),
            phase: float = 0.0,
            speed: float = 1.0,
            n_cycles: int = 1) -> Dancer:
    d = Dancer.new(f"preset:{preset}", get_preset(preset), name, color)
    d.start_offset_xy = start
    d.phase_offset = phase
    d.speed_multiplier = speed
    d.n_cycles = n_cycles
    print(f"  Generating {name} ({preset})...", end=" ", flush=True)
    generate_dancer(d)
    print("done")
    return d


# -- Scene 1: Trio -------------------------------------------------------------
# 세 댄서가 삼각형 꼭짓점에서 서로 다른 커브로 출발, 1/3씩 위상 차이
print("=== Scene 1: Trio ===")
s1 = DanceScene(duration_seconds=12.0, loop=True)
s1.add(_dancer("Alice", "figure_eight", "#e6194b", start=(-4.0,  2.5), phase=0.0))
s1.add(_dancer("Bob",   "star_5",       "#4363d8", start=( 0.0, -3.0), phase=0.33))
s1.add(_dancer("Carol", "peanut",       "#3cb44b", start=( 4.0,  2.5), phase=0.66))
save_scene(s1, OUT / "trio.tdance")
print("  -> trio.tdance\n")

# -- Scene 2: Orbit ------------------------------------------------------------
# 다섯 댄서가 원형으로 배치, 균등 위상 차이
print("=== Scene 2: Orbit ===")
s2 = DanceScene(duration_seconds=16.0, loop=True)
orbit_colors  = ["#e6194b", "#f58231", "#3cb44b", "#4363d8", "#911eb4"]
orbit_presets = ["circle", "peanut", "circle", "peanut", "circle"]
R = 5.0
for i in range(5):
    angle = 2 * np.pi * i / 5
    x, y = R * np.cos(angle), R * np.sin(angle)
    s2.add(_dancer(
        f"Orbit {i+1}", orbit_presets[i], orbit_colors[i],
        start=(x, y), phase=i / 5,
    ))
save_scene(s2, OUT / "orbit.tdance")
print("  -> orbit.tdance\n")

# -- Scene 3: Canon ------------------------------------------------------------
# 네 댄서 같은 star_5 커브, 일직선 배치, 카논 (1/4씩 위상 차이)
print("=== Scene 3: Canon ===")
s3 = DanceScene(duration_seconds=14.0, loop=True)
canon_colors = ["#e6194b", "#f032e6", "#4363d8", "#46f0f0"]
for i in range(4):
    s3.add(_dancer(
        f"C{i+1}", "star_5", canon_colors[i],
        start=(-4.5 + 3.0 * i, 0.0), phase=i / 4,
    ))
save_scene(s3, OUT / "canon.tdance")
print("  -> canon.tdance\n")

# -- Scene 4: Trefoil --------------------------------------------------------
print("=== Scene 4: Trefoil ===")
s4 = DanceScene(duration_seconds=12.0, loop=True)
s4.add(_dancer("Treo", "trefoil", "#ff7f00", start=(0.0, 0.0), phase=0.0))
save_scene(s4, OUT / "trefoil.tdance")
print("  -> trefoil.tdance\n")

print("All done. Files in:", OUT)
