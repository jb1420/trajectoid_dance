"""Serialization / deserialization for DanceScene into .tdance files.

A .tdance file is a zip archive containing:
  scene.json  — all scalar/string metadata (no numpy arrays)
  arrays.npz  — all numpy arrays, keyed by "{dancer_id}__{role}"
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Union

import numpy as np

from dancer import Dancer, DanceScene
from presets import LEGACY_KEY_MIGRATION
from trajectoids_adapter import GenerationResult, RollSimulationResult

TDANCE_VERSION = 1


def save_scene(scene: DanceScene, path: Union[str, Path]) -> None:
    path = Path(path)
    meta, arrays = _build_json_and_arrays(scene)

    tmp = path.with_suffix(".tdance.tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("scene.json", json.dumps(meta, indent=2))
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        zf.writestr("arrays.npz", buf.getvalue())
    tmp.replace(path)


def load_scene(path: Union[str, Path]) -> DanceScene:
    path = Path(path)
    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("scene.json"))
        arrays = np.load(io.BytesIO(zf.read("arrays.npz")), allow_pickle=False)

    version = meta.get("tdance_version", 0)
    if version != TDANCE_VERSION:
        raise ValueError(f"Unsupported .tdance version: {version}")

    return _reconstruct_scene(meta, arrays)


def _build_json_and_arrays(
    scene: DanceScene,
) -> tuple[dict, dict[str, np.ndarray]]:
    arrays: dict[str, np.ndarray] = {}
    dancer_dicts = []

    for d in scene.dancers:
        did = d.dancer_id
        arrays[f"{did}__curve_xy"] = np.asarray(d.curve_xy, dtype=float)

        dd: dict = {
            "dancer_id": d.dancer_id,
            "name": d.name,
            "curve_source": d.curve_source,
            "curve_params": dict(d.curve_params),
            "color_hex": d.color_hex,
            "start_offset_xy": [float(d.start_offset_xy[0]), float(d.start_offset_xy[1])],
            "phase_offset": float(d.phase_offset),
            "speed_multiplier": float(d.speed_multiplier),
            "n_cycles": int(d.n_cycles),
            "closed": bool(d.closed),
            "cycle_arc_length": float(d.cycle_arc_length),
            "has_gen_result": d.gen_result is not None,
            "has_sim_result": d.sim_result is not None,
        }

        if d.gen_result is not None:
            g = d.gen_result
            dd["gen_result_scalars"] = {
                "scale": float(g.scale),
                "mismatch_angle": float(g.mismatch_angle),
                "endpoint_gap": float(g.endpoint_gap),
            }
            arrays[f"{did}__gen_vertices"] = np.asarray(g.vertices, dtype=np.float32)
            arrays[f"{did}__gen_faces"] = np.asarray(g.faces, dtype=np.int32)
            arrays[f"{did}__gen_resampled_points"] = np.asarray(g.resampled_points, dtype=float)
            arrays[f"{did}__gen_normals"] = np.asarray(g.normals, dtype=float)
            arrays[f"{did}__gen_surface_contact_curve"] = np.asarray(g.surface_contact_curve, dtype=float)

        if d.sim_result is not None:
            s = d.sim_result
            dd["sim_result_scalars"] = {
                "achieved_roll_angle_rad": float(s.achieved_roll_angle_rad),
                "completed_target": bool(s.completed_target),
                "message": str(s.message),
            }
            arrays[f"{did}__sim_translations_xyz"] = np.asarray(s.translations_xyz, dtype=float)
            arrays[f"{did}__sim_rotations"] = np.asarray(s.rotations, dtype=float)
            arrays[f"{did}__sim_trajectory_xy"] = np.asarray(s.trajectory_xy, dtype=float)

        dancer_dicts.append(dd)

    meta = {
        "tdance_version": TDANCE_VERSION,
        "scene": {
            "duration_seconds": float(scene.duration_seconds),
            "loop": bool(scene.loop),
            "global_ticks": int(scene.global_ticks),
        },
        "dancers": dancer_dicts,
    }
    return meta, arrays


def _reconstruct_scene(
    meta: dict,
    arrays: dict,
) -> DanceScene:
    s = meta["scene"]
    scene = DanceScene(
        duration_seconds=float(s["duration_seconds"]),
        loop=bool(s["loop"]),
        global_ticks=int(s.get("global_ticks", 480)),
    )

    for dd in meta["dancers"]:
        did = dd["dancer_id"]

        gen_result = None
        if dd.get("has_gen_result"):
            sc = dd["gen_result_scalars"]
            gen_result = GenerationResult(
                vertices=arrays[f"{did}__gen_vertices"].astype(np.float32),
                faces=arrays[f"{did}__gen_faces"].astype(np.int32),
                scale=float(sc["scale"]),
                mismatch_angle=float(sc["mismatch_angle"]),
                endpoint_gap=float(sc["endpoint_gap"]),
                resampled_points=arrays[f"{did}__gen_resampled_points"].astype(float),
                normals=arrays[f"{did}__gen_normals"].astype(float),
                surface_contact_curve=arrays[f"{did}__gen_surface_contact_curve"].astype(float),
            )

        sim_result = None
        if dd.get("has_sim_result"):
            sc = dd["sim_result_scalars"]
            sim_result = RollSimulationResult(
                translations_xyz=arrays[f"{did}__sim_translations_xyz"].astype(float),
                rotations=arrays[f"{did}__sim_rotations"].astype(float),
                trajectory_xy=arrays[f"{did}__sim_trajectory_xy"].astype(float),
                achieved_roll_angle_rad=float(sc["achieved_roll_angle_rad"]),
                completed_target=bool(sc["completed_target"]),
                message=str(sc["message"]),
            )

        # Migrate legacy parametric preset keys ("star_5" → "star" + params).
        curve_source = dd["curve_source"]
        curve_params: dict = dict(dd.get("curve_params", {}))
        if curve_source.startswith("preset:"):
            key = curve_source.split(":", 1)[1]
            if key in LEGACY_KEY_MIGRATION:
                new_key, defaults = LEGACY_KEY_MIGRATION[key]
                curve_source = f"preset:{new_key}"
                # Saved params (likely empty for old files) win over defaults.
                curve_params = {**defaults, **curve_params}

        dancer = Dancer(
            dancer_id=did,
            name=dd["name"],
            curve_source=curve_source,
            curve_xy=arrays[f"{did}__curve_xy"].astype(float),
            color_hex=dd["color_hex"],
            start_offset_xy=tuple(dd["start_offset_xy"]),
            phase_offset=float(dd["phase_offset"]),
            speed_multiplier=float(dd["speed_multiplier"]),
            n_cycles=int(dd["n_cycles"]),
            closed=bool(dd["closed"]),
            curve_params=curve_params,
            gen_result=gen_result,
            sim_result=sim_result,
            cycle_arc_length=float(dd["cycle_arc_length"]),
        )
        scene.add(dancer)

    return scene
