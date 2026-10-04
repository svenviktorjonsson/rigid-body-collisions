"""Planar polygon/compound rigid dynamics through pinned Box2D comparators.

Build: cmake -S rigid_backend -B build/rigid_backend -DCMAKE_BUILD_TYPE=Release
       cmake --build build/rigid_backend -j 4
Run a scene: python rigid_engine.py scene.json --primary-steps 1 --substeps 4
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

import numpy as np


BOX2D_COMMITS = {"temporal": "8c661469c9507d3ad6fbd2fea3f1aa71669c2fe3",
                 "block": "9ebbbcd960ad424e03e5de6e66a40764c16f51bc"}
BINARIES = {"temporal": Path(__file__).parent / "build/rigid_backend/rigid_runner",
            "block": Path(__file__).parent / "build/rigid_block/rigid_runner"}
DEFAULT_BINARY = BINARIES["block"]
DEFAULT_POLICY = {"travel_threshold": 0.1, "penetration_threshold": 0.02,
                  "island_threshold": 6, "mass_ratio_threshold": 50,
                  "dwell_frames": 12, "minimum_level": 0,
                  "high_primary_steps": 4, "high_substeps": 16}


def _finite(value, name):
    array = np.asarray(value, dtype=float)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite")
    return array


def _positive_integer(value, name, maximum=128):
    if type(value) is not int or not 1 <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [1,{maximum}]")
    return value


def validate_scene(scene):
    if not scene.get("bodies"):
        raise ValueError("At least one body is required")
    duration = float(_finite(scene.get("duration"), "duration"))
    hertz = float(_finite(scene.get("contact_hertz", 10), "contact_hertz"))
    skin = float(_finite(scene.get("collision_skin_m", .01), "collision_skin_m"))
    if duration <= 0 or hertz <= 0:
        raise ValueError("Positive duration and contact frequency required")
    if skin < 0:
        raise ValueError("Collision skin cannot be negative")
    gravity = _finite(scene.get("gravity", [0, -9.81]), "gravity")
    if gravity.shape != (2,):
        raise ValueError("Gravity must have two components")
    for body in scene["bodies"]:
        if body.get("type", "dynamic") not in ("dynamic", "static"):
            raise ValueError("Supported body types are dynamic and static")
        for key in ("position", "velocity"):
            if _finite(body.get(key, [0, 0]), key).shape != (2,):
                raise ValueError(f"{key} must have two components")
        for key in ("angle", "omega"):
            if _finite(body.get(key, 0), key).shape != ():
                raise ValueError(f"{key} must be scalar")
        if np.linalg.norm(body.get("velocity", [0, 0])) > 500:
            raise ValueError("Velocity exceeds this adapter's declared 500 m/s limit")
        if body.get("type", "dynamic") == "static" and (
            np.any(body.get("velocity", [0, 0])) or body.get("omega", 0)
        ):
            raise ValueError("Static bodies cannot carry prescribed velocity")
        if not body.get("polygons"):
            raise ValueError("Every body needs one or more convex polygon fixtures")
        for shape in body["polygons"]:
            vertices = _finite(shape.get("vertices"), "vertices")
            if vertices.ndim != 2 or vertices.shape[1] != 2 or not 3 <= len(vertices) <= 8:
                raise ValueError("Each convex polygon needs 3 to 8 ordered vertices")
            edges = np.roll(vertices, -1, axis=0) - vertices
            following = np.roll(edges, -1, axis=0)
            cross = edges[:, 0] * following[:, 1] - edges[:, 1] * following[:, 0]
            if not (np.all(cross > 1e-8) or np.all(cross < -1e-8)):
                raise ValueError("Strictly convex polygons required; decompose concave shapes")
            side = (edges[:, 0, None] * (vertices[None, :, 1] - vertices[:, None, 1])
                    - edges[:, 1, None] * (vertices[None, :, 0] - vertices[:, None, 0]))
            if np.min(np.sign(cross[0]) * side) < -1e-8:
                raise ValueError("Polygon vertices must follow the convex boundary")
            if np.min(np.linalg.norm(edges, axis=1)) < 0.01:
                raise ValueError("Edges must be at least 0.01 m for this meter-scale adapter")
            for key, default in (("density", 1), ("friction", 0.3), ("restitution", 0), ("rolling", 0)):
                value = float(_finite(shape.get(key, default), key))
                if value < 0 or (key == "density" and body.get("type", "dynamic") == "dynamic" and value <= 0):
                    raise ValueError(f"Invalid {key}")
                if key == "restitution" and value > 1:
                    raise ValueError("Restitution must lie in [0,1]")


def run(scene, *, dt=1 / 120, primary_steps=1, substeps=4, policy=None, backend="block", binary=None):
    """Run an entire scene, retaining the same world through adaptive changes.

    State columns: COM x,y [m], angle [rad], vx,vy [m/s], omega [rad/s].
    Fixture density is areal [kg/m²]. Rolling capacity coefficient is a length
    [m] in Box2D's moment-impulse bound. Numerical contact hertz is not a modulus.
    The substeps argument means temporal substeps for the temporal backend and
    velocity iterations for the block backend. primary_steps subdivides collision
    updates in both. The block backend rejects nonzero rolling coefficients.
    """
    validate_scene(scene)
    if backend not in BINARIES:
        raise ValueError("Choose block or temporal backend")
    dt = float(_finite(dt, "dt"))
    if dt <= 0:
        raise ValueError("Positive output-frame timestep required")
    frames = int(round(scene["duration"] / dt))
    if frames < 1 or not np.isclose(frames * dt, scene["duration"], rtol=1e-9, atol=1e-12):
        raise ValueError("Duration must be an integer number of output frames")
    _positive_integer(primary_steps, "primary_steps", 64)
    _positive_integer(substeps, "substeps")
    p = dict(DEFAULT_POLICY)
    if policy is not None:
        if not isinstance(policy, dict) or set(policy) - set(p):
            raise ValueError("Unknown adaptive policy fields")
        p.update(policy)
    for key in ("travel_threshold", "penetration_threshold", "mass_ratio_threshold"):
        if float(_finite(p[key], key)) <= 0:
            raise ValueError(f"Positive {key} required")
    for key in ("island_threshold", "dwell_frames", "high_primary_steps", "high_substeps"):
        _positive_integer(p[key], key)
    if type(p["minimum_level"]) is not int or not 0 <= p["minimum_level"] <= 3:
        raise ValueError("minimum_level must lie in [0,3]")
    hertz = scene.get("contact_hertz", 10)
    # Box2D caps authored contact hertz at 0.125/internal_dt. Avoid silently
    # changing that regularization between numerical fidelity modes.
    largest_internal_dt = dt if policy is not None else dt / (primary_steps * substeps)
    if backend == "temporal" and hertz > 0.125 / largest_internal_dt:
        raise ValueError("Contact frequency would be clipped at this fidelity")
    if backend == "block":
        if scene.get("collision_skin_m", .01) < .005:
            raise ValueError("Block comparator's CCD requires at least 0.005 m collision skin")
        for body in scene["bodies"]:
            if any(s.get("rolling", 0) for s in body["polygons"]):
                raise ValueError("Block comparator has no rolling resistance model")
            if np.linalg.norm(body.get("velocity", [0, 0])) * dt > 2:
                raise ValueError("Initial speed would be clipped by the block backend's translation cap")
            if abs(body.get("omega", 0)) * dt > np.pi/2:
                raise ValueError("Initial spin would be clipped by the block backend's rotation cap")
    values = [dt, frames, *scene.get("gravity", [0, -9.81]), hertz, scene.get("collision_skin_m", .01),
              int(policy is not None), primary_steps, substeps,
              p["travel_threshold"], p["penetration_threshold"], p["island_threshold"],
              p["mass_ratio_threshold"], p["dwell_frames"], p["minimum_level"],
              p["high_primary_steps"], p["high_substeps"], len(scene["bodies"])]
    for body in scene["bodies"]:
        values.extend([2 if body.get("type", "dynamic") == "dynamic" else 0,
                       int(body.get("fixed_rotation", False)), int(body.get("bullet", False)),
                       *body.get("position", [0, 0]), body.get("angle", 0),
                       *body.get("velocity", [0, 0]), body.get("omega", 0), len(body["polygons"])])
        for shape in body["polygons"]:
            values.extend([len(shape["vertices"]), shape.get("density", 1), shape.get("friction", 0.3),
                           shape.get("restitution", 0), shape.get("rolling", 0)])
            values.extend(np.asarray(shape["vertices"]).ravel().tolist())
    executable = Path(binary) if binary is not None else BINARIES[backend]
    if not executable.is_file():
        raise FileNotFoundError(f"Build the pinned backend first: {executable}")
    start = time.perf_counter()
    process = subprocess.run([str(executable.resolve())], input=" ".join(map(str, values)),
                             text=True, capture_output=True)
    if process.returncode:
        raise RuntimeError(f"Rigid backend failed: {process.stderr.strip()}")
    result = json.loads(process.stdout)
    if not np.all(np.isfinite(result["states"])):
        raise RuntimeError("Nonfinite engine state")
    physical = {"bodies": scene["bodies"], "gravity": scene.get("gravity", [0, -9.81]),
                "duration": scene["duration"], "units": "m,kg,s; areal density",
                "collision_skin_m": scene.get("collision_skin_m", .01), "mass_geometry": "unrounded polygon cores",
                "friction_law": "single-coefficient dry friction", "restitution_threshold": 0}
    numerical_model = {"backend": backend, "commit": BOX2D_COMMITS[backend],
                       "contact_hertz": hertz if backend == "temporal" else None,
                       "contact_damping_ratio": 1 if backend == "temporal" else None,
                       "max_contact_push_speed": 1 if backend == "temporal" else None,
                       "position_iterations": 3 if backend == "block" else None,
                       "sleep": False, "continuous_collision": True}
    states = np.asarray(result["states"]); final = states[-1]
    mass = np.asarray(result["mass"]); inertia = np.asarray(result["inertia"])
    kinetic = .5 * np.sum(mass * np.sum(final[:, 3:5]**2, axis=1) + inertia * final[:, 5]**2)
    observables = {"kinetic_energy_final": {"value": float(kinetic), "unit": "J"},
                   "com_height_final": {"value": float(np.sum(mass * final[:, 1])/np.sum(mass)), "unit": "m"}}
    for i, state in enumerate(final):
        for column, name, unit in ((3, "vx", "m/s"), (4, "vy", "m/s"), (5, "omega", "rad/s")):
            observables[f"body_{i}_{name}"] = {"value": float(state[column]), "unit": unit}
    result.update({"schema_version": 1, "case_id": scene.get("id", "unnamed"),
                   "physical_setup_id": hashlib.sha256(json.dumps(physical, sort_keys=True).encode()).hexdigest(),
                   "numerical_model": numerical_model, "observables": observables,
                   "evidence_kind": "numerical_simulation", "parameter_provenance": scene.get("parameter_provenance", "synthetic_declared"),
                   "wall_time_s": time.perf_counter() - start,
                   "engine_and_controller_s": result["step_s"] + result["controller_s"],
                   "times": (np.arange(frames + 1) * dt).tolist(),
                   "fidelity": {"output_dt_s": dt, "primary_steps": primary_steps,
                                "solver_steps": substeps,
                                "solver_steps_meaning": "velocity_iterations" if backend == "block" else "temporal_substeps",
                                "policy": p if policy is not None else None}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-steps", type=int, default=1)
    parser.add_argument("--substeps", type=int, default=4)
    parser.add_argument("--adaptive", action="store_true")
    parser.add_argument("--backend", choices=BINARIES, default="block")
    args = parser.parse_args()
    result = run(json.loads(args.scene.read_text()), primary_steps=args.primary_steps,
                 substeps=args.substeps, policy={} if args.adaptive else None, backend=args.backend)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
