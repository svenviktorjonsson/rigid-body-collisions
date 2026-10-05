"""Planar polygon/circle/compound rigid dynamics through pinned Box2D comparators.

Build: see rigid_backend/README.md for both pinned backends.
Run a scene: python rigid_engine.py scene.json --output result.json --preset high
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
PRESETS = {"fast": (1, 1), "standard": (1, 8), "accurate": (4, 16), "high": (8, 32)}
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
        if body.get("type", "dynamic") not in ("dynamic", "static", "kinematic"):
            raise ValueError("Supported body types are dynamic, static and kinematic")
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
        if body.get("velocity_schedule") is not None:
            if body.get("type") != "kinematic":
                raise ValueError("Prescribed velocity schedules require a kinematic body")
            schedule = body["velocity_schedule"]
            if not isinstance(schedule, list) or not schedule:
                raise ValueError("Nonempty velocity schedule required")
            previous = -1
            for command in schedule:
                time_s = float(_finite(command.get("time_s"), "command time"))
                if not previous < time_s < duration or time_s < 0:
                    raise ValueError("Schedule times must increase within the simulated horizon")
                if _finite(command.get("velocity", [0, 0]), "command velocity").shape != (2,):
                    raise ValueError("Command velocity must have two components")
                omega = float(_finite(command.get("omega", 0), "command omega"))
                if np.linalg.norm(command.get("velocity", [0, 0])) > 500:
                    raise ValueError("Command exceeds the adapter speed limit")
                previous = time_s
        if not body.get("polygons") and not body.get("circles"):
            raise ValueError("Every body needs polygon or circle fixtures")
        for shape in body.get("polygons", []):
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
        for circle in body.get("circles", []):
            radius = float(_finite(circle.get("radius"), "circle radius"))
            if radius < .005:
                raise ValueError("Circle radius must be at least 0.005 m for this adapter")
            if _finite(circle.get("center", [0, 0]), "circle center").shape != (2,):
                raise ValueError("Circle center must have two components")
        for shape in body.get("polygons", []) + body.get("circles", []):
            for key, default in (("density", 1), ("friction", 0.3), ("restitution", 0), ("rolling", 0)):
                value = float(_finite(shape.get(key, default), key))
                if value < 0 or (key == "density" and body.get("type", "dynamic") == "dynamic" and value <= 0):
                    raise ValueError(f"Invalid {key}")
                if key == "restitution" and value > 1:
                    raise ValueError("Restitution must lie in [0,1]")



def run(scene, *, dt=1 / 120, primary_steps=8, substeps=32, policy=None, backend="block", binary=None,
        position_iterations=3):
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
    _positive_integer(primary_steps, "primary_steps", 4096)
    _positive_integer(substeps, "substeps", 4096 if backend == 'block' else 128)
    _positive_integer(position_iterations, 'position_iterations', 128)
    if backend != 'block' and position_iterations != 3:
        raise ValueError('Position iteration override applies to the block backend only')
    p = dict(DEFAULT_POLICY)
    if policy is not None:
        if not isinstance(policy, dict) or set(policy) - set(p):
            raise ValueError("Unknown adaptive policy fields")
        p.update(policy)
    for key in ("travel_threshold", "penetration_threshold", "mass_ratio_threshold"):
        if float(_finite(p[key], key)) <= 0:
            raise ValueError(f"Positive {key} required")
    for key in ("island_threshold", "dwell_frames", "high_primary_steps", "high_substeps"):
        maximum = 4096 if key == 'high_primary_steps' or (key == 'high_substeps' and backend == 'block') else 128
        _positive_integer(p[key], key, maximum)
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
            if any(s.get("rolling", 0) for s in body.get("polygons", []) + body.get("circles", [])):
                raise ValueError("Block comparator has no rolling resistance model")
            if np.linalg.norm(body.get("velocity", [0, 0])) * dt > 2:
                raise ValueError("Initial speed would be clipped by the block backend's translation cap")
            if abs(body.get("omega", 0)) * dt > np.pi/2:
                raise ValueError("Initial spin would be clipped by the block backend's rotation cap")
    commands = []
    for index, body in enumerate(scene["bodies"]):
        for command in body.get("velocity_schedule", []):
            frame = int(round(command["time_s"] / dt))
            if not np.isclose(frame*dt, command["time_s"], rtol=1e-9, atol=1e-12):
                raise ValueError("Velocity schedule changes must align with output frames")
            velocity, omega = command.get("velocity", [0, 0]), command.get("omega", 0)
            if backend == "block" and (np.linalg.norm(velocity)*dt > 2 or abs(omega)*dt > np.pi/2):
                raise ValueError("Prescribed motion would exceed block backend movement caps")
            commands.append((frame, index, *velocity, omega))
    values = ["rigid-v2", dt, frames, *scene.get("gravity", [0, -9.81]), hertz, scene.get("collision_skin_m", .01),
              int(policy is not None), primary_steps, substeps,
              p["travel_threshold"], p["penetration_threshold"], p["island_threshold"],
              p["mass_ratio_threshold"], p["dwell_frames"], p["minimum_level"],
              p["high_primary_steps"], p["high_substeps"], len(scene["bodies"])]
    for body in scene["bodies"]:
        values.extend([{"dynamic": 2, "kinematic": 1, "static": 0}[body.get("type", "dynamic")],
                       int(body.get("fixed_rotation", False)), int(body.get("bullet", False)),
                       *body.get("position", [0, 0]), body.get("angle", 0),
                       *body.get("velocity", [0, 0]), body.get("omega", 0), len(body.get("polygons", [])) + len(body.get("circles", []))])
        for shape in body.get("polygons", []):
            values.extend([1, shape.get("density", 1), shape.get("friction", 0.3),
                           shape.get("restitution", 0), shape.get("rolling", 0), len(shape["vertices"])])
            values.extend(np.asarray(shape["vertices"]).ravel().tolist())
        for shape in body.get("circles", []):
            values.extend([0, shape.get("density", 1), shape.get("friction", 0.3),
                           shape.get("restitution", 0), shape.get("rolling", 0),
                           shape["radius"], *shape.get("center", [0, 0])])
    values.append(len(commands))
    for command in commands: values.extend(command)
    values.extend([int(scene.get('suppress_internal_edges', False)), position_iterations,
                   int(scene.get('analytic_kinematics', False))])
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
                "collision_skin_m": scene.get("collision_skin_m", .01), "mass_geometry": "unrounded polygon cores and exact disks",
                "friction_law": "single-coefficient dry friction", "restitution_threshold": 0}
    numerical_model = {"backend": backend, "commit": BOX2D_COMMITS[backend],
                       "contact_hertz": hertz if backend == "temporal" else None,
                       "contact_damping_ratio": 1 if backend == "temporal" else None,
                       "max_contact_push_speed": 1 if backend == "temporal" else None,
                       "position_iterations": position_iterations if backend == "block" else None,
                       "sleep": False, "continuous_collision": True}
    numerical_model['suppress_internal_edges'] = bool(scene.get('suppress_internal_edges', False))
    numerical_model['scalar_precision'] = result.get('scalar_precision', 'float32')
    numerical_model['analytic_kinematics'] = bool(scene.get('analytic_kinematics', False))
    if numerical_model['scalar_precision'] == 'float64':
        manifest = executable.resolve().parent/'precision-source.json'
        if not manifest.is_file():
            raise RuntimeError('Float64 diagnostic requires its precision-source.json manifest')
        numerical_model['precision_source_sha256'] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        numerical_model['implementation_note'] = 'Locally transformed Float64 Box2D diagnostic; not an upstream Float64 release'
    states = np.asarray(result["states"])
    if states.ndim != 3 or states.shape[1] == 0:
        raise ValueError("At least one dynamic body is required for this state adapter")
    final = states[-1]
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
                   "fidelity": {"output_dt_s": dt,
                                "selection": "fixed" if policy is None else "adaptive",
                                "primary_steps": primary_steps if policy is None else None,
                                "solver_steps": substeps if policy is None else None,
                                "solver_steps_meaning": "velocity_iterations" if backend == "block" else "temporal_substeps",
                                "policy": p if policy is not None else None}})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preset", choices=PRESETS, default="high",
                        help="Fixed numerical effort; high is the measured conservative block baseline")
    parser.add_argument("--primary-steps", type=int, help="Override preset collision updates")
    parser.add_argument("--substeps", type=int, help="Override preset solver steps; see backend semantics")
    parser.add_argument('--position-iterations', type=int, default=3,
                        help='Block backend pose correction work, independently from velocity iterations')
    parser.add_argument("--adaptive", action="store_true", help="Use the experimental dynamic effort controller")
    parser.add_argument("--policy", type=Path, help="Load an adaptive policy dict or frozen-policy.json")
    parser.add_argument("--backend", choices=BINARIES, default="block")
    args = parser.parse_args()
    if args.policy and not args.adaptive:
        parser.error("--policy requires --adaptive")
    primary, solver = PRESETS[args.preset]
    policy = {} if args.adaptive else None
    if args.policy:
        document = json.loads(args.policy.read_text())
        policy = document.get("policy", document)
    result = run(json.loads(args.scene.read_text()),
                 primary_steps=primary if args.primary_steps is None else args.primary_steps,
                 substeps=solver if args.substeps is None else args.substeps, policy=policy, backend=args.backend,
                 position_iterations=args.position_iterations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
