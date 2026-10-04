"""Declared planar verification/stress cases; coefficients are not measurements."""
import math


def rectangle(hx=0.5, hy=0.5, offset=(0, 0), **material):
    x, y = offset
    return {"vertices": [[x-hx, y-hy], [x+hx, y-hy], [x+hx, y+hy], [x-hx, y+hy]], **material}


def regular_polygon(n=6, radius=0.5, **material):
    return {"vertices": [[radius * math.cos(2*math.pi*i/n), radius * math.sin(2*math.pi*i/n)]
                         for i in range(n)], **material}


def body(shape, position=(0, 0), velocity=(0, 0), **kwargs):
    return {"position": list(position), "velocity": list(velocity),
            "polygons": shape if isinstance(shape, list) else [shape], **kwargs}


def floor(friction=0.3):
    return body(rectangle(40, 0.2, friction=friction), (0, -0.2), type="static")


def make_scene(id, bodies, duration=3, gravity=(0, -9.81), split="test", **extra):
    return {"id": id, "duration": duration, "gravity": list(gravity), "contact_hertz": 10, "collision_skin_m": .01,
            "bodies": bodies, "split": split, "parameter_provenance": "synthetic_declared",
            "material_authenticity": "not experimentally characterized", **extra}


def scenes():
    cases = []
    cases.append(make_scene("free_flight_hexagon", [body(regular_polygon(), (0, 3), (1, 2), omega=1)],
                            duration=1, gravity=(0, 0), split="verification"))
    cases.append(make_scene("normal_rebound_boxes", [
        body(rectangle(friction=0, restitution=0.6), (-1.5, 0), (2, 0)),
        body(rectangle(friction=0, restitution=0.6), (1.5, 0), (-2, 0))],
        duration=1.2, gravity=(0, 0), split="verification"))
    cases.append(make_scene("sliding_box", [floor(), body(rectangle(friction=0.3), (0, 0.525), (3, 0), fixed_rotation=True)],
        duration=2, split="train", analytic={"law": "flat_slider", "speed": 3, "friction": 0.3}))
    for mu, name in ((0.6, "incline_stick"), (0.3, "incline_slide")):
        theta = math.atan(0.5); n = (math.sin(theta), math.cos(theta))
        plane = body(rectangle(30, 0.2, friction=mu), angle=-theta, type="static")
        slider = body(rectangle(friction=mu), [0.725*n[0], 0.725*n[1]], angle=-theta, fixed_rotation=True)
        cases.append(make_scene(name, [plane, slider], duration=2, split="verification",
                               analytic={"law": "incline", "theta": theta, "friction": mu}))
    cases.append(make_scene("triangle_drop", [floor(), body(regular_polygon(3, 0.6, friction=0.4), (0, 2), omega=0.5)],
                            duration=3, split="train"))
    cases.append(make_scene("hexagon_oblique", [floor(0.4), body(regular_polygon(6, 0.6, friction=0.4), (0, 2), (2, -1), omega=2)], duration=3))
    compound = [rectangle(0.5, 0.15, offset=(0, -0.35), friction=0.4),
                rectangle(0.15, 0.35, offset=(-0.35, 0.15), friction=0.4)]
    cases.append(make_scene("concave_L_drop", [floor(0.4), body(compound, (0, 2), (1, 0), omega=1)], duration=3))
    cases.append(make_scene("thin_bar_spin", [floor(0.4), body(rectangle(1.5, 0.1, friction=0.4), (0, 2), omega=2)], duration=3))
    cases.append(make_scene("thin_wall_ccd", [body(rectangle(.025, 3, friction=0), (0, 0), type="static"),
        body(rectangle(.15, .12, friction=0, restitution=0), (-3, 0), (100, 0), bullet=True)], duration=.2, gravity=(0, 0)))
    for count, name, split in ((6, "stack_6", "train"), (12, "stack_12", "test")):
        blocks = [body(rectangle(.4, .4, friction=.4), (0, .425 + .825*i)) for i in range(count)]
        cases.append(make_scene(name, [floor(.4), *blocks], duration=6, split=split,
                                source_topology="Box2D stacking sample; newly generated aligned variant"))
    for ratio, name, split in ((30, "mass_contrast_30", "train"), (100, "mass_contrast_100", "test")):
        blocks = [body(rectangle(.4, .4, friction=.4, density=ratio if i==5 else 1), (0, .425+.825*i))
                  for i in range(6)]
        cases.append(make_scene(name, [floor(.4), *blocks], duration=6, split=split,
                                source_topology="Box2D robustness sample; reduced aligned variant"))
    # Independent rigid bodies arranged in simultaneous impact contact chains.
    chain = [body(rectangle(.25, .35, friction=0), (.525*i, 0), (3, 0) if i==0 else (0, 0)) for i in range(12)]
    cases.append(make_scene("impact_chain_12", chain, duration=2, gravity=(0, 0)))
    # Public friction sample geometry/coefficient adaptation; original code is pinned.
    statics = [floor(.2)]
    for pos, half, angle in (((-4, 22), (13, .25), -.25), ((10.5, 19), (.25, 1), 0),
                             ((4, 14), (13, .25), .25), ((-10.5, 11), (.25, 1), 0),
                             ((-4, 6), (13, .25), -.25)):
        statics.append(body(rectangle(*half, friction=.2), pos, angle=angle, type="static"))
    moving = [body(rectangle(density=25, friction=mu), (-15+4*i, 28))
              for i, mu in enumerate((.75, .5, .35, .1, 0))]
    cases.append(make_scene("public_friction_slopes", [*statics, *moving], duration=12, split="stress",
        parameter_provenance="published_numerical_sample_inputs",
        source_topology="sample_shapes.cpp Friction; segment floor represented by a solid rectangle"))
    blocks = [body(rectangle(.45, .45, friction=.3), (.1*i, .505+i)) for i in range(20)]
    cases.append(make_scene("long_tilted_stack", [floor(.3), *blocks], duration=20, split="stress",
        source_topology="sample_stacking.cpp TiltedStack; reduced single column variant"))
    return cases
