"""Seeded SI polygon cores; synthetic materials, not scanned objects.

Convex hulls and star-shaped concave compounds have exact polygon mass moments.
Compound triangles share edges but have disjoint interiors. Collision skins can
overlap at those seams; this adapter does not suppress internal fixture features.
"""
import numpy as np
from scipy.spatial import ConvexHull

from research.rigid_scenes import body, floor, make_scene
from research.container_scenes import container


def moments(vertices):
    """Area, centroid and polar area moment about the centroid (CCW core)."""
    a = np.asarray(vertices, float); b = np.roll(a, -1, axis=0)
    cross = a[:, 0]*b[:, 1] - b[:, 0]*a[:, 1]
    area = cross.sum()/2
    if area <= 0: raise ValueError('CCW nonzero polygon required')
    center = ((a+b)*cross[:, None]).sum(axis=0)/(6*area)
    polar = (cross*(np.sum(a*a, axis=1)+np.sum(a*b, axis=1)+np.sum(b*b, axis=1))).sum()/12
    return float(area), center, float(polar-area*(center@center))


def generate(rng, concave=False, radius=.22, friction=.4):
    """Unit mass, COM-centred, irregular convex hull or concave triangular fan."""
    if concave:
        count = int(rng.integers(4, 7))
        angles = np.arange(2*count)*np.pi/count
        radii = np.where(np.arange(2*count)%2, rng.uniform(.28, .42, 2*count),
                         rng.uniform(.85, 1., 2*count))*radius
        outline = np.column_stack((np.cos(angles), np.sin(angles)))*radii[:, None]
        pieces = [np.array([[0., 0.], outline[i], outline[(i+1)%len(outline)]])
                  for i in range(len(outline))]
    else:
        for _ in range(1000):
            points = rng.uniform(-1, 1, (int(rng.integers(5, 9)), 2))
            outline = points[ConvexHull(points).vertices]
            outline *= radius/np.max(np.linalg.norm(outline, axis=1))
            if np.min(np.linalg.norm(np.roll(outline, -1, axis=0)-outline, axis=1)) >= .025:
                break
        else: raise RuntimeError('Convex generation failed')
        pieces = [outline]
    parts = [moments(p) for p in pieces]
    area = sum(p[0] for p in parts)
    center = sum(a*c for a, c, _ in parts)/area
    inertia = sum(i+a*((c-center)@(c-center)) for a, c, i in parts)/area
    scale = radius/np.max(np.linalg.norm(outline-center, axis=1))
    area *= scale**2; inertia *= scale**2
    fixtures = [{'vertices': ((p-center)*scale).tolist(), 'density': 1/area,
                 'friction': friction, 'restitution': 0} for p in pieces]
    return fixtures, {'mass_kg': 1., 'inertia_kg_m2': inertia,
                      'outline': ((outline-center)*scale).tolist(), 'concave': concave}


def scenes(seeds=(42, 7301)):
    cases = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        for concave in (False, True):
            shape, geometry = generate(rng, concave, radius=.4)
            scene = make_scene(f'random_{seed}_{"concave" if concave else "convex"}_drop',
                [floor(.4), body(shape, (0, 1.1), (1., -.5), angle=.37, omega=1.3)], duration=1.5)
            scene['generated_geometry'] = [geometry]; cases.append(scene)
        shapes = [generate(rng, radius=.3) for _ in range(2)]
        scene = make_scene(f'random_{seed}_oblique_pair',
            [body(shapes[0][0], (-.6, .04), (1., .1), omega=.7),
             body(shapes[1][0], (.6, -.04), (-1., -.1), omega=-.4)],
            duration=1, gravity=(0, 0))
        scene['generated_geometry'] = [s[1] for s in shapes]; cases.append(scene)
        shapes = [generate(rng, concave=i%3 == 0, radius=.18) for i in range(36)]
        # Bounding circles incl. skin guarantee no initial overlap in these cells.
        spacing = .5; half = 1.65
        schedule = [{'time_s': 0, 'velocity': [.6, 0]}, {'time_s': .5, 'velocity': [-.6, 0]},
                    {'time_s': 1, 'velocity': [.6, 0]}]
        scene = make_scene(f'random_{seed}_mixed36_shake',
            [container(half, half, schedule=schedule),
             *[body(s[0], ((i%6-2.5)*spacing, (i//6-2.5)*spacing),
                     angle=float(rng.uniform(-np.pi, np.pi))) for i, s in enumerate(shapes)]], duration=1.5)
        scene['generated_geometry'] = [s[1] for s in shapes]
        scene['container_half_extents_m'] = [half, half]; cases.append(scene)
    return cases


def contact_chain(count, seed, concave=False):
    """Real polygon support-vertex contacts, no skin, zero gaps, two driven walls.

    Adjacent disjoint x-intervals touch at shared support vertices. Horizontal
    normals belong to both support cones. The two wall planes share one prescribed
    rigid body. Unlike centred disks, these contacts couple normal and tangent.
    This is a frozen contact test, not the native engine's contact discovery.
    """
    rng = np.random.default_rng(seed)
    shapes = [generate(rng, concave) for _ in range(count)]
    centers = []; contacts = []; cursor = np.array([0., 0.])
    for i, (_, geometry) in enumerate(shapes):
        vertices = np.asarray(geometry['outline'])
        left, right = vertices[np.argmin(vertices[:, 0])], vertices[np.argmax(vertices[:, 0])]
        center = cursor-left; centers.append(center)
        contacts.append((i, count if i == 0 else i-1, cursor.tolist(), [1., 0.]))
        cursor = center+right
    contacts.append((count-1, count, cursor.tolist(), [-1., 0.]))
    centers.append([0., 0.])
    velocity = rng.normal(0, .35, (count+1, 3)); velocity[-1] = [1., .2, 0.]
    return {'centers': np.asarray(centers), 'mass': np.array([1.]*count+[np.inf]),
            'inertia': np.array([s[1]['inertia_kg_m2'] for s in shapes]+[np.inf]),
            'contacts': contacts, 'velocity': velocity.ravel(), 'geometry': [s[1] for s in shapes]}
