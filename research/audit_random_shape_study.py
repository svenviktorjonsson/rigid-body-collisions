"""Independently audit archived polygon geometry, mechanics and trajectory gates."""
import hashlib
import io
import json
from pathlib import Path
import statistics
import zipfile

import numpy as np

from research.run_rigid_study import errors, normalized_error

ROOT = Path(__file__).parent/'random-shapes'


def polygon_moments(vertices):
    # Integrate triangles from the origin; this does not invoke the generator.
    mass = 0.; first = np.zeros(2); polar = 0.
    for a, b in zip(np.asarray(vertices), np.roll(vertices, -1, axis=0)):
        area = (a[0]*b[1]-a[1]*b[0])/2
        mass += area; first += area*(a+b)/3
        polar += area*(a@a+a@b+b@b)/6
    return mass, first/mass, polar


def point_velocities(velocity, centers, pairs, points, normals):
    v = velocity.reshape(-1, 3); result = []
    for (a, b), point, normal in zip(pairs, points, normals):
        ra, rb = point-centers[a], point-centers[b]
        va = v[a, :2]+v[a, 2]*np.array([-ra[1], ra[0]])
        vb = v[b, :2]+v[b, 2]*np.array([-rb[1], rb[0]])
        tangent = np.array([-normal[1], normal[0]])
        result.append([(va-vb)@normal, (va-vb)@tangent])
    return np.asarray(result)


def audit():
    output = ROOT/'results'; summary = json.loads((output/'summary.json').read_text())
    plan = json.loads((ROOT/'plan.json').read_text())
    for file, field in [('traces.zip', 'traces_sha256'), ('scenes.json', 'scenes_sha256'),
                        ('execution-source.zip', 'source_archive_sha256')]:
        assert hashlib.sha256((output/file).read_bytes()).hexdigest() == summary[field]
    assert hashlib.sha256((ROOT/'plan.json').read_bytes()).hexdigest() == summary['plan_sha256']
    with zipfile.ZipFile(output/'execution-source.zip') as source:
        assert source.read('research/random-shapes/plan.json') == (ROOT/'plan.json').read_bytes()
    scenes = json.loads((output/'scenes.json').read_text()); qualified_count = 0
    assert len(scenes) == len(plan['seeds'])*len(plan['trajectory_cases']) == 8
    assert len(summary['kernels']) == 24
    with zipfile.ZipFile(output/'traces.zip') as archive:
        assert len(archive.namelist()) == 80+12
        traces = {r['trace']: json.loads(archive.read(r['trace']+'.json')) for r in summary['records']}
        assert len(traces) == 80
        for scene in scenes:
            records = [r for r in summary['records'] if r['scene'] == scene['id']]
            for record in records:
                trace = traces[record['trace']]
                if not record['accepted']:
                    assert trace == record and record['failure']
                    continue
                assert len(record['samples_s']) == plan['repeats']
                assert record['median_s'] == statistics.median(record['samples_s'])
                assert np.isfinite(trace['states']).all()
                assert np.min(record['samples_s']) >= 0
                # Integrate the original fixtures, independently of stored generator properties.
                for index, body in enumerate(b for b in scene['bodies'] if b.get('type', 'dynamic') == 'dynamic'):
                    mass = 0.; first = np.zeros(2); polar = 0.
                    for fixture in body['polygons']:
                        a, c, j = polygon_moments(fixture['vertices']); d = fixture['density']
                        mass += d*a; first += d*a*c; polar += d*j
                    inertia = polar-first@first/mass
                    np.testing.assert_allclose(trace['mass'][index], mass, rtol=2e-6, atol=1e-7)
                    np.testing.assert_allclose(trace['inertia'][index], inertia, rtol=2e-5, atol=1e-7)
            key = lambda p, s: f'{scene["id"]}__block_p{p}_s{s}'
            edges = [{'from': a, 'to': b, 'errors': errors(traces[key(*b)], traces[key(*a)])}
                     for a, b in plan['reference_edges']]
            q = all(normalized_error(e['errors'], plan['reference_budget']) <= 1 for e in edges)
            assert summary['qualifications'][scene['id']] == {'qualified': q, 'edges': edges}
            qualified_count += q
            for comp in (c for c in summary['comparisons'] if c['scene'] == scene['id']):
                if not next(r for r in records if r['trace'] == comp['trace'])['accepted']:
                    assert comp['passed'] is False and comp['failure']
                    continue
                error = errors(traces[key(16, 64)], traces[comp['trace']])
                assert comp['errors'] == error and comp['reference_qualified'] == q
                assert comp['passed'] == (q and normalized_error(error, plan['budget']) <= 1)
        for name in sorted(n for n in archive.namelist() if n.endswith('.npz')):
            with np.load(io.BytesIO(archive.read(name)), allow_pickle=False) as data:
                centers = data['centers']; mass = data['mass']; inertia = data['inertia']
                geometry = json.loads(str(data['geometry_json']))
                pairs, points, normals = data['bodies'], data['points'], data['normals']
                for i, g in enumerate(geometry):
                    a, c, j = polygon_moments(g['outline'])
                    np.testing.assert_allclose(c, 0, atol=1e-12)
                    np.testing.assert_allclose(j/a, inertia[i], atol=1e-12)
                for (a, b), point, normal in zip(pairs, points, normals):
                    np.testing.assert_allclose(np.linalg.norm(normal), 1, atol=1e-14)
                    for index, sign in ((a, 1), (b, -1)):
                        if index == len(geometry): continue
                        world = np.asarray(geometry[index]['outline'])+centers[index]
                        assert np.min(np.linalg.norm(world-point, axis=1)) < 1e-12
                        assert np.min(sign*((world-point)@normal)) > -1e-12
                v = data['velocity']; inverse = np.column_stack((1/mass, 1/mass, 1/inertia)).ravel()
                before = point_velocities(v, centers, pairs, points, normals)
                for item in (k for k in summary['kernels'] if k['snapshot']+'.npz' == name):
                    law = item['law']
                    if not item['accepted']:
                        assert item['failure'] and law+'_post' not in data.files
                        continue
                    post, impulse = data[law+'_post'], data[law+'_impulse'].reshape(-1, 3)
                    assert np.isfinite(post).all() and np.isfinite(impulse).all()
                    after = point_velocities(post, centers, pairs, points, normals)
                    body_impulse = np.zeros_like(centers); torque = np.zeros(len(centers))
                    for (a, b), point, normal, p in zip(pairs, points, normals, impulse):
                        force = normal*p[0]+np.array([-normal[1], normal[0]])*p[1]
                        assert p[2] == 0
                        for index, sign in ((a, 1), (b, -1)):
                            r = point-centers[index]; body_impulse[index] += sign*force
                            torque[index] += sign*(r[0]*force[1]-r[1]*force[0])
                    generalized = np.column_stack((body_impulse, torque)).ravel()
                    np.testing.assert_allclose(post, v+inverse*generalized, atol=1e-9, rtol=1e-10)
                    pn, pt = impulse[:, 0], impulse[:, 1]; tol = plan['contact_tolerance']
                    assert np.min(pn) >= -tol and np.min(after[:, 0]) >= -tol
                    assert np.max(np.abs(after[pn > tol, 0]), initial=0) <= tol
                    if law == 'normal': assert np.max(np.abs(pt)) == 0
                    else:
                        assert np.max(np.abs(pt)-plan['friction']*pn) <= tol
                        slip = np.abs(after[:, 1]) > tol
                        assert np.max(np.abs(pt[slip]+plan['friction']*pn[slip]*np.sign(after[slip, 1])), initial=0) <= tol
                    contact_energy = np.sum(.5*(before+after)*impulse[:, :2])
                    dynamic = slice(0, -3)
                    energy_change = .5*np.sum((post[dynamic]**2-v[dynamic]**2)/inverse[dynamic])
                    work = -v[-3:]@generalized[-3:]
                    np.testing.assert_allclose(contact_energy, energy_change-work, atol=1e-8, rtol=1e-9)
                    assert contact_energy <= 1e-8*max(1., abs(energy_change), abs(work))
                    assert item['normal_tangent_cross_nnz'] > 0
                    assert len(item['timings']) == plan['repeats']
                    assert item['median_total_s'] == statistics.median(t['total_s'] for t in item['timings'])
    retained = sum(r['accepted'] for r in summary['records'])
    print(f'Random shapes: 80 native attempts, {retained} retained histories, 12 physical contact geometries, '
          f'{sum(k["accepted"] for k in summary["kernels"])}/24 accepted solves; '
          f'{qualified_count}/8 references qualified. All archived results audit.')


if __name__ == '__main__': audit()
