"""Execute the committed random-shape plan; retain failures and full trajectories."""
import hashlib
import io
import json
from pathlib import Path
import platform
import statistics
import subprocess
import time
import zipfile

import numpy as np

from rigid_engine import run, BOX2D_COMMITS
from research.random_shapes import scenes, contact_chain
from research.run_rigid_study import errors, normalized_error, diagnostics
from research.sparse_contact import assemble_sparse, normal_solve, friction_solve

ROOT = Path(__file__).parent/'random-shapes'


def execute():
    plan = json.loads((ROOT/'plan.json').read_text())
    output = ROOT/'results'; output.mkdir(exist_ok=True)
    checkpoint = output/'checkpoints'; checkpoint.mkdir(exist_ok=True)
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    source_paths = ['rigid_engine.py', 'rigid_backend/runner.cpp', 'rigid_backend/compat2.h',
        'research/random_shapes.py', 'research/run_random_shape_study.py', 'research/sparse_contact.py',
        'research/rigid_scenes.py', 'research/container_scenes.py', 'research/run_rigid_study.py',
        'research/random-shapes/plan.json']
    with zipfile.ZipFile(output/'execution-source.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in source_paths:
            z.writestr(path, subprocess.check_output(['git', 'show', f'{commit}:{path}']))
    records = []; qualifications = {}; comparisons = []; kernels = []
    generated = scenes(plan['seeds'])
    (output/'scenes.json').write_text(json.dumps(generated, indent=2)+'\n')

    def trajectory(scene, backend, p, s):
        key = f'{scene["id"]}__{backend}_p{p}_s{s}'
        for _ in range(plan['warmups']): run(scene, backend=backend, primary_steps=p, substeps=s)
        samples = [run(scene, backend=backend, primary_steps=p, substeps=s) for _ in range(plan['repeats'])]
        result = samples[-1]
        (checkpoint/(key+'.json')).write_text(json.dumps(result, separators=(',', ':')))
        timings = [r['engine_and_controller_s'] for r in samples]
        records.append({'trace': key, 'scene': scene['id'], 'backend': backend, 'primary': p, 'solver': s,
            'samples_s': timings, 'median_s': statistics.median(timings), 'diagnostics': diagnostics(scene, result)})
        print(key, flush=True)
        return result, records[-1]

    for scene in generated:
        references = {tuple(mode): trajectory(scene, 'block', *mode)[0] for mode in plan['reference_modes']}
        reference = references[16, 64]
        edges = [{'from': a, 'to': b, 'errors': errors(references[tuple(b)], references[tuple(a)])}
                 for a, b in plan['reference_edges']]
        qualified = all(normalized_error(e['errors'], plan['reference_budget']) <= 1 for e in edges)
        qualifications[scene['id']] = {'qualified': qualified, 'edges': edges}
        for backend, p, s in plan['candidates']:
            candidate, record = trajectory(scene, backend, p, s)
            error = errors(reference, candidate)
            comparisons.append({'trace': record['trace'], 'scene': scene['id'], 'errors': error,
                'reference_qualified': qualified,
                'passed': qualified and normalized_error(error, plan['budget']) <= 1})
    for seed in plan['seeds']:
        for concave in (False, True):
            for count in plan['contact_counts']:
                data = contact_chain(count, seed, concave)
                key = f'chain_{count}_{seed}_{"concave" if concave else "convex"}'
                arrays = {k: data[k] for k in ('centers', 'mass', 'inertia', 'velocity')}
                arrays.update({'bodies': np.array([c[:2] for c in data['contacts']]),
                    'points': np.array([c[2] for c in data['contacts']]),
                    'normals': np.array([c[3] for c in data['contacts']]),
                    'geometry_json': np.array(json.dumps(data['geometry']))})
                for law in ('normal', 'friction'):
                    item = {'snapshot': key, 'law': law, 'count': count, 'accepted': False}
                    timing = []
                    try:
                        for repeat in range(plan['warmups']+plan['repeats']):
                            start = time.perf_counter()
                            system = assemble_sparse(data['centers'], data['mass'], data['inertia'], data['contacts'])
                            assembled = time.perf_counter()
                            if law == 'normal': post, impulse, stats = normal_solve(system, data['velocity'])
                            else: post, impulse, stats = friction_solve(system, data['velocity'], plan['friction'])
                            finished = time.perf_counter()
                            if repeat >= plan['warmups']:
                                timing.append({'assemble_s': assembled-start, 'solve_s': finished-assembled,
                                               'total_s': finished-start})
                        arrays[law+'_post'] = post; arrays[law+'_impulse'] = impulse
                        _, mobility = system.mobility((0, 1))
                        item.update({'accepted': True, 'stats': stats, 'timings': timing,
                            'median_total_s': statistics.median(t['total_s'] for t in timing),
                            'normal_tangent_cross_nnz': mobility[::2, 1::2].nnz})
                    except RuntimeError as error:
                        item['failure'] = str(error)
                    kernels.append(item); print(key, law, item['accepted'], flush=True)
                np.savez_compressed(checkpoint/(key+'.npz'), **arrays)
    with zipfile.ZipFile(output/'traces.zip', 'w', zipfile.ZIP_DEFLATED) as z:
        for path in sorted(checkpoint.iterdir()): z.write(path, path.name)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    summary = {'source_commit': commit, 'scope': plan['scope'], 'machine': platform.platform(),
        'backend_commits': BOX2D_COMMITS, 'plan_sha256': digest(ROOT/'plan.json'),
        'traces_sha256': digest(output/'traces.zip'), 'scenes_sha256': digest(output/'scenes.json'),
        'source_archive_sha256': digest(output/'execution-source.zip'), 'records': records,
        'qualifications': qualifications, 'comparisons': comparisons, 'kernels': kernels}
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__': execute()
