"""Run with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.run_sparse_study."""
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time
import zipfile

import numpy as np
import scipy
from research.container_scenes import packed_row_contact_system
from research.contact_solver import inelastic_normal_solve
from research.sparse_contact import assemble_sparse, friction_solve, normal_solve

ROOT = Path(__file__).parent / 'sparse-islands'


def matrix_bytes(system):
    total = system.inverse_mass.nbytes
    seen = set()
    for matrix in [system.contact_map] + [x for pair in system._mobility.values() for x in pair]:
        if id(matrix) in seen: continue
        seen.add(id(matrix)); total += matrix.data.nbytes + matrix.indices.nbytes + matrix.indptr.nbytes
    return total


def study():
    plan = json.loads((ROOT / 'plan.json').read_text())
    if os.environ.get('OPENBLAS_NUM_THREADS') != '1' or os.environ.get('OMP_NUM_THREADS') != '1':
        raise SystemExit('Start with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 for declared timings')
    output = ROOT / 'results'; output.mkdir(exist_ok=True)
    checkpoints = output / 'checkpoints'; checkpoints.mkdir(exist_ok=True)
    records = []; rng = np.random.default_rng(plan['interleave_seed'])
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    inputs = (ROOT / 'plan.json', Path(__file__), Path(__file__).parent / 'sparse_contact.py', Path(__file__).parent / 'container_scenes.py', Path(__file__).parent / 'contact_solver.py')
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs}
    source_archive = output / 'execution-source.zip'
    with zipfile.ZipFile(source_archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in inputs: z.write(p, p.name)

    def checkpoint(key, record, **arrays):
        records.append(record)
        snapshot = checkpoints / (key + '.npz')
        pending_snapshot = checkpoints / (key + '.tmp.npz')
        np.savez_compressed(pending_snapshot, **arrays)
        pending_snapshot.replace(snapshot)
        temporary = output / 'partial.tmp'
        temporary.write_text(json.dumps({'source_commit': source, 'source_sha256': hashes, 'records': records}, indent=2))
        temporary.replace(output / 'partial.json')
        print(key, record.get('passed'), record.get('median_total_s'), record.get('error', ''), flush=True)

    def normal_call(n, method):
        t0 = time.perf_counter()
        if method == 'dense_optimizer':
            inverse, G, v = packed_row_contact_system(n)
            assembly = time.perf_counter()-t0; t1 = time.perf_counter()
            post, p, stats = inelastic_normal_solve(inverse, G, v)
            storage = inverse.nbytes + G.nbytes + 8*(n+1)**2
        else:
            system, v = packed_row_contact_system(n, sparse=True)
            system.mobility((0,))
            assembly = time.perf_counter()-t0; t1 = time.perf_counter()
            post, p, stats = normal_solve(system, v, strategy={'dense_active':'dense', 'sparse_active':'sparse', 'auto_active':'auto'}[method])
            storage = matrix_bytes(system) + (8*(n+1)**2 if method == 'dense_active' or (method == 'auto_active' and n+1 <= 128) else 0)
        solved = time.perf_counter()-t1
        return post, p, stats, {'assembly_s': assembly, 'solve_s': solved, 'total_s': assembly+solved}, storage

    for n in plan['normal_counts']:
        methods = [m for m in plan['normal_methods'] if not (m == 'dense_optimizer' and n > plan['dense_optimizer_max_count']) and not (m == 'dense_active' and n > plan['dense_active_max_count'])]
        samples = {m: [] for m in methods}; last = {}; failed = {}
        for m in methods:
            try: normal_call(n, m)
            except RuntimeError as e: failed[m] = str(e)
        for _ in range(plan['repeats']):
            for method in rng.permutation(methods):
                if method in failed: continue
                try:
                    last[method] = normal_call(n, method); samples[method].append(last[method][3])
                except RuntimeError as e: failed[method] = str(e)
        for method in methods:
            record = {'kind':'normal', 'count':n, 'method':method, 'samples':samples[method], 'passed':False}
            if method in failed:
                record['error'] = failed[method]; checkpoint(f'normal_{n}_{method}', record)
                continue
            post, p, stats, _, storage = last[method]
            stats.pop('normal_velocity', None)
            error = float(np.max(np.abs(post[:-3].reshape(n, 3) - [1,0,0])))
            reaction = float(p[0]-p[-3]); energy = .5*float(np.sum(post[:-3:3]**2))
            record.update({'max_velocity_error_m_s':error, 'wall_impulse_kg_m_s':reaction,
                'actuator_work_J':reaction, 'kinetic_J':energy, 'dissipation_J':reaction-energy,
                'passed':error <= plan['normal_velocity_budget_m_s'] and abs(reaction-n) <= plan['normal_impulse_budget_kg_m_s'],
                'explicit_operator_bytes':int(storage), 'stats':stats,
                **{'median_'+k:statistics.median(s[k] for s in samples[method]) for k in ('assembly_s','solve_s','total_s')}})
            checkpoint(f'normal_{n}_{method}', record, post=post, impulse=p)

    def friction_call(n, mu, strategy):
        t0 = time.perf_counter(); system, v = packed_row_contact_system(n, sparse=True)
        v[1:-3:3] = 2*np.sin(np.arange(n)*.37); v[-2] = .2
        system.mobility((0, 1)); system.mobility((0,))
        assembly = time.perf_counter()-t0; t1 = time.perf_counter()
        post, p, stats = friction_solve(system, v, mu, strategy=strategy)
        solved = time.perf_counter()-t1
        return v, post, p, stats, {'assembly_s':assembly,'solve_s':solved,'total_s':assembly+solved}, matrix_bytes(system)

    for n in plan['friction_counts']:
        for mu in plan['friction_coefficients']:
            samples = {s:[] for s in plan['friction_methods']}; last = {}; failed = {}
            for strategy in samples:
                try: friction_call(n, mu, strategy)
                except RuntimeError as e: failed[strategy] = str(e)
            for _ in range(plan['repeats']):
                for strategy in rng.permutation(plan['friction_methods']):
                    if strategy in failed: continue
                    try:
                        last[strategy] = friction_call(n, mu, strategy); samples[strategy].append(last[strategy][4])
                    except RuntimeError as e: failed[strategy] = str(e)
            for strategy in samples:
                record = {'kind':'friction', 'count':n, 'friction':mu, 'strategy':strategy, 'passed':False, 'samples':samples[strategy]}
                key = f'friction_{n}_{mu}_{strategy}'
                if strategy in failed: record['error']=failed[strategy]; checkpoint(key, record); continue
                v, post, p, stats, _, storage = last[strategy]
                error = float(np.max(np.abs(post[:-3:3]-1)))
                record.update({'passed':error <= plan['normal_velocity_budget_m_s'], 'stats':stats,
                    'max_normal_velocity_error_m_s':error, 'explicit_operator_bytes':int(storage),
                    **{'median_'+k:statistics.median(s[k] for s in samples[strategy]) for k in ('assembly_s','solve_s','total_s')}})
                checkpoint(key, record, initial=v, post=post, impulse=p)

    stress = plan['held_out_stress']; rng = np.random.default_rng(stress['seed'])
    for i in range(stress['count']):
        centers = rng.normal(size=(4,2))
        contacts = [(int(rng.integers(3)),3,rng.normal(size=2),rng.normal(size=2)) for _ in range(7)]
        v = rng.normal(size=12); v[-3:] = 0
        system = assemble_sparse(centers,[1,2,3,np.inf],[.5,1,1,np.inf],contacts)
        t0 = time.perf_counter(); record = {'kind':'stress','index':i,'passed':False}
        arrays = {'centers':centers, 'bodies':np.array([c[:2] for c in contacts]),
                  'points':np.array([c[2] for c in contacts]),'normals':np.array([c[3] for c in contacts]),'initial':v}
        try:
            post, p, stats = friction_solve(system,v,stress['friction'])
            record.update({'passed':True,'stats':stats}); arrays.update(post=post,impulse=p)
        except RuntimeError as e: record['error']=str(e)
        record['solve_s']=time.perf_counter()-t0
        checkpoint(f'stress_{i}',record,**arrays)

    archive = output / 'states.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(checkpoints.glob('*.npz')): z.write(p,p.name)
    summary = {'schema_version':1,'scope':plan['scope'],'source_commit':source,'source_sha256':hashes,
        'machine':platform.platform(),'numpy':np.__version__,'scipy':scipy.__version__,
        'thread_environment':{k:os.environ.get(k) for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS')},
        'records':records,'states_zip_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'execution_source_zip_sha256':hashlib.sha256(source_archive.read_bytes()).hexdigest()}
    (output / 'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__ == '__main__': study()
