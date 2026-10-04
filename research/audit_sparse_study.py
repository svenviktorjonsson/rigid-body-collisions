"""Independently audit explicit row mechanics and dense general-contact algebra."""
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from research.contact_solver import assemble_planar


def audit():
    root = Path(__file__).parent / 'sparse-islands'
    report = json.loads((root / 'results/summary.json').read_text())
    plan = json.loads((root / 'plan.json').read_text())
    archive = root / 'results/states.zip'
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == report['states_zip_sha256']
    sources = root / 'results/execution-source.zip'
    assert hashlib.sha256(sources.read_bytes()).hexdigest() == report['execution_source_zip_sha256']
    with zipfile.ZipFile(sources) as z:
        for name, digest in report['source_sha256'].items():
            assert hashlib.sha256(z.read(Path(name).name)).hexdigest() == digest, name
        assert z.read('plan.json') == (root/'plan.json').read_bytes()
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        states = {Path(name).stem: dict(np.load(io.BytesIO(z.read(name)))) for name in z.namelist()}
    assert len(states) == len(report['records']) == 139
    for record in report['records']:
        kind = record['kind']
        if kind == 'stress': key = f"stress_{record['index']}"
        elif kind == 'normal': key = f"normal_{record['count']}_{record['method']}"
        else: key = f"friction_{record['count']}_{record['friction']}_{record['strategy']}"
        data = states[key]
        if kind != 'stress' and record['passed']:
            assert len(record['samples']) == plan['repeats']
            for sample in record['samples']:
                assert all(np.isfinite(v) and v > 0 for v in sample.values())
                assert abs(sample['assembly_s']+sample['solve_s']-sample['total_s']) < 1e-12
            for metric in ('assembly_s','solve_s','total_s'):
                assert np.isclose(np.median([s[metric] for s in record['samples']]), record['median_'+metric])
        if not record['passed']:
            assert kind == 'stress' and 'error' in record and 'post' not in data
            continue
        post = data['post']; impulse = data['impulse']
        assert np.isfinite(post).all() and np.isfinite(impulse).all()
        if kind in ('normal','friction'):
            n = record['count']; body = post[:-3].reshape(n,3)
            pn, pt = impulse[::3], impulse[1::3]
            np.testing.assert_allclose(body[:,0], pn[:-1]-pn[1:], atol=1e-8)
            np.testing.assert_allclose(body[:,0], 1, atol=1e-8)
            wn = np.r_[body[0,0]-1, np.diff(body[:,0]), 1-body[-1,0]]
            assert np.min(pn) >= -1e-8 and np.min(wn) >= -1e-8
            assert np.max(np.abs(wn[pn > 1e-8]), initial=0) <= 1e-8
            if kind == 'normal':
                np.testing.assert_allclose(body[:,1:], 0, atol=1e-8)
                np.testing.assert_allclose(impulse[1::3], 0, atol=1e-8)
                assert abs(pn[0]-pn[-1]-n) <= 1e-5
                assert np.isclose(record['dissipation_J'], n/2, atol=1e-5)
                base = 200*n+188
                if record['method']=='dense_optimizer': storage = 152*(n+1)**2
                elif record['method']=='dense_active' or (record['method']=='auto_active' and n+1<=128): storage = base+8*(n+1)**2
                else: storage=base
                assert record['explicit_operator_bytes']==storage
            else:
                initial = data['initial'][:-3].reshape(n,3)
                np.testing.assert_allclose(body[:,1], initial[:,1]+pt[:-1]-pt[1:], atol=1e-8)
                np.testing.assert_allclose(body[:,2], -.1*(pt[:-1]+pt[1:])/.005, atol=1e-8)
                wt = np.r_[body[0,1]-.1*body[0,2]-.2,
                    np.diff(body[:,1])-.1*(body[1:,2]+body[:-1,2]), .2-body[-1,1]-.1*body[-1,2]]
                mu = record['friction']
                assert np.max(np.abs(pt)-mu*pn, initial=0)<=1e-8
                slip = np.abs(wt)>1e-8
                assert np.max(np.abs(pt[slip]+mu*pn[slip]*np.sign(wt[slip])), initial=0)<=1e-8
                work = pn[0]-pn[-1]+.2*(pt[0]-pt[-1])
                kinetic_change = .5*np.sum(body[:,:2]**2-initial[:,:2]**2)+.5*.005*np.sum(body[:,2]**2)
                assert np.isclose(kinetic_change-work,record['stats']['contact_energy_change_minus_boundary_work_J'],rtol=1e-8,atol=1e-7)
                assert kinetic_change<=work+1e-7
        else:
            contacts = [(int(a),int(b),p,n) for (a,b),p,n in zip(data['bodies'],data['points'],data['normals'])]
            inverse, G, K = assemble_planar(data['centers'],[1,2,3,np.inf],[.5,1,1,np.inf],contacts)
            np.testing.assert_allclose(post, data['initial']+inverse @ G.T @ impulse, atol=1e-8)
            w = G @ post; pn,pt=impulse[::3],impulse[1::3]; wn,wt=w[::3],w[1::3]
            assert np.min(pn)>=-1e-8 and np.min(wn)>=-1e-8
            assert np.max(np.abs(wn[pn>1e-8]),initial=0)<=1e-8
            mu=plan['held_out_stress']['friction']
            assert np.max(np.abs(pt)-mu*pn,initial=0)<=1e-8
            slip=np.abs(wt)>1e-8
            assert np.max(np.abs(pt[slip]+mu*pn[slip]*np.sign(wt[slip])),initial=0)<=1e-8
            H=1/inverse.diagonal()[:-3]
            energy=.5*np.sum(H*(post[:-3]**2-data['initial'][:-3]**2))
            assert energy<=1e-7
    for n in plan['friction_counts']:
        for mu in plan['friction_coefficients']:
            a,b=states[f'friction_{n}_{mu}_auto'],states[f'friction_{n}_{mu}_general']
            np.testing.assert_allclose(a['post'],b['post'],atol=1e-8)
            np.testing.assert_allclose(a['impulse'],b['impulse'],atol=1e-8)
    print('Audited 139 snapshots: exact row mechanics, Coulomb gates, external work, timings, storage and retained failures.')


if __name__ == '__main__': audit()
