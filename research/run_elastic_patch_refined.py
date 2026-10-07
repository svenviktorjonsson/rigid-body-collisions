"""Execute a preregistered elastic-contact study; retain every attempted run."""
import argparse
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import subprocess
import time
import zipfile
import numpy as np
import scipy
from research.elastic_patch import Material, Plane, simulate

SOURCES=['research/elastic_patch.py','research/run_elastic_patch_refined.py',
         'tests/test_elastic_patch.py','research/elastic-patch-refined/plan.json']


def write_sources(root, destination):
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip()
    hashes={}
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in SOURCES:
            committed=subprocess.check_output(['git','show',sha+':'+path],cwd=root)
            current=(root/path).read_bytes()
            if current!=committed: raise RuntimeError('uncommitted execution-source change: '+path)
            hashes[path]=hashlib.sha256(current).hexdigest()
            info=zipfile.ZipInfo(path,date_time=(2026,1,1,0,0,0)); info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,current)
    return sha,hashes


def metrics(result):
    v=result['states'][-1,3:6]; omega=result['states'][-1,6:9]
    return dict(final_velocity_m_s=v.tolist(),final_omega_rad_s=omega.tolist(),
                final_linear_impulse_N_s=result['linear_impulse_N_s'][-1].tolist(),
                final_couple_impulse_N_m_s=result['couple_impulse_N_m_s'][-1].tolist(),
                max_energy_residual_J=float(np.max(abs(result['energy_residual_J']))),
                initial_energy_J=float(result['kinetic_J'][0]+result['potential_J'][0]+result['stored_J'][0]),
                dissipated_J=float(result['dissipated_J'][-1]),
                max_history_stored_J=float(np.max(result['history_stored_J'])),
                max_yield_excess_N=float(result['max_yield_excess_N']),
                separation_loss_J=float(sum(e['separation_loss_J'] for e in result['events'])),
                bounces=sum(e['kind']=='lift_off' for e in result['events']),
                rhs_evaluations=result['rhs_evaluations'],events=result['events'])


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',default='research/elastic-patch-refined')
    args=parser.parse_args(); root=Path(__file__).resolve().parents[1]; output=root/args.output
    plan=json.loads((root/'research/elastic-patch-refined/plan.json').read_text())
    output.mkdir(parents=True,exist_ok=True)
    sha,hashes=write_sources(root,output/'execution-source.zip')
    summary=dict(schema=1,execution_source_commit=sha,source_sha256=hashes,
                 scipy_version=scipy.__version__,numpy_version=np.__version__,scope=plan['scope'],cases=[])
    gates=plan['gates']
    with zipfile.ZipFile(output/'traces.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for case in plan['cases']:
            row=dict(name=case['name'],runs=[],qualified=False); results=[]
            simulation=dict(case['simulation'])
            for key in ['expected_velocity','expected_omega','expected_dissipation','minimum_bounces',
                        'require_spin_reversal','step_scale','retain_rejection']:
                simulation.pop(key,None)
            if 'planes' in simulation: simulation['planes']=tuple(Plane(**p) for p in simulation['planes'])
            material=Material(**case['material'])
            for level in plan['levels']:
                settings=dict(level); settings.pop('name')
                settings['max_step']*=case['simulation'].get('step_scale',1.)
                t0=time.perf_counter(); attempt=dict(level=level['name'],settings=settings)
                try:
                    result=simulate(material,**simulation,**settings)
                    attempt.update(status='completed',metrics=metrics(result))
                    buffer=io.BytesIO(); arrays={k:v for k,v in result.items() if isinstance(v,np.ndarray)}
                    np.savez_compressed(buffer,**arrays)
                    archive.writestr(case['name']+'/'+level['name']+'.npz',buffer.getvalue())
                    results.append(result)
                except (RuntimeError,ValueError) as exception:
                    attempt.update(status='rejected',reason=str(exception)); results.append(None)
                attempt['elapsed_s']=time.perf_counter()-t0; row['runs'].append(attempt)
                print(case['name'],level['name'],attempt['status'],round(attempt['elapsed_s'],3),flush=True)
            checks={}
            if all(r is not None for r in results):
                checks['energy']=all(r['metrics']['max_energy_residual_J']<=gates['energy_absolute_J']+
                                    gates['energy_relative']*abs(r['metrics']['initial_energy_J']) for r in row['runs'])
                checks['yield']=all(r['metrics']['max_yield_excess_N']<=gates['yield_excess_N'] for r in row['runs'])
                checks['nonnegative_dissipation']=all(np.min(r['dissipated_J'])>=-gates['energy_absolute_J'] and
                    np.min(np.diff(r['dissipated_J']))>=-gates['energy_absolute_J'] for r in results)
                edges=[]
                for a,b in zip(results,results[1:]):
                    difference=a['states']-b['states']
                    edge=dict(position_m=float(np.max(np.linalg.norm(difference[:,:3],axis=1))),
                              velocity_m_s=float(np.max(np.linalg.norm(difference[:,3:6],axis=1))),
                              omega_rad_s=float(np.max(np.linalg.norm(difference[:,6:9],axis=1))))
                    edges.append(edge)
                row['refinement_edges']=edges
                checks['refinement']=all(e['position_m']<=gates['position_refinement_m'] and
                                        e['velocity_m_s']<=gates['velocity_refinement_m_s'] and
                                        e['omega_rad_s']<=gates['omega_refinement_rad_s'] for e in edges)
                sim=case['simulation']; fine=results[-1]; met=row['runs'][-1]['metrics']
                if 'expected_velocity' in sim:
                    checks['analytic_velocity']=np.linalg.norm(fine['states'][-1,3:6]-sim['expected_velocity'])<=gates['analytic_velocity_m_s']
                if 'expected_omega' in sim:
                    checks['analytic_omega']=np.linalg.norm(fine['states'][-1,6:9]-sim['expected_omega'])<=gates['analytic_omega_rad_s']
                if 'expected_dissipation' in sim:
                    checks['analytic_dissipation']=abs(met['dissipated_J']-sim['expected_dissipation'])<=1e-5
                if 'require_spin_reversal' in sim: checks['spin_reversal']=fine['states'][-1,8]*sim['omega'][2]<0
                if 'minimum_bounces' in sim:
                    checks['bounces']=met['bounces']>=sim['minimum_bounces']
                    lifts=[e for e in fine['events'] if e['kind']=='lift_off']
                    checks['alternating_bounce_velocities']=all(abs(e['velocity_m_s'][2]-(-1)**i)<1e-5 and
                        abs(e['omega_rad_s'][2]+10*(-1)**i)<1e-4 for i,e in enumerate(lifts))
                checks['separation_loss']=met['separation_loss_J']<=gates['separation_loss_J']
                row['qualified']=all(checks.values())
            row['checks']={k:bool(v) for k,v in checks.items()}
            summary['cases'].append(row)
            (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    summary['qualified_cases']=sum(r['qualified'] for r in summary['cases'])
    summary['completed_histories']=sum(run['status']=='completed' for row in summary['cases'] for run in row['runs'])
    summary['rejected_attempts']=sum(run['status']=='rejected' for row in summary['cases'] for run in row['runs'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')


if __name__=='__main__': main()
