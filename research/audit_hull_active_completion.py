"""Independent authored-frame audit of native principal-frame checkpoints.

No production preparation, energy, containment, error or acceptance function is
imported. Completed trajectories also receive the independent historical full
archive audit. Prefix observations never imply trajectory completion.
"""
import argparse,hashlib,json
from pathlib import Path
import subprocess,zipfile
import numpy as np
from scipy.spatial.transform import Rotation
from research.audit_shared_hulls import body_integrals,rotation,trajectory_metrics,audit as audit_archive

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'research/hull-active-completion'


def sha(raw):return hashlib.sha256(raw).hexdigest()


def authored_progress(scene,progress,dt):
    assert progress['schema']=='native-spatial-progress-v1'
    assert progress['orientation_frame']=='backend principal inertia axes'
    backend=np.asarray(progress['states_backend'],dtype=float);wire=progress['wire_bodies'];times=np.asarray(progress['times'])
    expected=round(scene['duration']/dt);completed=progress['completed_output_frames']
    assert backend.shape==(completed+1,len(scene['bodies']),13) and np.isfinite(backend).all()
    assert 0<=completed<=expected==progress['expected_output_frames']
    assert progress['complete']==(completed==expected)
    np.testing.assert_allclose(times,np.arange(completed+1)*dt,rtol=0,atol=1e-12)
    assert len(wire)==len(scene['bodies'])
    authored=backend.copy();mass=[];inertia=[];certificates=[];native_energy=np.zeros(len(times));g=np.asarray(scene.get('gravity',[0,0,-9.81]))
    for i,body in enumerate(scene['bodies']):
        geometric_mass,center,I=body_integrals(body);m=geometric_mass if body.get('type','dynamic')=='dynamic' else 0.
        mass.append(m);inertia.append(I.tolist());w=wire[i]
        assert w['type']==body.get('type','dynamic') and np.isclose(w['mass'],m,rtol=1e-12,atol=1e-12)
        np.testing.assert_allclose(backend[0,i,:3],body.get('position',[0,0,0]),rtol=0,atol=1e-12)
        np.testing.assert_allclose(backend[0,i,7:10],body.get('velocity',[0,0,0]),rtol=0,atol=1e-12)
        np.testing.assert_allclose(backend[0,i,10:13],body.get('omega',[0,0,0]),rtol=0,atol=1e-12)
        R0=rotation(body.get('orientation',[0,0,0,1]));P0=rotation(backend[0,i,3:7]);Q=R0.T@P0
        np.testing.assert_allclose(Q.T@Q,np.eye(3),rtol=0,atol=1e-12);assert abs(np.linalg.det(Q)-1)<1e-12
        diagonal=np.asarray(w['principal_inertia']);defect=0.
        if m:
            defect=float(np.max(abs(Q@np.diag(diagonal)@Q.T-I)))
            np.testing.assert_allclose(Q@np.diag(diagonal)@Q.T,I,rtol=1e-11,atol=1e-11)
        else:np.testing.assert_array_equal(diagonal,[0,0,0])
        assert len(w['shapes'])==len(body['shapes'])
        for shape,native in zip(body['shapes'],w['shapes']):
            assert shape['kind']==native['kind']
            for key in ('radius','half_extents','vertices'):
                if key in shape:np.testing.assert_array_equal(shape[key],native[key])
            Rs=rotation(shape.get('orientation',[0,0,0,1]));offset=np.asarray(shape.get('center',[0,0,0]))-center
            np.testing.assert_allclose(native['center'],Q.T@offset,rtol=0,atol=1e-12)
            np.testing.assert_allclose(rotation(native['orientation']),Q.T@Rs,rtol=0,atol=1e-12)
        for key,default in [('friction',.5),('restitution',0)]:assert w[key]==body.get(key,default)
        for j in range(len(times)):
            P=rotation(backend[j,i,3:7]);R=P@Q.T;authored[j,i,3:7]=Rotation.from_matrix(R).as_quat()
            if m:
                spin=P.T@backend[j,i,10:13];v=backend[j,i,7:10]
                native_energy[j]+=.5*m*(v@v)+.5*np.sum(diagonal*spin*spin)-m*g@backend[j,i,:3]
        certificates.append(dict(body=i,principal_to_authored_basis=Q.tolist(),independent_full_tensor_defect_kg_m2=defect))
    result=dict(states=authored.tolist(),times=times.tolist(),mass=mass,inertia_body_kg_m2=inertia,
                boundary_work_J=progress['boundary_work_J'],max_container_surface_excess_m=progress['max_container_surface_excess_m'])
    metrics=trajectory_metrics(scene,result)
    direct=float(native_energy[-1]-native_energy[0]-progress['boundary_work_J'])
    np.testing.assert_allclose(metrics['energy_change_minus_boundary_work_J'],direct,rtol=1e-10,atol=1e-10)
    raw_qerror=float(np.max(abs(np.linalg.norm(backend[:,:,3:7],axis=2)-1)))
    # Normalizing a quaternion for geometry must not hide a malformed raw state.
    metrics['quaternion_norm_error']=max(metrics['quaternion_norm_error'],raw_qerror)
    return result,metrics,certificates


def audit_progress(study,source,require_all=False,output_path=None):
    directory=study/'results';plan=json.loads((study/'plan.json').read_text());scenes=json.loads((directory/'scenes.json').read_text())
    provenance=json.loads((directory/'checkpoints/provenance.json').read_text());assert provenance['execution_source_commit']==source
    assert provenance['plan_sha256']==sha((study/'plan.json').read_bytes())
    with zipfile.ZipFile(directory/'execution-source.zip') as archive:
        assert set(archive.namelist())==set(provenance['source_hashes'])
        for name,digest in provenance['source_hashes'].items():
            raw=archive.read(name);assert sha(raw)==digest
            assert raw==subprocess.check_output(['git','show',f'{source}:{name}'],cwd=ROOT)
    records=[]
    for config in plan['scenes']:
        name=config['id'];scene=scenes[name]['scene']
        for i,fraction in enumerate(config['fractions']):
            lane=f'reference_{i}';path=directory/'progress'/name/(lane+'.json')
            if not path.exists():
                assert not require_all;continue
            raw=path.read_bytes();progress=json.loads(raw);result,metrics,certificates=authored_progress(scene,progress,plan['dt_s'])
            physical=all(np.isfinite(metrics[key]) and metrics[key]<=limit for key,limit in plan['physical_gates'].items())
            contact=bool(np.isfinite(progress['coulomb_residual_max_m_s']) and progress['coulomb_residual_max_m_s']<=plan['common']['contact_tolerance_m_s'])
            record=dict(scene=name,lane=lane,fraction=fraction,path=str(path.relative_to(study)),sha256=sha(raw),complete=progress['complete'],completed_output_frames=progress['completed_output_frames'],
                expected_output_frames=progress['expected_output_frames'],collision_updates=progress['collision_updates'],orientation_frame=progress['orientation_frame'],
                output_conversion='R_authored = R_backend Q transpose; Q = R_initial_authored transpose R_initial_backend',basis_certificates=certificates,
                diagnostics=metrics,contact_residual_max_m_s=progress['coulomb_residual_max_m_s'],physical_gates_passed=bool(physical),strict_contact_gate_passed=contact,
                prefix_observation_only=not progress['complete'],trajectory_qualified=False)
            checkpoint=directory/'checkpoints'/name/(lane+'.json')
            if checkpoint.exists():
                completed=json.loads(checkpoint.read_text());same_snapshot=completed['native_progress']['sha256']==sha(raw)
                if require_all:assert same_snapshot
                if not same_snapshot:
                    # A live writer can finish after this auditor reads an
                    # earlier atomic prefix. Final audit requires exact hashes.
                    record['checkpoint_comparison']='not available for this observed earlier snapshot'
                elif 'rejected' not in completed:
                    assert progress['complete'];a=np.asarray(result['states']);b=np.asarray(completed['states'])
                    np.testing.assert_allclose(a[:,:,:3],b[:,:,:3],rtol=0,atol=1e-12);np.testing.assert_allclose(a[:,:,7:13],b[:,:,7:13],rtol=0,atol=1e-12)
                    np.testing.assert_allclose(abs(np.sum(a[:,:,3:7]*b[:,:,3:7],axis=-1)),1,rtol=0,atol=1e-12)
                    assert completed['boundary_work_J']==progress['boundary_work_J'] and completed['coulomb_residual_max_m_s']==progress['coulomb_residual_max_m_s']
                    record['completed_authored_trace_agrees']=True
            records.append(record)
    output=dict(schema='independent-native-hull-progress-audit-v1',execution_source_commit=source,plan_sha256=provenance['plan_sha256'],snapshot_count=len(records),
                all_observed_snapshots_integrity_passed=True,cases=records,scope='Snapshots of output-frame prefixes. Only completed final histories and both prescribed refinement edges can qualify a trajectory; partial progress cannot.')
    (Path(output_path) if output_path is not None else study/'independent-progress-audit.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print('Native principal/authored-frame progress audit PASS:',len(records),'snapshots;',sum(r['complete'] for r in records),'complete;',sum(not r['physical_gates_passed'] for r in records),'physical-gate failures retained')
    return output


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=STUDY);p.add_argument('--source-commit',required=True);p.add_argument('--progress-only',action='store_true');args=p.parse_args()
    source=subprocess.check_output(['git','rev-parse',args.source_commit],cwd=ROOT,text=True).strip()
    if not args.progress_only:audit_archive(args.directory.resolve(),source)
    audit_progress(args.directory.resolve(),source,require_all=not args.progress_only)
