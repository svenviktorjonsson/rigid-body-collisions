"""Reproducible mechanical controls, old-input equivalence and native fixtures."""
import argparse,hashlib,importlib.util,json,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from supported_contact import Resistance,advance_planar,advance_spatial
from spatial_engine import prepare
from research.spatial_scenes import wall_impact,container


INPUT_FIELDS=['mass_kg','inertia_kg_m2','radius_m','normal_load_N','drive_force_N','velocity_m_s','omega_rad_s','spin_rad_s','duration_s','mu_s','mu_d','mu_r','rolling_length_m','mu_n','spin_length_m']
OUTPUT_FIELDS=['velocity_m_s','omega_rad_s','spin_rad_s','distance_m','tangent_impulse_Ns','independent_rolling_impulse_Nms','independent_spin_impulse_Nms','sliding_loss_J','rolling_loss_J','spin_loss_J','energy_residual_J']


def run(out):
    rng=np.random.default_rng(20261007);rows=[];errors=[];energy=[];phases=[]
    for k in range(400):
        m=10**rng.uniform(-3,2);R=10**rng.uniform(-3,0);I=rng.uniform(.3,.7)*m*R*R;N=m*rng.uniform(1,15)
        mus=rng.uniform(.05,.8);mud=mus*rng.uniform(.1,1)
        law=Resistance(mus,mud,rng.uniform(0,.1),R*rng.uniform(.1,1),rng.uniform(0,.1),R*rng.uniform(.1,1))
        v=rng.uniform(-5,5);w=rng.uniform(-5,5)/R;spin=rng.uniform(-5,5)/R;duration=10**rng.uniform(-2,1);drive=rng.uniform(-.2,.2)*N
        args=dict(mass_kg=m,inertia_kg_m2=I,radius_m=R,normal_load_N=N,drive_force_N=drive,material=law)
        whole=advance_planar(**args,velocity_m_s=v,omega_rad_s=w,spin_rad_s=spin,duration_s=duration)
        first=advance_planar(**args,velocity_m_s=v,omega_rad_s=w,spin_rad_s=spin,duration_s=.37*duration)
        second=advance_planar(**args,velocity_m_s=first['velocity_m_s'],omega_rad_s=first['omega_rad_s'],spin_rad_s=first['spin_rad_s'],duration_s=.63*duration)
        velocity_scale=max(abs(v),R*abs(w),R*abs(spin),abs(drive/m)*duration,1e-30)
        error=max(abs(whole['velocity_m_s']-second['velocity_m_s']),R*abs(whole['omega_rad_s']-second['omega_rad_s']),R*abs(whole['spin_rad_s']-second['spin_rad_s']))/velocity_scale
        assert error<1e-10
        loss=sum(whole[name] for name in ['sliding_loss_J','rolling_loss_J','spin_loss_J'])
        energy_error=abs(whole['energy_residual_J'])/max(whole['initial_kinetic_J'],whole['final_kinetic_J'],loss,abs(whole['external_work_J']),1e-300)
        errors.append(error);energy.append(energy_error);phases.append(len(whole['events']))
        rows.append(dict(case_index=k,inputs=[m,I,R,N,drive,v,w,spin,duration,mus,mud,law.mu_r,law.rolling_length_m,law.mu_n,law.spin_length_m],outputs=[whole[name] for name in OUTPUT_FIELDS]))
    (out/'controls.txt').write_text(str(len(rows))+'\n'+'\n'.join(' '.join(format(x,'.17g') for x in row['inputs']+row['outputs']) for row in rows)+'\n')
    rotation_errors=[];moment_errors=[];span_errors=[]
    for k in range(100):
        Q=Rotation.random(random_state=rng).as_matrix();n=Q[:,2];d=Q[:,0];axis=np.cross(n,d)
        v=rng.uniform(-3,3);w=v/.1;spin=rng.uniform(-10,10);U=rng.uniform(-1,1)*d+rng.uniform(-1,1)*axis
        law=Resistance(.5,.3,.02,.1,.03,.02);h=rng.uniform(.01,20)
        args=dict(mass_kg=1.,inertia_kg_m2=.004,radius_m=.1,normal_load_N=9.81,drive_force_N=0.,duration_s=h,material=law)
        planar=advance_planar(**args,velocity_m_s=v,omega_rad_s=w,spin_rad_s=spin)
        spatial=advance_spatial(**args,normal=n,direction=d,velocity=U+v*d,omega=w*axis+spin*n,plane_velocity=U)
        rotation_errors.append(float(np.linalg.norm(np.array(spatial['velocity'])-U-planar['velocity_m_s']*d)))
        delta=np.array(spatial['body_angular_change']);moment_errors.append(float(np.linalg.norm(delta-.004*(np.array(spatial['omega'])-w*axis-spin*n))))
        span_errors.extend(frame['angular_span_residual_Nm'] for frame in spatial['directional_branch_frames'])
    # Import baseline adapter from an immutable pre-fix revision, not current code.
    baseline=subprocess.check_output(['git','show','512a1b2:spatial_engine.py'],cwd=ROOT,text=True)
    old_path=out/'baseline-spatial-engine.py';old_path.write_text(baseline)
    spec=importlib.util.spec_from_file_location('baseline_spatial_engine',old_path);old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
    scenes=[wall_impact(),container(side=2,shape='sphere')[0],container(side=2,shape='hull')[0]]
    encode=lambda x:json.dumps(x,sort_keys=True,default=lambda a:a.tolist())
    for scene in scenes:assert encode(old.prepare(scene))==encode(prepare(scene))
    result=dict(random_cases=len(rows),random_passes=len(rows),max_scaled_step_composition_error=max(errors),max_relative_energy_residual=max(energy),max_branch_intervals=max(phases),
                rotated_spatial_cases=100,max_rotated_velocity_error_m_s=max(rotation_errors),max_rotated_angular_impulse_error_Nms=max(moment_errors),max_directional_span_error_Nm=max(span_errors),
                unchanged_default_scene_preparations=len(scenes),baseline_revision='512a1b2',input_fields=INPUT_FIELDS,output_fields=OUTPUT_FIELDS,
                scope='Exact supported branch and native input mechanics. No new empirical accuracy qualification or interacting group simulation.',
                sources={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['supported_contact.py','spatial_engine.py','material_profiles.py']})
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    print(json.dumps(run(out),indent=2))
