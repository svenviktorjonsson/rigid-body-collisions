"""Retain an independent whole-step split-rotation energy counterexample."""
import hashlib,json,subprocess
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from spatial_engine import run,BINARY


def kinetic(result):
    states=np.asarray(result['states']);energy=np.zeros(len(states))
    for i,mass in enumerate(result['mass']):
        if mass==0:continue
        frame=Rotation.from_quat(states[:,i,3:7]).as_matrix()
        inertia=frame@np.asarray(result['inertia_body_kg_m2'][i])@frame.transpose(0,2,1)
        energy+=.5*mass*np.einsum('ti,ti->t',states[:,i,7:10],states[:,i,7:10])
        energy+=.5*np.einsum('ti,tij,tj->t',states[:,i,10:13],inertia,states[:,i,10:13])
    return energy


def main():
    ext=np.array([.2,.1,.05]);rotation=Rotation.from_rotvec([0,.4,0])
    lowest=float(abs(rotation.as_matrix()[2])@ext)
    scene=dict(duration=1e-5,gravity=[0,0,0],bodies=[
        dict(type='static',position=[0,0,-.05],friction=0,shapes=[dict(kind='box',half_extents=[10,10,.05])]),
        dict(position=[0,0,lowest-.005],orientation=rotation.as_quat().tolist(),omega=[0,0,10],friction=0,
             shapes=[dict(kind='box',half_extents=ext.tolist(),density=1000)])])
    settings=dict(dt=1e-5,primary_steps=1,iterations=4096,solver='coulomb',travel_fraction=0,kinematic_contact_phase='start')
    results={mode:run(scene,position_stabilization=mode,**settings) for mode in ('split','velocity_only')}
    deltas={mode:float(np.diff(kinetic(result))[0]) for mode,result in results.items()}
    assert deltas['split']>1e-3 and abs(deltas['velocity_only'])<1e-7
    assert results['split']['coulomb_passive_change_max_J']==0
    package=dict(schema='split-pose-energy-counterexample-v1',scene=scene,settings=settings,results=results,
                 kinetic_delta_J=deltas,binary_sha256=hashlib.sha256(BINARY.read_bytes()).hexdigest(),
                 source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                 source_hashes={f:hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in
                                ('spatial_backend/coulomb.h','spatial_backend/shared_contact.h','spatial_backend/runner.cpp','spatial_engine.py')},
                 limitation='Initial overlap is 5 mm. This counterexample isolates numerical pose repair, not a spontaneous energy gain in a zero-overlap physical collision. The physical impulse phase remains passive.')
    Path('research/completion-review/split-energy-counterexample.json').write_text(json.dumps(package,indent=2)+'\n')
    print(json.dumps(deltas))


if __name__=='__main__':main()
