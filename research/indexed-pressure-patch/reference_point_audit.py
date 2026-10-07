"""Check contact-origin dependence before interpreting directional residuals."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from audit import footprint
from model import Patch, evaluate, directional_residual


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(7102702);worst=0.;rows=[]
    for case in range(100):
        patch=footprint(['interval','hertz','ellipse','irregular'][case%4])
        d=np.r_[rng.uniform(-.001,.003),rng.uniform(-.1,.1,2)]
        u,w=rng.normal(size=(2,3));shift=np.r_[rng.normal(size=2)*.01,0.]
        K,C,mu=10000.,20.,.4
        before=evaluate(patch,d,u,w,K,C,mu)
        moved=Patch(patch.points-shift[:2],patch.weights)
        moved_d=d.copy();moved_d[0]+=shift[1]*d[1]-shift[0]*d[2]
        moved_u=u+np.cross(w,shift)
        after=evaluate(moved,moved_d,moved_u,w,K,C,mu)
        target=before['moment']-np.cross(shift,before['force'])
        arm=rng.normal(size=3)
        body_before=np.cross(arm,before['force'])+before['moment']
        body_after=np.cross(arm+shift,after['force'])+after['moment']
        scale=max(1,np.linalg.norm(before['force']),np.linalg.norm(before['moment']),abs(before['power']))
        error=max(np.linalg.norm(after['force']-before['force']),np.linalg.norm(after['moment']-target),
                  np.linalg.norm(body_after-body_before),abs(after['power']-before['power']),
                  abs(after['stored_energy']-before['stored_energy']))/scale
        worst=max(worst,float(error));assert error<1e-12
        for ell in [.001,1.,1000.]:
            combined_velocity=np.r_[u,ell*w];combined_force=np.r_[before['force'],before['moment']/ell]
            assert abs(combined_force@combined_velocity-before['power'])<1e-12*scale
    # Center of normal pressure is a physical alternative origin, not a t redefinition.
    for kind in ['hertz','ellipse','irregular']:
        u=np.array([.1,.2,-.03]);w=np.array([3.,5.,40.])
        r=evaluate(footprint(kind,24,128),[.002,.03,-.02],u,w,10000,20,.4)
        f,m=r['force'],r['moment'];shift=np.array([-m[1]/f[2],m[0]/f[2],0.])
        moved_m=m-np.cross(shift,f);moved_u=u+np.cross(w,shift)
        rows.append(dict(kind=kind,center_of_pressure_offset=shift.tolist(),
                         nominal_origin=directional_residual(f,m,u,w),
                         pressure_origin=directional_residual(f,moved_m,moved_u,w)))
    result=dict(pass_=True,reference_origin_controls=100,length_scale_controls=300,maximum_scaled_error=worst,
                center_of_pressure_examples=rows,experimental_validation=False,
                conclusion='Moving to normal center of pressure removes transverse contact couple for this flat patch, but mixed friction force may still lie outside n/t at that actual point. Total body torque and work are origin invariant. This does not establish incompatibility of every possible directional closure.',
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('audit.py')]})
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
