"""Audit discontinuous static/dynamic switching at adjacent Float64 limits."""
import argparse
import dataclasses
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE=Path(__file__).resolve().parent


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    return module.ContactHistory


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate',type=Path,default=HERE.parents[1]/'contact_history.py')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--exact-only',action='store_true')
    args=parser.parse_args();old=load('boundary_frozen',HERE/'frozen_contact_history.py');new=load('boundary_candidate',args.candidate)
    rng=np.random.default_rng(20261007);flips=0;largest=0.;count=0;physical_max=0.
    for _ in range(100):
        R=rng.normal(size=(3,3));G=R@R.T;K=10**rng.uniform(0,4,3);h=.01
        u=rng.normal(size=3);eta=rng.normal(size=3)*.02
        reference=old(G,K,h);candidate=new(G,K,h)
        # Same signed sum as the original step, then its exact static boundary.
        required=reference._solve(np.arange(3),-(h*u+2*eta));exact=np.abs(required)
        limits=[exact] if args.exact_only else [exact,np.nextafter(exact,0.),np.nextafter(exact,np.inf)]
        for cs in limits:
            cd=cs*.1;a=reference.step(u,eta,cs,cd);b=candidate.step(u,eta,cs,cd)
            count+=1;flips+=int(a.sliding!=b.sliding);largest=max(largest,float(np.max(np.abs(a.motion-b.motion))))
            for field in dataclasses.fields(a):
                x=np.asarray(getattr(a,field.name),dtype=float);y=np.asarray(getattr(b,field.name),dtype=float)
                physical_max=max(physical_max,float(np.max(np.abs(x-y)/np.maximum(1.,np.abs(x)))))
    receipt=dict(seed=20261007,cases=count,branch_flips=flips,largest_motion_difference=largest,
                 max_scaled_physical_difference=physical_max,candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
                 status='PASS' if flips==0 and physical_max<1e-10 else 'FAIL')
    args.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
    if receipt['status']!='PASS':raise SystemExit(1)


if __name__=='__main__':main()
