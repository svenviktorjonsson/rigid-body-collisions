"""Bounded refinement probe; estimator is not a rigorous quadrature certificate."""
import argparse, json, hashlib
from pathlib import Path
import numpy as np
from audit import footprint
from model import evaluate


def scaled_change(a,b,radius):
    scale=max(abs(b['force'][2]),1e-30)
    return max(np.linalg.norm(a['force']-b['force'])/scale,
               np.linalg.norm(a['moment']-b['moment'])/(scale*radius))


def adaptive(templates,arguments,radius,tolerance):
    previous=None;consecutive=0;work=0
    for sites,patch in templates:
        current=evaluate(patch,*arguments);work+=sites
        change=scaled_change(current,previous,radius) if previous is not None else float('inf')
        consecutive=consecutive+1 if change<tolerance/4 else 0
        # Two successive agreements reduce accidental nonmonotone early stopping.
        if consecutive==2:return current,dict(accepted_estimate=True,sites=sites,total_evaluated_sites=work,estimated_change=float(change))
        previous=current
    return current,dict(accepted_estimate=False,sites=sites,total_evaluated_sites=work,estimated_change=float(change))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    rows=[];radius=.012;tol=1e-4
    for kind in ['hertz','ellipse','irregular']:
        templates=[(n*4*n,footprint(kind,n,4*n,radius)) for n in [4,8,16,32,64,128]]
        fine=footprint(kind,192,768,radius)
        for ratio in [.1,.3,.75,1.,2.,8.]:
            arguments=([.002,.015,-.02],[ratio*radius*10,0,-.01],[.1,.2,10],10000,10,.4)
            reference=evaluate(fine,*arguments)
            value,meta=adaptive(templates,arguments,radius,tol)
            actual=float(scaled_change(value,reference,radius))
            baseline=evaluate(templates[0][1],*arguments)
            fixed=evaluate(templates[1][1],*arguments)
            meta.update(kind=kind,velocity_spin_ratio=ratio,tolerance=tol,
                        measured_error_against_294912_site_reference=actual,
                        fixed_64_site_error=float(scaled_change(baseline,reference,radius)),
                        fixed_256_site_error=float(scaled_change(fixed,reference,radius)))
            meta['measured_gate_pass']=bool(actual<=tol)
            # Accurate estimator acceptance is checked on this finite synthetic set only.
            assert not meta['accepted_estimate'] or meta['measured_gate_pass']
            rows.append(meta)
    result=dict(pass_=True,rows=rows,experimental_validation=False,
                note='Finite synthetic refinement study, not a rigorous bound or a production acceptance gate. Templates cached; worst-case site count is costly and declines are retained.',
                source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('model.py'),Path(__file__).with_name('audit.py')]})
    (out/'refinement.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'},indent=2))
