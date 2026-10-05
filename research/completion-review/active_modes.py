"""Independent bounded mode enumeration of a captured Coulomb system.

This diagnostic fixes the warm iterate's positive-normal contact set and
enumerates its sticking/sliding tangent modes. It is neither a complete
existence certificate nor an engine solver; original full equations gate
every discovered candidate. No matrix compliance or friction change occurs.
"""
import argparse,itertools,json
from pathlib import Path
import numpy as np
from scipy.optimize import root
from research.coulomb_diagnostics import System


def examine(path,starts=3):
    data=json.loads(Path(path).read_text()); sys=System.from_dump(data)
    warm=np.asarray(data['p']); active=[c for c in sys.contacts if warm[c[0]]>1e-9]
    if len(active)>10:raise ValueError('bounded diagnostic supports at most ten positive normals')
    attempts=[]; best=None
    for modes in itertools.product((False,True),repeat=len(active)):
        index=[]; size=0
        for slide in modes:index.append(size);size+=2 if slide else 3
        def impulses(z):
            p=np.zeros_like(warm)
            for c,slide,j in zip(active,modes,index):
                k,(t,s),mu,*_=c;p[k]=z[j]
                if slide:p[[t,s]]=mu*z[j]*np.array([np.cos(z[j+1]),np.sin(z[j+1])])
                else:p[[t,s]]=z[j+1:j+3]
            return p
        def equations(z):
            p=impulses(z);w=sys.A@p-sys.b;F=[]
            for c,slide,j in zip(active,modes,index):
                k,(t,s),mu,*_=c;F.append(w[k])
                if slide:F.append(-np.sin(z[j+1])*w[t]+np.cos(z[j+1])*w[s])
                else:F.extend(w[[t,s]])
            return np.asarray(F)
        initial=np.zeros(size)
        for c,slide,j in zip(active,modes,index):
            k,(t,s),mu,*_=c;initial[j]=warm[k]
            if slide:initial[j+1]=np.arctan2(warm[s],warm[t])
            else:initial[j+1:j+3]=warm[[t,s]]
        for restart in range(starts):
            z=initial.copy()
            for slide,j in zip(modes,index):
                if slide:z[j+1]+=restart*2*np.pi/starts
            result=root(equations,z,method='lm',options={'ftol':1e-12,'xtol':1e-12,'gtol':1e-12,'maxiter':1500})
            p=impulses(result.x);gate=sys.gate(p,data['tolerance_m_s'])
            record={'sliding':list(modes),'restart':restart,'optimizer_success':bool(result.success),'nfev':int(result.nfev),**gate}
            attempts.append(record)
            if best is None or gate['residual_m_s']<best['residual_m_s']:best={**record,'p':p.tolist(),'w':(sys.A@p-sys.b).tolist()}
            if gate['accepted']:return {'capture':str(path),'active_normal_rows':[c[0] for c in active],'attempts':attempts,'best':best,'complete':False,'found_exact_solution':True}
    return {'capture':str(path),'active_normal_rows':[c[0] for c in active],'attempts':attempts,'best':best,'complete':True,'found_exact_solution':False}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('capture');parser.add_argument('--output',required=True);parser.add_argument('--starts',type=int,default=3)
    args=parser.parse_args();result=examine(args.capture,args.starts)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:result[key] for key in ('capture','active_normal_rows','complete','found_exact_solution')}));print(json.dumps({key:value for key,value in result['best'].items() if key not in ('p','w')}))
