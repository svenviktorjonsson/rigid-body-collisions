"""Second independent audit of retained final native continuation solutions."""
import hashlib,json
from pathlib import Path
import numpy as np


def audit():
    source=Path('research/coulomb-trust/native-final-six.jsonl');records=[]
    for line in source.read_text().splitlines():
        result=json.loads(line);capture=Path(result['capture']);data=json.loads(capture.read_text())
        A=np.asarray(data['A']);b=np.asarray(data['b']);p=np.asarray(result['p']);dep=np.asarray(data['dependencies']);hi=np.asarray(data['hi']);w=A@p-b
        residual=0.;cap_excess=0.;normal_negative=0.
        for k in np.flatnonzero(dep<0):
            t=np.flatnonzero(dep==k);normal_negative=max(normal_negative,-p[k],-w[k])
            normal_error=abs(p[k]-max(0,p[k]-w[k]/A[k,k]))*A[k,k]
            tangent_mobility=np.linalg.eigvalsh(A[np.ix_(t,t)])[-1]
            z=p[t]-w[t]/tangent_mobility;capacity=hi[t[0]]*max(0,p[k]);length=np.linalg.norm(z)
            nearest=z*min(1.,capacity/length) if length>0 else z
            tangent_error=np.linalg.norm(p[t]-nearest)*tangent_mobility
            residual=max(residual,float(normal_error),float(tangent_error))
            cap_excess=max(cap_excess,float(np.linalg.norm(p[t])-capacity))
            assert p[k]<=hi[k]
        energy=float(.5*p@A@p-b@p);scale=float(1+np.sum(abs(p*b)))
        assert np.isfinite(p).all() and np.isfinite(w).all() and np.isfinite(energy)
        assert residual<=data['tolerance_m_s'] and energy<=data['tolerance_m_s']*scale
        assert result['svd_calls']<=256 and result['newton_steps']<=512 and result['damped_steps']<=512 and result['attempts']<=96
        assert result['normal_pivot_attempts']<=1
        records.append(dict(capture=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),rows=len(p),
                            independent_original_residual_m_s=residual,passive_change_bound_J=energy,
                            maximum_cap_excess_N_s=cap_excess,maximum_normal_negative=normal_negative,
                            original_equations_accepted=True,svd_calls=result['svd_calls'],normal_pivot_calls=result['normal_pivot_attempts']))
    assert len(records)==6
    package=dict(schema='second-independent-native-coulomb-trust-audit-v1',results_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 cases=records,interpretation='All six fixed captures satisfy their original normal complementarity, circular tangential law and passive-energy bound. This is not acceptance of later full trajectories or a material calibration.')
    out=Path('research/completion-trust-review');out.mkdir(exist_ok=True);(out/'independent-audit.json').write_text(json.dumps(package,indent=2)+'\n')
    print('Independent native continuation audit PASS (six original-equation systems)')


if __name__=='__main__':audit()
