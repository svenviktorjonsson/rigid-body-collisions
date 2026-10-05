"""Replay frozen captures and check unchanged law, covariance, and effort cap."""
import json
from pathlib import Path
import numpy as np
from research.coulomb_diagnostics import System,recover
from research.hull_recovery_followup import opposing_slip_restart
ROOT=Path(__file__).parent

def mechanical_audit(S,p,tolerance=1e-8):
 p=np.asarray(p);w=S.A@p-S.b;rows=[]
 for k,ts,mu,rn,rt in S.contacts:
  t=list(ts);z=p[t]-rt*w[t];length=np.linalg.norm(z);cap=mu*max(p[k],0.)
  projected=z if length<=cap else z*(cap/length) if length else np.zeros(2)
  rows.append(dict(normal=k,normal_residual_m_s=float(abs(min(p[k]/rn,w[k]))),
                   tangent_residual_m_s=float(np.linalg.norm((p[t]-projected)/rt)),
                   cone_excess_N_s=float(max(0.,np.linalg.norm(p[t])-cap)),
                   positive_tangent_work_J=float(max(0.,p[t]@w[t]))))
 residual=max(max(r['normal_residual_m_s'],r['tangent_residual_m_s']) for r in rows)
 work=float(.5*p@S.A@p-S.b@p);scale=1+float(np.sum(abs(p*S.b)))
 return dict(accepted=bool(np.isfinite(p).all() and np.isfinite(work) and np.isfinite(scale) and residual<=tolerance and work<=tolerance*scale),residual_m_s=residual,passive_change_bound_J=work,contacts=rows)

def main():
 d=json.loads((ROOT/'hull42-second-rejected.json').read_text());S=System.from_dump(d)
 r=opposing_slip_restart(S,d['p']);audit=mechanical_audit(S,r['result']['impulse'])
 assert audit['accepted'] and r['newton_calls']<=256
 bounded=opposing_slip_restart(S,d['p'],max_calls=4)
 assert bounded['newton_calls']<=4 and not bounded['result']['accepted']
 # Independently rotate every local tangent coordinate pair. The restart is
 # based on the physical post-slip vector, never a preferred tangent axis.
 T=np.eye(len(S.b));angle=.731;c=np.cos(angle);s=np.sin(angle)
 for k,t,*_ in S.contacts:T[np.ix_(t,t)]=[[c,-s],[s,c]]
 rotated=System(T@S.A@T.T,T@S.b,S.contacts,S.upper)
 rr=opposing_slip_restart(rotated,T@np.asarray(d['p']))
 assert mechanical_audit(rotated,rr['result']['impulse'])['accepted']
 assert np.max(abs((S.A@np.asarray(r['result']['impulse'])-S.b)-T.T@(rotated.A@np.asarray(rr['result']['impulse'])-rotated.b)))<1e-8
 d2=json.loads((ROOT/'hull7301-second-rejected.json').read_text());S2=System.from_dump(d2)
 r2=recover(S2,d2['p']);audit2=mechanical_audit(S2,r2['impulse'])
 assert audit2['accepted']
 output=dict(hull42=audit,hull7301=audit2,budget_rejection={'calls':bounded['newton_calls'],'accepted':bounded['result']['accepted']},rotated_hull42=mechanical_audit(rotated,rr['result']['impulse']))
 (ROOT/'independent-verification.json').write_text(json.dumps(output,indent=2)+'\n')
 print('4 checks passed: two captures, bounded rejection, tangent-basis covariance')
if __name__=='__main__':main()
