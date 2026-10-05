"""General outgoing-normal release guide; no capture-specific contact index."""
import json,hashlib,time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from research.coulomb_diagnostics import System

def run(data,max_guesses=4,max_nfev=1000):
 s=System.from_dump(data);tol=data['tolerance_m_s'];seed=np.array(data['p']);ks=np.array([c[0] for c in s.contacts]);contacts=[c for c in s.contacts if seed[c[0]]>0];receipts=[]
 def attempt(kept,initial,label):
  rows=np.array([r for k,t,*_ in kept for r in (k,*t)]);pos={r:i for i,r in enumerate(rows)};cs=tuple((pos[k],tuple(pos[r] for r in t),mu,rn,rt) for k,t,mu,rn,rt in kept);model=System(s.A[np.ix_(rows,rows)],s.b[rows],cs,s.upper[rows]);ns=np.array([c[0] for c in cs]);rn=np.array([c[3] for c in cs])
  def fb(p,jac=False):
   F,J=model.equations(p,True);u=p[ns]/rn;w=model.A[ns]@p-model.b[ns];l=np.hypot(u,w);F[ns]=u+w-l
   if jac:
    ca=1-np.divide(u,l,out=np.zeros_like(u),where=l>0);cb=1-np.divide(w,l,out=np.zeros_like(w),where=l>0);J[ns]=cb[:,None]*model.A[ns];J[ns,ns]+=ca/rn;return J
   return F
  start=time.perf_counter();r=least_squares(fb,initial[rows],jac=lambda p:fb(p,True),max_nfev=max_nfev,ftol=1e-14,xtol=1e-14,gtol=1e-14);p=np.zeros(len(s.b));p[rows]=r.x;raw=p.copy();p[ks]=np.maximum(p[ks],0);gate=s.gate(p,tol);receipts.append(dict(label=label,rows=rows.tolist(),initial_impulse=initial.tolist(),raw_impulse=raw.tolist(),impulse=p.tolist(),gate=gate,nfev=r.nfev,njev=r.njev,elapsed_s=time.perf_counter()-start));return p,gate
 base=seed.copy();gate=s.gate(base,tol)
 if gate['accepted']:return dict(accepted=True,attempts=receipts)
 w=s.A@base-s.b;ranked=sorted([c for c in contacts if base[c[0]]>0 and w[c[0]]>tol],key=lambda c:(-w[c[0]],c[0]));scores=[dict(normal=k,outgoing_velocity_m_s=float(w[k]),pressure_Ns=float(base[k])) for k,*_ in ranked]
 for c in ranked[:max_guesses]:
  p,gate=attempt([c2 for c2 in contacts if c2[0]!=c[0]],base,'release-ranked-outgoing-normal-'+str(c[0]))
  if gate['accepted']:return dict(accepted=True,release_ranking=scores,attempts=receipts)
 return dict(accepted=False,release_ranking=scores,attempts=receipts)

def main():
 capture=Path('research/hull-translation-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_1.json');d=json.loads(capture.read_text());result=run(d);result.update(capture=str(capture),capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),limits=dict(max_release_guesses=4,max_nfev_per_search=1000));dest=Path(__file__).with_name('native-seed-ranked-release.json');assert not dest.exists();dest.write_text(json.dumps(result,indent=2)+'\n');print(result['accepted'],result.get('release_ranking'));print([dict(label=r['label'],nfev=r['nfev'],njev=r['njev'],gate=r['gate']) for r in result['attempts']])
if __name__=='__main__':main()
