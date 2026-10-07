"""PSD normal active-set with mechanical-null pressure descent, no compliance."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from research.coulomb_diagnostics import System

def main():
 source=Path('research/hull-active-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_0.json');d=json.loads(source.read_text());s=System.from_dump(d);ks=np.array([c[0] for c in s.contacts]);M=s.A[np.ix_(ks,ks)];b=s.b[ks];tol=d['tolerance_m_s'];out=[]
 for start_name in ['warm','cold']:
  q=np.maximum(0,np.array(d['p'])[ks]) if start_name=='warm' else np.zeros(len(ks));active=set(np.flatnonzero(q>0));trace=[];begin=time.perf_counter()
  def gate(q):
   p=np.zeros(len(s.b));p[ks]=q;return s.gate(p,tol)
  for it in range(512):
   record=dict(iteration=it,active=list(map(int,sorted(active))),gate=gate(q));trace.append(record)
   if record['gate']['accepted']:break
   w=M@q-b
   if not active or max(abs(w[list(active)]))<tol*.1:
    entering=int(np.argmin(w));
    if w[entering]>=-tol:record['stop']='no_violated_normal';break
    active.add(entering);record['entering']=entering
   free=np.array(sorted(active));B=M[np.ix_(free,free)];U,sv,V=np.linalg.svd(B);rank=int(np.sum(sv>sv[0]*1e-13));null=V[rank:];g=(M@q-b)[free]
   ng=null.T@(null@g) if len(null) else np.zeros(len(free))
   if np.max(abs(ng))>1e-12:
    delta=-ng/np.max(abs(ng));kind='null-pressure-descent';record['null_gradient_max']=float(np.max(abs(ng)))
   else:
    delta=-(V[:rank].T@((U[:,:rank].T@g)/sv[:rank]));kind='range-newton'
   direction=np.zeros(len(q));direction[free]=delta;negative=free[delta<0]
   alpha=min((q[j]/-direction[j] for j in negative),default=np.inf)
   if kind=='range-newton':alpha=min(1.,alpha)
   if not np.isfinite(alpha):record['stop']='unbounded_null_descent';break
   response=M@direction;record.update(kind=kind,rank=rank,alpha=float(alpha),direction=direction.tolist(),mobility_response_inf=float(np.max(abs(response))),velocity_change_inf=float(np.max(abs(alpha*response))))
   q+=alpha*direction
   released=[j for j in negative if q[j]<=1e-12]
   for j in released:q[j]=0;active.discard(int(j))
   record['released']=list(map(int,released))
   if alpha==0 and not released:record['stop']='zero_step';break
  report=dict(start=start_name,elapsed_s=time.perf_counter()-begin,gate=gate(q),normal_impulse=q.tolist(),trace=trace);out.append(report);print(start_name,len(trace),report['gate'],flush=True)
 Path(__file__).with_name('normal291-active.json').write_text(json.dumps(dict(capture=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),normal_rows=ks.tolist(),attempts=out),indent=2)+'\n')
if __name__=='__main__':main()
