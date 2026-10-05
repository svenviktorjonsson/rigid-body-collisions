"""Independent original all-row mechanical audit; no solver/helper imports."""
import hashlib,json
from pathlib import Path
import numpy as np

def check(d,p):
 A=np.array(d['A']);b=np.array(d['b']);dep=np.array(d['dependencies']);hi=np.array(d['hi']);p=np.array(p);tol=d['tolerance_m_s'];w=A@p-b;ks=np.flatnonzero(dep<0);res=0.;nn=0.;cone=0.;support=0.;complement=0.;positive_work=0.
 for k in ks:
  t=np.flatnonzero(dep==k);assert len(t)==2 and hi[t[0]]==hi[t[1]];u,v=t;mu=hi[u];eig=.5*(A[u,u]+A[v,v]+np.hypot(A[u,u]-A[v,v],2*A[u,v]));assert eig>0;z=p[t]-w[t]/eig;length=np.linalg.norm(z);cap=mu*max(0,p[k]);disk=z*(min(1,cap/length) if length>0 else 1);res=max(res,abs(p[k]-max(0,p[k]-w[k]/A[k,k]))*A[k,k],np.linalg.norm(p[t]-disk)*eig);nn=max(nn,-w[k]);cone=max(cone,(np.linalg.norm(p[t])-cap)*eig);work=p[t]@w[t];support=max(support,abs(work+cap*np.linalg.norm(w[t])));complement=max(complement,abs(p[k]*w[k]));positive_work=max(positive_work,work)
 energy=.5*p@A@p-b@p;scale=1+np.sum(abs(p*b));impulse_scale=max(1,float(np.max(abs(p))));finite=np.isfinite(p).all() and np.isfinite(w).all() and np.isfinite(energy) and np.isfinite(scale);bounds=np.min(p[ks])>=0 and np.all(p[ks]<=hi[ks]);original=bool(finite and bounds and res<=tol and energy<=tol*scale);physical=bool(finite and bounds and nn<=tol and cone<=tol and support<=tol*impulse_scale and complement<=tol*impulse_scale and positive_work<=tol*impulse_scale and energy<=tol*scale)
 return dict(accepted=original and physical,original_projection_gate=original,independent_physical_gate=physical,residual_m_s=float(res),passive_change_bound_J=float(energy),passivity_scale=float(scale),normal_negative_m_s=float(nn),cone_violation_m_s=float(cone),support_gap_J=float(support),normal_complementarity_work_J=float(complement),positive_friction_work_J=float(positive_work))

def main():
 folder=Path(__file__).parent;cap=Path('research/hull-translation-completion/results/rejections/fast_rotate_shake27_hulls7301/reference_1.json');expected='87c8637e8df9bbcbe085c184b27e012808afcc897d26547bdf53ae5312c49d7b';assert hashlib.sha256(cap.read_bytes()).hexdigest()==expected;data=json.loads(cap.read_text());rows=[]
 for filename in ['full-warm-FB.json','positive-warm-FB.json','native-seed-ranked-release.json','normal-release.json','component-warm-FB.json']:
  r=json.loads((folder/filename).read_text());trials=r.get('attempts',[r])
  for trial in trials:
   p=trial.get('clipped_impulse',trial['impulse']);gate=trial.get('clipped_gate',trial['gate']);audit=check(data,p);assert audit['accepted']==gate['accepted'];rows.append(dict(receipt=filename,label=trial.get('label',str(trial.get('released_normal',trial.get('component','direct')))),**audit))
 r=json.loads((folder/'component-warm-FB.json').read_text());trial=r['attempts'][0];selected=np.array(trial['rows']);A=np.array(data['A']);before=np.array(trial['initial_impulse']);after=np.array(trial['impulse']);outside=np.setdiff1d(np.arange(len(before)),selected);otherpositive=outside[before[outside]!=0];assert np.all(A[np.ix_(selected,otherpositive)]==0);assert np.array_equal(before[outside],after[outside]);assert len(selected)==9
 out=dict(capture_sha256=expected,status='PASS',component_cross_positive_mobility_exactly_zero=True,other_impulses_unchanged=True,trials=rows);(folder/'independent-audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
