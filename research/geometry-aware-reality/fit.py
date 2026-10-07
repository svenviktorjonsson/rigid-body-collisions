"""Shared geometry/inertia hypotheses; out-of-fit grouped prediction checks."""
from pathlib import Path
import json,math
import numpy as np
from scipy.optimize import least_squares
P=Path(__file__).resolve().parent;source=json.loads((P.parent/'restitution-validation-report/fits.json').read_text());rows=source['data']
kinds=['baseline','inertia','facet']
def predict(theta,kind,data,diagnostics=False):
 en,et,mu=theta[:3];alpha=theta[3] if kind!='baseline' else .4;s=theta[4] if kind=='facet' else 0
 vn=np.array([r['vn_before_m_s'] for r in data]);vt=np.array([r['vt_before_m_s'] for r in data]);R=np.array([r['diameter_m']/2 for r in data]);wnn=1+s*s/alpha;wtt=1+1/alpha;wnt=s/alpha;det=wnn*wtt-wnt*wnt
 rhsn=(1+en)*vn;rhst=-(1+et)*vt;pn=(wtt*rhsn-wnt*rhst)/det;pt=(wnn*rhst-wnt*rhsn)/det
 capped=np.abs(pt)>mu*pn;sign=np.where(pt>=0,1.,-1.);den=wnn+sign*mu*wnt
 pn=np.where(capped,rhsn/np.where(abs(den)>1e-12,den,1e-12),pn);pt=np.where(capped,sign*mu*pn,pt)
 omega=(s*pn+pt)/(alpha*R);out=np.column_stack((-vn+pn,vt+pt,abs(omega)))
 energy=.5*((-vn+pn)**2+(vt+pt)**2+alpha*R**2*omega**2-vn**2-vt**2)
 before=np.column_stack((-vn,vt));after=np.column_stack((-vn+pn+s*R*omega,vt+pt+R*omega));normal_error=after[:,0]+en*before[:,0];tangent_error=after[:,1]+et*before[:,1]
 info={'energy_change_per_mass':energy,'pn':pn,'pt':pt,'omega':omega,'capped':capped,'normal_error':normal_error,'tangent_error':tangent_error,'alpha':alpha,'s':s}
 return (out,info) if diagnostics else out

def obs(data):return np.array([[r['vn_after_m_s'],r['vt_after_m_s'],r['omega_after_rad_s']] for r in data])
def residual(theta,kind,data):
 out,d=predict(theta,kind,data,True);v2=np.array([r['vn_before_m_s']**2+r['vt_before_m_s']**2 for r in data]);R=np.array([r['diameter_m']/2 for r in data]);e=out-obs(data);e[:,2]*=R
 # Physics penalties cannot be mistaken for prediction improvement; final law
 # check rejects any surviving failure regardless of optimization penalty.
 penalties=np.r_[10*np.maximum(d['energy_change_per_mass']/v2,0),10*np.minimum(d['pn']/np.sqrt(v2),0)]
 return np.r_[(e/np.sqrt(v2)[:,None]).ravel(),penalties]

def fit(kind,train):
 base=[[.5,0,.2],[.8,-.5,.5],[.4,.5,1]];starts=base if kind=='baseline' else [x+[a] for x,a in zip(base,[.4,.6,.3])]
 if kind=='facet':starts=[x+[s] for x,s in zip(starts,[0,-.15,.15])]
 lower=[0,-1,0]+([.2] if kind!='baseline' else [])+([-.4] if kind=='facet' else []);upper=[1,1,2]+([1.] if kind!='baseline' else [])+([.4] if kind=='facet' else [])
 runs=[least_squares(lambda t:residual(t,kind,train),start,bounds=(lower,upper),max_nfev=1800,ftol=1e-11,xtol=1e-11,gtol=1e-11) for start in starts];b=min(runs,key=lambda r:r.fun@r.fun);singular=np.linalg.svd(b.jac,compute_uv=False)
 return {'theta':b.x.tolist(),'train_cost':float(b.fun@b.fun),'converged':bool(b.success),'rank':int(sum(singular>max(1,singular[0])*1e-8)),'parameter_count':len(b.x),'start_costs':[float(r.fun@r.fun) for r in runs]}

def evaluate(theta,kind,data):
 p,d=predict(theta,kind,data,True);o=obs(data);R=np.array([r['diameter_m']/2 for r in data]);v=np.array([math.hypot(r['vn_before_m_s'],r['vt_before_m_s']) for r in data]);e=p-o;scaled=e.copy();scaled[:,2]*=R;joint=float(np.sqrt(np.mean((scaled/v[:,None])**2)));metric=[];records=[]
 for i,r in enumerate(data):
  qo=abs(o[i]).copy();qp=abs(p[i]).copy();qo[2]*=R[i];qp[2]*=R[i];a=float(np.degrees(np.arccos(np.clip(qo@qp/(np.linalg.norm(qo)*np.linalg.norm(qp)),-1,1))));mag=float(100*(np.linalg.norm(qp)/np.linalg.norm(qo)-1));metric.append((a,mag))
  valid=bool(d['pn'][i]>=-1e-9 and abs(d['pt'][i])<=theta[2]*d['pn'][i]+1e-9 and d['energy_change_per_mass'][i]<=1e-8 and abs(d['normal_error'][i])<1e-8 and (d['capped'][i] or abs(d['tangent_error'][i])<1e-8))
  records.append({'source_row':r['row'],'diameter_m':r['diameter_m'],'release_height_m':r['release_height_m'],'plate_angle_deg':r['plate_angle_deg'],'observed':o[i].tolist(),'predicted':p[i].tolist(),'map_angle_deg':a,'map_magnitude_error_percent':mag,'normalised_squared_error':float(np.mean((scaled[i]/v[i])**2)),'normal_impulse_per_mass':float(d['pn'][i]),'tangent_impulse_per_mass':float(d['pt'][i]),'energy_change_per_mass':float(d['energy_change_per_mass'][i]),'contact_law_passed':valid})
 metric=np.array(metric);return {'count':len(data),'joint_rmse':joint,'component_rmse':np.sqrt(np.mean(e**2,axis=0)).tolist(),'map_angle_rmse_deg':float(np.sqrt(np.mean(metric[:,0]**2))),'map_magnitude_rmse_percent':float(np.sqrt(np.mean(metric[:,1]**2))),'law_pass_count':sum(r['contact_law_passed'] for r in records),'predictions':records}
report={'plan':json.loads((P/'plan.json').read_text()),'models':{},'grouped':{},'independent_blind_test':False}
train=[r for r in rows if r['release_height_m']!=4.5];test=[r for r in rows if r['release_height_m']==4.5]
for kind in kinds:
 f=fit(kind,train);report['models'][kind]={'fit':f,'out_of_fit_development':evaluate(f['theta'],kind,test)};print('development',kind,report['models'][kind]['out_of_fit_development']['joint_rmse'],f['theta'],flush=True)
for group in ['plate_angle_deg','release_height_m']:
 folds=[]
 for value in sorted({r[group] for r in rows}):
  train=[r for r in rows if r[group]!=value];test=[r for r in rows if r[group]==value];fold={'withheld':value,'models':{}}
  for kind in kinds:
   f=fit(kind,train);fold['models'][kind]={'fit':f,'test':evaluate(f['theta'],kind,test)}
  folds.append(fold);print('fold',group,value,{k:round(fold['models'][k]['test']['joint_rmse'],6) for k in kinds},flush=True)
 pooled={}
 for kind in kinds:
  predictions=[r for fold in folds for r in fold['models'][kind]['test']['predictions']];assert len(predictions)==75
  pooled[kind]={'joint_rmse':math.sqrt(sum(r['normalised_squared_error'] for r in predictions)/75),'map_angle_rmse_deg':math.sqrt(sum(r['map_angle_deg']**2 for r in predictions)/75),'map_magnitude_rmse_percent':math.sqrt(sum(r['map_magnitude_error_percent']**2 for r in predictions)/75),'law_pass_count':sum(r['contact_law_passed'] for r in predictions),'predictions':predictions}
 for kind in kinds[1:]:
  b={r['source_row']:r for r in pooled['baseline']['predictions']};pooled[kind]['improved_case_count']=sum(r['normalised_squared_error']<b[r['source_row']]['normalised_squared_error'] for r in pooled[kind]['predictions']);pooled[kind]['regressed_case_count']=75-pooled[kind]['improved_case_count']
 report['grouped'][group]={'folds':folds,'pooled':pooled};print('POOLED',group,{k:round(v['joint_rmse'],6) for k,v in pooled.items()},flush=True)
balls=json.loads((P.parent/'contact-moment-identification/comparison.json').read_text())['rows'];rubber=[]
for ball in ['Superball','golf']:
 batch=[r for r in balls if r['ball']==ball]
 for held,row in enumerate(batch):
  training=[r for j,r in enumerate(batch) if j!=held];K=np.array([(1+r['tangential_restitution'])*math.sin(math.radians(25))/r['radius_m'] for r in training]);S=np.array([r['observed_spin_factor_rad_m'] for r in training]);beta=float(np.clip(K@S/(K@K),1/(1+2/3),1/(1+.2)));alpha=1/beta-1;pred=(1+row['tangential_restitution'])*math.sin(math.radians(25))/(row['radius_m']*(1+alpha))
  rubber.append({'ball':ball,'withheld_surface':row['surface'],'alpha_from_other_surfaces':alpha,'predicted_spin':pred,'observed_spin':row['observed_spin_factor_rad_m'],'baseline_spin':row['point_law_spin_factor_rad_m'],'training_surfaces':[r['surface'] for r in training]})
report['rubber']={'folds':rubber,'baseline_rmse':math.sqrt(sum((r['baseline_spin']-r['observed_spin'])**2 for r in rubber)/8),'candidate_rmse':math.sqrt(sum((r['predicted_spin']-r['observed_spin'])**2 for r in rubber)/8),'independently_measured_inertia':False}
for kind in kinds[1:]:report['models'][kind]['passes_declared_pooled_acceptance']=all(report['grouped'][g]['pooled'][kind]['joint_rmse']<report['grouped'][g]['pooled']['baseline']['joint_rmse'] and report['grouped'][g]['pooled'][kind]['law_pass_count']==75 for g in report['grouped'])
report['production_adopted']=False;report['actual_shape_geometry_validated']=False
(P/'results.json').write_text(json.dumps(report,indent=2)+'\n');print('rubber',report['rubber']['baseline_rmse'],report['rubber']['candidate_rmse'],flush=True)
