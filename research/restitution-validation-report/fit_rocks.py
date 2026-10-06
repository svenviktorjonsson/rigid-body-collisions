"""Prospectively split experimental conditions; fit sphere diagnostic only."""
import argparse,hashlib,json,math
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
import numpy as np
from scipy.optimize import least_squares
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
plan=json.loads((H/'plan.json').read_text());parser=argparse.ArgumentParser();parser.add_argument('--xlsx',type=Path,default=Path('/home/viktor/.cache/physics-public-rock-data-20261006/wang2018/extracted/data set of nhess-2018-108.xlsx'));p=parser.parse_args().xlsx
assert hashlib.sha256(p.read_bytes()).hexdigest()==plan['input_xlsx_sha256'];ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
with ZipFile(p) as z:
 strings=[''.join(t.itertext()) for t in E.fromstring(z.read('xl/sharedStrings.xml'))];xml=E.fromstring(z.read('xl/worksheets/sheet1.xml'))
 data=[];current={}
 for row in xml.findall('s:sheetData/s:row',ns):
  cells={}
  for c in row:
   v=c.find('s:v',ns)
   if v is None:continue
   val=v.text
   if c.get('t')=='s':val=strings[int(val)]
   cells[''.join(x for x in c.get('r') if x.isalpha())]=val
  if int(row.get('r'))<=17 or cells.get('A')=='D':continue
  for k in ('A','B','C'):
   if cells.get(k):current[k]=float(cells[k])
  if not all(cells.get(k) for k in ('D','E','H','I','N')):continue
  data.append({'row':int(row.get('r')),'diameter_m':current['A']/100,'plate_angle_deg':current['B'],'release_height_m':current['C'],
   'vn_before_m_s':float(cells['D']),'vt_before_m_s':float(cells['E']),'impact_angle_deg':float(cells['G']),
   'vn_after_m_s':float(cells['H']),'vt_after_m_s':float(cells['I']),'omega_after_rad_s':float(cells['N'])})
assert len(data)==75

def coefficients(theta,kind,rows):
 theta=np.array(theta);n=len(rows)
 if kind=='fixed':return np.tile(theta,(n,1))
 if kind=='size':return np.array([theta[:3] if r['diameter_m']<.15 else theta[3:] for r in rows])
 x=np.array([(r['impact_angle_deg']-35)/30 for r in rows]);return np.column_stack([np.clip(theta[0]+theta[3]*x,0,1),np.clip(theta[1]+theta[4]*x,-1,1),np.clip(theta[2]+theta[5]*x,0,2)])

def predict(theta,kind,rows):
 coeff=coefficients(theta,kind,rows);vn=np.array([r['vn_before_m_s'] for r in rows]);vt=np.array([r['vt_before_m_s'] for r in rows]);R=np.array([r['diameter_m']/2 for r in rows])
 en,et,mu=coeff.T;jt=np.clip(-(1+et)*vt/3.5,-mu*(1+en)*vn,mu*(1+en)*vn)
 return np.column_stack([en*vn,vt+jt,2.5*abs(jt)/R])

def observed(rows):return np.array([[r['vn_after_m_s'],r['vt_after_m_s'],r['omega_after_rad_s']] for r in rows])

def residual(theta,kind,rows):
 errors=predict(theta,kind,rows)-observed(rows);speed=np.array([math.hypot(r['vn_before_m_s'],r['vt_before_m_s']) for r in rows]);R=np.array([r['diameter_m']/2 for r in rows]);errors[:,2]*=R;return (errors/speed[:,None]).ravel()

def fit(kind,train):
 starts=[np.array([.5,0,.2]),np.array([.8,-.5,.5]),np.array([.4,.5,1.])]
 bounds=([0,-1,0],[1,1,2])
 if kind=='size':starts=[np.tile(x,2) for x in starts];bounds=(bounds[0]*2,bounds[1]*2)
 if kind=='angle':starts=[np.r_[x,[0,0,0]] for x in starts];bounds=([0,-1,0,-2,-2,-2],[1,1,2,2,2,2])
 candidates=[least_squares(lambda x:residual(x,kind,train),x,bounds=bounds,max_nfev=3000,ftol=1e-11,xtol=1e-11,gtol=1e-11) for x in starts]
 best=min(candidates,key=lambda x:float(x.fun@x.fun));singular=np.linalg.svd(best.jac,compute_uv=False);rank=int(np.sum(singular>max(singular[0],1)*1e-8))
 return {'theta':best.x.tolist(),'cost':float(best.fun@best.fun),'jacobian_rank':rank,'parameter_count':len(best.x),'singular_values':singular.tolist(),'all_start_costs':[float(x.fun@x.fun) for x in candidates],'converged':bool(best.success)}

def evaluate(theta,kind,rows):
 predictions=predict(theta,kind,rows);errors=predictions-observed(rows);rmse=np.sqrt(np.mean(errors**2,axis=0));return {'count':len(rows),'normal_velocity_rmse_m_s':float(rmse[0]),'tangent_velocity_rmse_m_s':float(rmse[1]),'angular_speed_rmse_rad_s':float(rmse[2]),'normalized_joint_rmse':float(np.sqrt(np.mean(residual(theta,kind,rows)**2))),
 'predictions':[dict(source_row=r['row'],diameter_m=r['diameter_m'],plate_angle_deg=r['plate_angle_deg'],release_height_m=r['release_height_m'],observed=obs.tolist(),predicted=pred.tolist(),coefficients=c.tolist()) for r,obs,pred,c in zip(rows,observed(rows),predictions,coefficients(theta,kind,rows))]}

training=[r for r in data if r['release_height_m']!=4.5];heldout=[r for r in data if r['release_height_m']==4.5];models={}
for kind in ('fixed','size','angle'):
 f=fit(kind,training);models[kind]={'fit':f,'train':evaluate(f['theta'],kind,training),'heldout':evaluate(f['theta'],kind,heldout)};print(kind,'height holdout',models[kind]['heldout']['normalized_joint_rmse'],flush=True)
folds=[]
for plate in sorted({r['plate_angle_deg'] for r in data}):
 train=[r for r in data if r['plate_angle_deg']!=plate];test=[r for r in data if r['plate_angle_deg']==plate];fold={}
 for kind in ('fixed','size','angle'):
  f=fit(kind,train);fold[kind]={'fit':f,'test':evaluate(f['theta'],kind,test)}
 folds.append({'held_out_plate_angle_deg':plate,'models':fold})
report={'plan':plan,'training_count':len(training),'heldout_count':len(heldout),'data':data,'models':models,'leave_one_plate_angle_out':folds,
 'coefficient_values_are_true_material_properties':False,'location_or_direction_material_dependence_identified':False,'actual_shape_replay_qualified':False,
 'interpretation':'Empirical predictive comparison under sphere/zero-spin assumptions. Size/angle covariates do not identify intrinsic contact-location or directional restitution; geometry and force history are unobserved.'}
(H/'fits.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('Fit comparison complete',len(training),'train',len(heldout),'heldout',flush=True)
