import importlib.util,sys,json,subprocess,hashlib,os,copy
from pathlib import Path
p=Path(__file__).resolve().parent
os.environ.update({k:'1' for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']})
os.environ['PYTHONDONTWRITEBYTECODE']='1'
spec=importlib.util.spec_from_file_location('spatial_snapshot',p/'snapshot/spatial_engine.py');engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
passed=[]; failures=[]
def check(label,f):
 try:f();passed.append(label)
 except Exception as e:failures.append({'case':label,'exception':repr(e)})
def native(w):return subprocess.run([str(p/'runner')],input=json.dumps(w),capture_output=True,text=True)
def need(condition):
 if not condition:raise AssertionError('condition false')
scene=dict(duration=.01,gravity=[0,0,0],bodies=[dict(position=[0,0,1],shapes=[dict(kind='box',half_extents=[.2,.1,.05],density=1000)])])
bodies,*_=engine.prepare(scene)
wire=dict(bodies=bodies,gravity=[0,0,0],frames=1,dt=.01,primary_steps=1,iterations=256,solver='coulomb',travel_fraction=0,minimum_feature_m=.1,margin_m=0,kinematic_contact_phase='start',position_stabilization='split')
def rejects(label,updates,needle):
 r=native(dict(wire,**updates));(p/(label+'.stderr')).write_text(r.stderr);need(r.returncode!=0 and needle in r.stderr)
for name in ['translation_combined_checks','coulomb_early_checks']:
 def runcheck(name=name):
  r=subprocess.run([str(p/name)],capture_output=True,text=True);(p/(name+'.stdout')).write_text(r.stdout);(p/(name+'.stderr')).write_text(r.stderr);need(r.returncode==0)
 check(name,runcheck)
check('native noLAPACK optin rejects',lambda:rejects('no_lapack',{'early_component_recovery':True},'enabled compiled recovery'))
check('native incompatible solver rejects',lambda:rejects('bad_solver',{'early_component_recovery':True,'solver':'coupled'},'requires Coulomb'))
check('native disabled recovery rejects',lambda:rejects('disabled_recovery',{'early_component_recovery':True,'contact_recovery':False},'enabled compiled recovery'))
check('native nonbool option rejects',lambda:rejects('nonbool',{'early_component_recovery':1},'type must be boolean'))
check('native combined incompatible solver rejects',lambda:rejects('combined_coupled',{'position_stabilization':'split_translation_combined','solver':'coupled'},'requires coulomb'))
def defaultparity():
 a=native(wire);b=native(dict(wire,early_component_recovery=False));need(a.returncode==b.returncode==0)
 a=json.loads(a.stdout);b=json.loads(b.stdout);a.pop('step_s');b.pop('step_s');need(a==b and a['lapack_contact_recovery_compiled']==False)
check('native omitted vs explicit false exact default receipts',defaultparity)
base=dict(dt=.01,primary_steps=1,iterations=256,solver='coulomb',kinematic_contact_phase='start',travel_fraction=0,binary=p/'runner')
for name,updates in [('nonbool',dict(early_component_recovery=1)),('noncoulomb',dict(early_component_recovery=True,solver='coupled')),('disabled',dict(early_component_recovery=True,contact_recovery=False)),('combined_non_coulomb',dict(position_stabilization='split_translation_combined',solver='coupled'))]:
 def api_reject(updates=updates):
  try:engine.run(scene,**dict(base,**updates))
  except ValueError:return
  raise AssertionError('API did not reject')
 check('Python '+name+' rejects',api_reject)
def api_meta():
 r=engine.run(scene,position_stabilization='split_translation_combined',**base);need(r['numerical_model']['position_stabilization']=='split_translation_combined' and not r['numerical_model']['early_component_recovery'] and 'accepted physical' in r['numerical_model']['position_clearance_target'])
check('Python combined wire and metadata accepted noLAPACK',api_meta)
def coupled_contact():
 contact=dict(duration=1e-5,gravity=[0,0,0],bodies=[dict(type='kinematic',position=[0,0,-.05],omega=[0,2,0],friction=0,shapes=[dict(kind='box',half_extents=[10,10,.05])]),dict(position=[0,0,.04],omega=[0,0,10],velocity=[0,0,-1],friction=0,shapes=[dict(kind='box',half_extents=[.2,.1,.05],density=1000)])])
 opts=dict(base,dt=1e-5,iterations=4096)
 a=engine.run(contact,position_stabilization='velocity_only',**opts);b=engine.run(contact,position_stabilization='split_translation_combined',**opts)
 import numpy as np
 need(np.array_equal(np.array(a['states'])[:,:,3:],np.array(b['states'])[:,:,3:]) and b['translation_split_solves']>0 and b['translation_split_residual_max_m_s']<=1e-8)
 (p/'combined-contact.json').write_text(json.dumps({'reference':a,'combined':b},indent=2)+'\n')
check('oneframe rotating kinematic contact combined preserves physical endpoint',coupled_contact)
def cmake():
 commands=json.loads((p/'cmake-no-lapack/compile_commands.json').read_text())
 for name in ['runner.cpp','translation_combined_checks.cpp','coulomb_early_checks.cpp']:
  row=[r for r in commands if r['file'].endswith('/'+name)];need(len(row)==1 and 'SPATIAL_LAPACK_RECOVERY' not in row[0]['command'] and 'BT_USE_DOUBLE_PRECISION' in row[0]['command'])
check('CMake noLAPACK includes new targets and correct definitions',cmake)
manifest=json.loads((p/'source-manifest.json').read_text())['sources']
check('snapshot source guards',lambda:need(all(hashlib.sha256((p/f).read_bytes()).hexdigest()==h for f,h in manifest.items())))
receipt=json.loads((p/'compile-receipt.json').read_text())
check('compiler libraries executable guards',lambda:need(all(hashlib.sha256(Path(f).read_bytes()).hexdigest()==h for f,h in receipt['before'].items()) and all(hashlib.sha256((p/f).read_bytes()).hexdigest()==h for f,h in receipt['executables'].items())))
(p/'verification.json').write_text(json.dumps({'passed':passed,'failures':failures,'cost_claim':'None; correctness checks only; standalone executables use existing read-only libs.'},indent=2)+'\n');print(json.dumps({'passed':len(passed),'failures':failures}))
raise SystemExit(bool(failures))
