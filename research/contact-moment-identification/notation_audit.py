"""Coordinate-scaling invariance checks, not material validation."""
from pathlib import Path
import json,numpy as np
P=Path(__file__).resolve().parent;rng=np.random.default_rng(20261006);records=[]
def skew(r):
 x,y,z=r;return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
for dimension in [2,3]:
 for sample in range(100):
  m=float(rng.uniform(.1,20));ell=float(10**rng.uniform(-5,3));v=rng.normal(size=dimension);r=rng.normal(size=dimension)
  if dimension==3:
   q=rng.normal(size=(3,3));I=q.T@q+np.eye(3)*.1;omega=rng.normal(size=3);B=-skew(r);rot=omega;physical_torque=lambda J:np.cross(r,J)
  else:
   I=np.array([[float(rng.uniform(.1,5))]]);rot=rng.normal(size=1);B=np.array([[-r[1]],[r[0]]]);physical_torque=lambda J:np.array([r[0]*J[1]-r[1]*J[0]])
  U=np.r_[v,ell*rot];M=np.zeros((len(U),len(U)));M[:dimension,:dimension]=m*np.eye(dimension);M[dimension:,dimension:]=I/ell**2
  C=np.column_stack((np.eye(dimension),B/ell));C0=np.column_stack((np.eye(dimension),B));M0=np.zeros_like(M);M0[:dimension,:dimension]=m*np.eye(dimension);M0[dimension:,dimension:]=I
  momentum=M@U;J=rng.normal(size=dimension);impulse=C.T@J
  err_velocity=float(np.max(abs(C@U-(v+B@rot))));err_momentum=float(np.max(abs(momentum-np.r_[m*v,(I@rot)/ell])));err_duality=abs(float(U@impulse-(C@U)@J));err_torque=float(np.max(abs(impulse[dimension:]-physical_torque(J)/ell)))
  A=C@np.linalg.solve(M,C.T);A0=C0@np.linalg.solve(M0,C0.T);err_A=float(np.max(abs(A-A0)));energy=.5*U@M@U;expected=.5*m*(v@v)+.5*rot@I@rot
  assert err_velocity<1e-10 and err_duality<1e-10 and err_torque<1e-8 and err_A<1e-10 and abs(energy-expected)<1e-10
  assert np.allclose(momentum,np.r_[m*v,(I@rot)/ell],rtol=1e-12,atol=1e-10)
  records.append({'dimension':dimension,'ell':ell,'contact_velocity_error':err_velocity,'momentum_absolute_error':err_momentum,'duality_error':err_duality,'torque_absolute_error':err_torque,'delassus_error':err_A,'energy_error':float(abs(energy-expected)),'passed':True})
(P/'notation-audit.json').write_text(json.dumps({'case_count':len(records),'pass_count':len(records),'seed':20261006,'interpretation':'Numerical checks of coordinate transform, virtual power, angular impulse and scale-invariant contact mobility; no experimental/material validation.','records':records},indent=2)+'\n');print('Notation covariance and duality',len(records),'PASS')
