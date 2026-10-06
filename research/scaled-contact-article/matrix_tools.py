"""Length-scaled matrix tools; matrix-free coupling and native prediction wrapper.

ell is a coordinate scale, L is angular momentum. This module does not introduce
an independently validated material law or replace native contact acceptance.
"""
from dataclasses import dataclass
from pathlib import Path
import sys,copy
import numpy as np
@dataclass
class Body:
    mass: float
    inertia: np.ndarray
    ell: float
    def __post_init__(self):
        self.inertia=np.asarray(self.inertia,dtype=float)
        if not np.isfinite(self.mass) or self.mass<=0 or not np.isfinite(self.ell) or self.ell<=0:raise ValueError('positive finite mass and ell required')
        if self.inertia.shape!=(3,3) or not np.all(np.isfinite(self.inertia)) or not np.allclose(self.inertia,self.inertia.T):raise ValueError('symmetric finite 3x3 world inertia required')
        np.linalg.cholesky(self.inertia)
    def mobility(self,p):
        """Apply M^-1 without forming its inverse."""
        return np.r_[p[:3]/self.mass,self.ell**2*np.linalg.solve(self.inertia,p[3:])]
    def momentum(self,V):return np.r_[self.mass*V[:3],self.inertia@V[3:]/self.ell**2]

def skew(r):
    x,y,z=r;return np.array([[0.,-z,y],[z,0.,-x],[-y,x,0.]])
def contact_map(body,r):return np.column_stack((np.eye(3),-skew(np.asarray(r))/body.ell))
@dataclass
class Contact:
    """impulse on a, opposite impulse on b; r is world COM-to-point."""
    a: int
    b: int
    ra: np.ndarray
    rb: np.ndarray

def apply_delassus(bodies,contacts,impulses):
    """J M^-1 J^T p in O(body+contact) block operations, retaining coupling.

    Inputs here are finite dynamic bodies. Prescribed boundaries require the
    native wrapper; their velocities enter u, their inverse mass is zero.
    Cache the contact maps and inertia factorization in repeated iterations.
    """
    force=np.zeros((len(bodies),6));maps=[]
    for c,j in zip(contacts,impulses,strict=True):
        A=contact_map(bodies[c.a],c.ra);B=contact_map(bodies[c.b],c.rb);maps.append((A,B))
        force[c.a]+=A.T@j;force[c.b]-=B.T@j
    delta=[b.mobility(f) for b,f in zip(bodies,force,strict=True)]
    return np.array([A@delta[c.a]-B@delta[c.b] for c,(A,B) in zip(contacts,maps,strict=True)])

def predict_native(scene,normal_restitution,tangential_restitution,friction,dt=1e-6):
    """Run current coupled 3D solver, with both restitution parameters explicit.

    One common friction overrides both sides, yielding the given pair value.
    Heterogeneous material-pair mixing needs an explicitly declared policy;
    do not silently substitute fitted pair values for independent materials.
    """
    if not np.isfinite(friction) or friction<0:raise ValueError('finite nonnegative friction required')
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from spatial_engine import run
    cfg=copy.deepcopy(scene)
    for body in cfg['bodies']:body['friction']=float(friction)
    return run(cfg,dt=dt,primary_steps=1,iterations=4096,travel_fraction=0,solver='coulomb',kinematic_contact_phase='start',position_stabilization='split_translation_combined',normal_restitution=normal_restitution,tangential_restitution=tangential_restitution,record_contact_impacts=True)

def predict_planar(scene,normal_restitution,tangential_restitution,friction,dt=1e-6):
    """Run isolated coupled planar restitution build; not full-baseline qualified."""
    if not np.isfinite(friction) or friction<0:raise ValueError('finite nonnegative friction required')
    root=Path(__file__).resolve().parents[2];sys.path.insert(0,str(root))
    from rigid_engine import run
    cfg=copy.deepcopy(scene)
    for body in cfg['bodies']:
        for polygon in body.get('polygons',[]):polygon['friction']=float(friction)
    return run(cfg,dt=dt,primary_steps=1,substeps=1,backend='block',position_iterations=12,normal_restitution=normal_restitution,tangential_restitution=tangential_restitution)

def verify():
    rng=np.random.default_rng(20261006);largest=0.
    for trial in range(100):
        bodies=[]
        for j in range(4):
            q=rng.normal(size=(3,3));bodies.append(Body(float(rng.uniform(.1,10)),q.T@q+.1*np.eye(3),10**rng.uniform(-3,2)))
        contacts=[Contact(0,1,rng.normal(size=3),rng.normal(size=3)),Contact(1,2,rng.normal(size=3),rng.normal(size=3)),Contact(2,3,rng.normal(size=3),rng.normal(size=3))]
        J=np.zeros((9,24))
        for i,c in enumerate(contacts):
            J[3*i:3*i+3,6*c.a:6*c.a+6]=contact_map(bodies[c.a],c.ra)
            J[3*i:3*i+3,6*c.b:6*c.b+6]=-contact_map(bodies[c.b],c.rb)
        p=rng.normal(size=(3,3));f=J.T@p.ravel();dense=J@np.concatenate([b.mobility(f[6*i:6*i+6]) for i,b in enumerate(bodies)])
        error=float(np.max(abs(dense-apply_delassus(bodies,contacts,p).ravel())));largest=max(largest,error)
        assert np.allclose(dense,apply_delassus(bodies,contacts,p).ravel(),rtol=1e-11,atol=1e-11)
    return {'case_count':100,'pass_count':100,'max_dense_matrixfree_error':largest,'performance_speedup_claimed':False}
if __name__=='__main__':
    import json
    receipt=verify();Path(__file__).with_name('matrix-tools-audit.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt)
