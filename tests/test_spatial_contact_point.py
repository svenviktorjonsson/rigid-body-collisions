"""Independent physical gates for the shared-wrench 3D contact convention."""
import copy
import subprocess
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from spatial_engine import BINARY, run, energy
from research.spatial_scenes import sphere


def total_momentum(result):
    states=np.asarray(result['states']);masses=np.asarray(result['mass'])
    P=np.einsum('i,tij->tj',masses,states[:,:,7:10]);L=np.zeros((len(states),3))
    for i,m in enumerate(masses):
        if m==0:continue
        Q=Rotation.from_quat(states[:,i,3:7]).as_matrix()
        I=Q@np.asarray(result['inertia_body_kg_m2'][i])@Q.transpose(0,2,1)
        L+=np.cross(states[:,i,:3],m*states[:,i,7:10])+np.einsum('tij,tj->ti',I,states[:,i,10:13])
    return P,L


def simulate(scene,**kwargs):
    return run(scene,dt=scene['duration'],primary_steps=1,iterations=4096,
               solver='coulomb',travel_fraction=0,kinematic_contact_phase='start',
               position_stabilization='velocity_only',**kwargs)


@unittest.skipUnless(BINARY.exists(),'Build spatial backend')
class SharedContactPoint3D(unittest.TestCase):
    def test_dynamic_pair_conserves_momenta_and_matches_coupled_analytic_impulse(self):
        gap=-.001;lever=.1+gap/2;I=.004;kt=2+2*lever**2/I
        for mu in (.1,1):
            scene=dict(duration=.001,gravity=[0,0,0],bodies=[
                sphere([0,0,.1+gap/2],velocity=[1,0,-1],friction=mu**.5),
                sphere([0,0,-.1-gap/2],velocity=[-1,0,1],friction=mu**.5)])
            for policy in (None,'shared'):
                shared=simulate(scene,contact_point_policy=policy)
                self.assertEqual(shared['numerical_model']['contact_point_policy'],'shared')
                state=np.asarray(shared['states'])
                pt=-min(2/kt,mu);impulse=np.array([pt,0,1])
                np.testing.assert_allclose(state[-1,0,7:10],[1,0,-1]+impulse,atol=1e-10)
                np.testing.assert_allclose(state[-1,1,7:10],[-1,0,1]-impulse,atol=1e-10)
                spin=np.cross([0,0,-lever],impulse)/I
                np.testing.assert_allclose(state[-1,:,10:13],[spin,spin],atol=1e-10)
                P,L=total_momentum(shared)
                np.testing.assert_allclose(P[-1],P[0],atol=1e-12)
                np.testing.assert_allclose(L[-1],L[0],atol=1e-12)
                self.assertAlmostEqual(energy(shared)[-1],1+2*pt+.5*kt*pt**2,delta=1e-11)
                self.assertLessEqual(shared['coulomb_residual_max_m_s'],1e-8)
            legacy=simulate(scene,contact_point_policy='separate')
            self.assertEqual(legacy['numerical_model']['contact_point_policy'],'separate')
            state=np.asarray(legacy['states']);impulse=state[-1,0,7:10]-state[0,0,7:10]
            P,L=total_momentum(legacy)
            np.testing.assert_allclose(L[-1]-L[0],np.cross([0,0,gap],impulse),atol=1e-12)
            self.assertGreater(np.linalg.norm(L[-1]-L[0]),1e-5)

    def test_pair_is_covariant_under_world_rotation_and_body_labels(self):
        gap=-.001;lever=.1+gap/2;pt=-2/(2+2*lever**2/.004)
        for rotvec in ([0,0,0],[.4,-.2,.7]):
            Q=Rotation.from_rotvec(rotvec);n=Q.apply([0,0,1]);t=Q.apply([1,0,0])
            for swap in (False,True):
                bodies=[sphere((.1+gap/2)*n,velocity=(t-n).tolist(),friction=1),
                        sphere(-(.1+gap/2)*n,velocity=(-t+n).tolist(),friction=1)]
                if swap:bodies.reverse()
                result=simulate(dict(duration=.001,gravity=[0,0,0],bodies=bodies))
                i=1 if swap else 0;state=np.asarray(result['states'])
                np.testing.assert_allclose(state[-1,i,7:10],t-n+pt*t+n,atol=1e-10)
                np.testing.assert_allclose(state[-1,i,10:13],Q.apply([0,-lever*pt/.004,0]),atol=1e-10)
                P,L=total_momentum(result)
                np.testing.assert_allclose(P[-1],P[0],atol=1e-12)
                np.testing.assert_allclose(L[-1],L[0],atol=1e-12)

    def test_asymmetric_compounds_preserve_full_tensor_angular_momentum(self):
        parts=[dict(kind='sphere',radius=.1,center=[0,.2,0],density=100),
               dict(kind='sphere',radius=.1,center=[.4,-.2,.15],density=100)]
        mirrored=copy.deepcopy(parts)
        for part in mirrored:part['center']=(-np.asarray(part['center'])).tolist()
        r=np.array([-.2,.2,-.075]);gap=-.001
        scene=dict(duration=1e-5,gravity=[0,0,0],bodies=[
            dict(position=[0,0,0],velocity=[.3,.2,-1],friction=1,shapes=parts),
            dict(position=(2*r+[0,0,-.2-gap]).tolist(),velocity=[-.3,-.2,1],friction=1,shapes=mirrored)])
        result=simulate(scene);I=np.asarray(result['inertia_body_kg_m2'][0])
        self.assertGreater(np.linalg.norm(I-np.diag(np.diag(I))),.001)
        P,L=total_momentum(result)
        np.testing.assert_allclose(P[-1],P[0],atol=1e-11)
        # Audit the impulse phase with its unchanged pre-integration world I.
        # Native free gyro is computed before collision creates this new spin;
        # later orientation integration has its own finite-step L drift.
        immediate=copy.deepcopy(result)
        immediate['states'][-1]=np.asarray(immediate['states'][-1]).tolist()
        for i in range(2):immediate['states'][-1][i][3:7]=result['states'][0][i][3:7]
        _,impulseL=total_momentum(immediate)
        np.testing.assert_allclose(impulseL[-1],impulseL[0],atol=1e-11)
        angular_rate_bound=0.
        for i in range(2):
            q=Rotation.from_quat(result['states'][0][i][3:7]).as_matrix()
            worldI=q@np.asarray(result['inertia_body_kg_m2'][i])@q.T
            omega=np.asarray(result['states'][-1][i][10:13])
            angular_rate_bound+=np.linalg.norm(omega)*np.linalg.norm(worldI@omega)
        self.assertLessEqual(np.linalg.norm(L[-1]-L[0]),2*scene['duration']*angular_rate_bound+1e-11)
        self.assertLessEqual(energy(result)[-1],energy(result)[0]+1e-9)
        self.assertLessEqual(result['coulomb_residual_max_m_s'],1e-8)
        legacy=simulate(scene,contact_point_policy='separate')
        for i in range(2):legacy['states'][-1][i][3:7]=legacy['states'][0][i][3:7]
        _,oldL=total_momentum(legacy)
        self.assertGreater(np.linalg.norm(oldL[-1]-oldL[0]),1e-5)

    def test_finite_body_surface_preserves_fixed_floor_lever(self):
        scene=dict(duration=.001,gravity=[0,0,0],bodies=[
            dict(type='static',position=[0,0,-.05],friction=1,
                 shapes=[dict(kind='box',half_extents=[10,10,.05])]),
            sphere([0,0,.099],velocity=[1,0,-1],friction=1)])
        shared=simulate(scene);legacy=simulate(scene,contact_point_policy='separate')
        np.testing.assert_allclose(shared['states'],legacy['states'],atol=1e-11)
        state=np.asarray(shared['states'])[-1,1]
        np.testing.assert_allclose(state[7:10],[5/7,0,0],atol=1e-10)
        np.testing.assert_allclose(state[10:13],[0,50/7,0],atol=1e-10)

    def test_policy_validation(self):
        scene=dict(duration=.001,gravity=[0,0,0],bodies=[sphere([0,0,0])])
        with self.assertRaises(ValueError):simulate(scene,contact_point_policy='invented')
        with self.assertRaises(ValueError):run(scene,dt=.001,solver='coupled',contact_point_policy='shared')

    def test_native_row_targets_warmstart_gyro_and_full_inertia(self):
        binary=Path(BINARY).parent/'shared_contact_checks'
        if not binary.exists():self.skipTest('Build shared_contact_checks target')
        result=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
        self.assertIn('Shared contact checks PASS',result.stdout)
