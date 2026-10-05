import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from spatial_engine import BINARY, run, moments, prepare, energy, errors
from research.spatial_scenes import wall_impact, driven_row, container
from research.spatial_metrics import diagnostics


class Geometry3D(unittest.TestCase):
    def test_exact_hull_volume_and_full_inertia(self):
        R=Rotation.from_rotvec([.4,-.3,.2]).as_matrix()
        vertices=np.array([[x,y,z] for x in [-1,1] for y in [-2,2] for z in [-3,3]])@R.T+[4,5,6]
        v,c,I,_=moments(dict(kind='hull',vertices=vertices.tolist()))
        self.assertAlmostEqual(v,48,places=11)
        np.testing.assert_allclose(c,[4,5,6],atol=1e-13)
        np.testing.assert_allclose(I,R@np.diag([208,160,80])@R.T,atol=1e-11)
        self.assertGreater(abs(I[0,1]),1)

    def test_compound_parallel_axis_and_principal_frame(self):
        s=dict(duration=.1,bodies=[dict(shapes=[dict(kind='sphere',radius=1,center=[-2,0,0]),dict(kind='sphere',radius=1,center=[2,0,0])])])
        bodies,mass,I,Q,_=prepare(s)
        m=4*np.pi/3
        np.testing.assert_allclose(I[0],np.diag([.8*m,8.8*m,8.8*m]),atol=1e-12)
        self.assertAlmostEqual(mass[0],2*m)
        np.testing.assert_allclose(Q[0]@np.diag(bodies[0]['principal_inertia'])@Q[0].T,I[0],atol=1e-12)

    def test_scene_validation(self):
        s=wall_impact();s['bodies'][0]['type']='static'
        with self.assertRaises(ValueError):prepare(s)
        s=wall_impact();s['bodies'][1]['orientation']=[0,0,0,2]
        with self.assertRaises(ValueError):prepare(s)


@unittest.skipUnless(BINARY.exists(),'Build spatial_backend first')
class Mechanics3D(unittest.TestCase):
    def test_fast_wall_restitution_and_actuator_work(self):
        for solver in ('sequential','coupled'):
            for e in (0.,.5,1.):
                r=run(wall_impact(restitution=e),solver=solver,dt=.01)
                s=np.asarray(r['states'])
                np.testing.assert_allclose(s[-1,1,7:10],[(1+e)*20,0,0],atol=1e-8)
                np.testing.assert_allclose(s[:,0,0],-1+20*np.asarray(r['times']),atol=1e-12)
                self.assertAlmostEqual(r['boundary_work_J'],20*(1+e)*20,places=6)
                self.assertLessEqual(energy(r)[-1]-r['boundary_work_J'],1e-8)
                self.assertEqual(r['coupled_fallbacks'],0)

    def test_stationary_object_wall_tunneling_control(self):
        # Wall crosses the entire sphere in one output frame; dynamic body starts at rest.
        r=run(wall_impact(),dt=.1,primary_steps=1)
        self.assertAlmostEqual(r['states'][-1][1][7],40,places=7)
        self.assertGreater(r['collision_updates'],100)
        unguarded=run(wall_impact(),dt=.1,primary_steps=1,travel_fraction=0)
        self.assertEqual(unguarded['states'][-1][1][7],0)

    def test_many_simultaneous_contacts_all_three_axes(self):
        for axis in range(3):
            r=run(driven_row(16,axis=axis),dt=.01)
            velocity=np.asarray(r['states'])[-1,1:,7:10]
            expected=np.tile(np.eye(3)[axis]*20,(16,1))
            np.testing.assert_allclose(velocity,expected,atol=.015)
            # Upstream MLCP can fall back on degenerate friction/split rows.
            self.assertEqual(r['coupled_updates'],r['collision_updates'])

    def test_100_m_s_wall_drives_64_bodies(self):
        r=run(driven_row(64,speed=100),dt=.01)
        np.testing.assert_allclose(np.asarray(r['states'])[-1,1:,7:10],np.tile([100,0,0],(64,1)),atol=.015)
        self.assertAlmostEqual(r['boundary_work_J'],64*100**2,delta=1.)
        self.assertLess(energy(r)[-1]-r['boundary_work_J'],0)

    def test_fast_container_64_objects_and_random_rotating_shapes(self):
        for side,shape,shake,spin in ((4,'sphere',False,0),(3,'hull',True,10)):
            scene,half=container(side=side,shape=shape,shake=shake,spin=spin)
            r=run(scene,dt=.01,solver='adaptive')
            d=diagnostics(scene,r,half)
            self.assertLess(d['container_surface_excess_m'],.002)
            self.assertLess(d['quaternion_norm_error'],1e-12)
            self.assertLess(d['energy_change_minus_boundary_work_J'],1.)
            self.assertGreater(r['coupled_updates'],0)
            self.assertGreater(r['sequential_updates'],0)
            # Prescribed wall path, including reversals, is independent of contents.
            expected=.8 if shake else 2.4
            self.assertAlmostEqual(r['states'][-1][0][0],expected,places=10)
            if not shake:
                self.assertAlmostEqual(np.mean(np.asarray(r['states'])[-1,1:,7]),20,delta=.02)

    def test_all_predictive_hull_impulses_count_in_wall_work(self):
        scene,half=container(shape='hull')
        r=run(scene,dt=.01,solver='sequential')
        s=np.asarray(r['states']);mass=np.asarray(r['mass'])
        # Every wall translates at the same U; gravity has no x component.
        momentum_change=np.sum(mass*(s[-1,:,7]-s[0,:,7]))
        self.assertAlmostEqual(r['boundary_work_J'],20*momentum_change,delta=1e-7)

    def test_angular_collision_against_analytic_impulse(self):
        # Corner sphere hits translating plane, creating an off-center torque.
        s=dict(duration=.01,gravity=[0,0,0],bodies=[dict(type='kinematic',position=[-.15,0,0],velocity=[2,0,0],friction=0,shapes=[dict(kind='box',half_extents=[.05,2,2])]),dict(friction=0,shapes=[dict(kind='sphere',radius=.1,center=[0,.2,0]),dict(kind='sphere',radius=.1,center=[.4,-.2,.15])])])
        # Prepare recenters shape COM: move wall to first sphere's left support.
        s['bodies'][0]['position'][0]=-.35
        r=run(s,dt=.01,primary_steps=100)
        mass=r['mass'][1];I=np.asarray(r['inertia_body_kg_m2'][1]);lever=np.array([-.3,.2,-.075]);n=np.array([1.,0,0]);cross=np.cross(lever,n)
        impulse=2/(1/mass+cross@np.linalg.solve(I,cross))
        np.testing.assert_allclose(r['states'][-1][1][7:10],n*impulse/mass,atol=.01)
        np.testing.assert_allclose(r['states'][-1][1][10:13],np.linalg.solve(I,cross*impulse),atol=.1)

    def test_free_spin_full_tensor_and_quaternion_norm(self):
        scene=dict(duration=.1,gravity=[0,0,0],bodies=[dict(omega=[1,2,3],orientation=Rotation.from_rotvec([.3,.2,.1]).as_quat().tolist(),shapes=[dict(kind='box',half_extents=[.2,.3,.4])])])
        r=run(scene,dt=.01,primary_steps=64)
        states=np.asarray(r['states'])[:,0];R=Rotation.from_quat(states[:,3:7]).as_matrix();I=np.asarray(r['inertia_body_kg_m2'][0])
        L=np.einsum('tij,jk,tlk,tl->ti',R,I,R,states[:,10:13])
        np.testing.assert_allclose(np.linalg.norm(states[:,3:7],axis=1),1,atol=1e-13)
        self.assertLess(np.max(np.linalg.norm(L-L[0],axis=1)),2e-6)
        self.assertLess(np.ptp(energy(r)),2e-6)

    def test_friction_slows_sliding_and_orientation_metric(self):
        s=dict(duration=.2,gravity=[0,0,-9.81],bodies=[dict(type='static',position=[0,0,-.05],friction=1,shapes=[dict(kind='box',half_extents=[10,10,.05])]),dict(position=[0,0,.101],velocity=[2,1,0],friction=.4,shapes=[dict(kind='box',half_extents=[.1,.1,.1],density=125)])])
        r=run(s,dt=.01);a=np.asarray(r['states'])
        self.assertLess(np.linalg.norm(a[-1,1,7:9]),np.linalg.norm(a[0,1,7:9])-.5)
        self.assertEqual(errors(r,r),dict(position_m=0,velocity_m_s=0,omega_rad_s=0,orientation_rad=0))
        neg=copy.deepcopy(r);data=np.asarray(neg['states']);data[:,:,3:7]*=-1;neg['states']=data.tolist()
        self.assertLess(errors(r,neg)['orientation_rad'],1e-14)
        bad=copy.deepcopy(r);bad['physical_setup_id']='different'
        with self.assertRaises(ValueError):errors(r,bad)
