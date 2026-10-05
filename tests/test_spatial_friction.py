"""Independent mechanics expectations for circular 3D contact friction."""
import unittest
import subprocess
import numpy as np
from spatial_engine import BINARY, run, energy
from research.spatial_scenes import sphere, container
from research.spatial_metrics import diagnostics


def floor_ball(velocity, duration=.01, gravity=0):
    return dict(duration=duration,gravity=[0,0,-gravity],bodies=[
        dict(type='static',position=[0,0,-.05],friction=1,
             shapes=[dict(kind='box',half_extents=[10,10,.05])]),
        sphere([0,0,.1-1e-10],velocity=list(velocity),friction=.4)])


@unittest.skipUnless(BINARY.exists(), 'Build spatial backend')
class Coulomb3D(unittest.TestCase):
    def simulate(self,scene,**kwargs):
        return run(scene,solver='coulomb',kinematic_contact_phase='start',iterations=4096,**kwargs)

    def test_oblique_sliding_is_independent_of_tangent_basis(self):
        for angle in (0,.3,.7,1.5):
            tangent=np.array([np.cos(angle),np.sin(angle),0.])
            result=self.simulate(floor_ball(10*tangent+[0,0,-2]),dt=.01,
                                 primary_steps=1,travel_fraction=0,position_stabilization='velocity_only')
            # Unit mass, I=2mr^2/5, pn=2, |pt|=.4*pn=.8.
            state=np.asarray(result['states'])[-1,1]
            np.testing.assert_allclose(state[7:10],9.2*tangent,atol=1e-10)
            np.testing.assert_allclose(state[10:13],20*np.cross([0,0,1],tangent),atol=1e-10)
            self.assertLess(result['coulomb_residual_max_m_s'],1e-8)
            self.assertEqual(result['coupled_fallbacks'],0)
            self.assertLess(energy(result)[-1],energy(result)[0])

    def test_sticking_impulse_cancels_contact_slip_not_com_velocity(self):
        result=self.simulate(floor_ball([.1,.2,-2]),dt=.01,primary_steps=1,
                             travel_fraction=0,position_stabilization='velocity_only')
        state=np.asarray(result['states'])[-1,1]
        impulse=np.array([-.1,-.2,0])/3.5
        np.testing.assert_allclose(state[7:10],[.1,.2,0]+impulse,atol=1e-10)
        contact_velocity=state[7:10]+np.cross(state[10:13],[0,0,-.1])
        np.testing.assert_allclose(contact_velocity,0,atol=1e-10)

    def test_slow_and_fast_sliding_transition_to_rolling(self):
        for speed,duration in ((.02,.01),(2.,.2)):
            scene=floor_ball([speed,0,0],duration=duration,gravity=9.81)
            result=self.simulate(scene,dt=.01,primary_steps=32)
            state=np.asarray(result['states'])[-1,1]
            # No-slip final state follows horizontal momentum plus rolling torque:
            # vt_final=5/7 vt_initial, r*omega=vt_final.
            self.assertAlmostEqual(state[7],5*speed/7,delta=1e-8)
            self.assertAlmostEqual(.1*state[11],5*speed/7,delta=1e-8)
            self.assertLess(result['coulomb_residual_max_m_s'],1e-8)

    def test_shaking_27_spheres_preserves_work_and_geometry(self):
        scene,half=container(side=3,shake=True)
        result=self.simulate(scene,dt=.01,travel_fraction=.06)
        d=diagnostics(scene,result,half)
        self.assertLess(d['container_surface_excess_m'],.002)
        self.assertLess(d['energy_change_minus_boundary_work_J'],1.)
        self.assertLess(d['quaternion_norm_error'],1e-12)
        self.assertEqual(result['coupled_fallbacks'],0)
        self.assertGreater(result['coulomb_solves'],0)
        self.assertGreater(result['coulomb_fast_solves'],0)
        self.assertLessEqual(result['coulomb_residual_max_m_s'],1e-8)
        self.assertAlmostEqual(result['states'][-1][0][0],.8,places=10)

    def test_offcenter_sticking_retains_full_inertia_and_cross_coupling(self):
        U=np.array([2,.3,-.4]);lever=np.array([-.3,.2,-.075])
        scene=dict(duration=1e-6,gravity=[0,0,0],bodies=[
            dict(type='kinematic',position=[-.35+1e-10,0,0],velocity=U.tolist(),friction=1,
                 shapes=[dict(kind='box',half_extents=[.05,2,2])]),
            dict(friction=10,shapes=[dict(kind='sphere',radius=.1,center=[0,.2,0]),
                                    dict(kind='sphere',radius=.1,center=[.4,-.2,.15])])])
        result=self.simulate(scene,dt=1e-6,primary_steps=1,position_stabilization='velocity_only')
        mass=result['mass'][1];I=np.asarray(result['inertia_body_kg_m2'][1])
        r=np.array([[0,-lever[2],lever[1]],[lever[2],0,-lever[0]],[-lever[1],lever[0],0]])
        K=np.eye(3)/mass-r@np.linalg.solve(I,r)
        impulse=np.linalg.solve(K,U)
        self.assertLess(np.linalg.norm(impulse[1:]),10*impulse[0])
        state=np.asarray(result['states'])[-1,1]
        np.testing.assert_allclose(state[7:10],impulse/mass,atol=1e-8)
        np.testing.assert_allclose(state[10:13],np.linalg.solve(I,np.cross(lever,impulse)),atol=1e-8)

    def test_exhausted_contact_budget_rejects_without_law_fallback(self):
        scene,_=container(side=3,shake=True)
        with self.assertRaises(subprocess.CalledProcessError) as caught:
            run(scene,dt=.01,solver='coulomb',kinematic_contact_phase='start',iterations=1)
        self.assertIn('residual gate failed',caught.exception.stderr)
        self.assertIn('no friction-law fallback',caught.exception.stderr)

    def test_profile_rejects_unsupported_material_and_phase(self):
        scene=floor_ball([1,0,-2]);scene['bodies'][1]['restitution']=.5
        with self.assertRaises(ValueError):self.simulate(scene)
        with self.assertRaises(ValueError):run(floor_ball([1,0,-2]),solver='coulomb')
