"""Independent mechanics expectations for circular 3D contact friction."""
import unittest
import subprocess
import json
import tempfile
from pathlib import Path
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

    def test_rejected_system_snapshot_is_reproducible_and_never_accepted(self):
        scene,_=container(side=3,shake=True)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rejected.json'
            with self.assertRaises(subprocess.CalledProcessError):
                run(scene,solver='coulomb',kinematic_contact_phase='start',dt=.01,iterations=1,rejected_contact_path=path)
            data=json.loads(path.read_text())
            self.assertEqual(data['schema'],'circular-coulomb-rejection-v1')
            self.assertEqual(data['phase'],'velocity')
            A=np.asarray(data['A']);b=np.asarray(data['b']);p=np.asarray(data['p'])
            self.assertEqual(A.shape,(len(p),len(p)))
            np.testing.assert_allclose(A,A.T,atol=1e-12)
            self.assertTrue(np.isfinite(A@p-b).all())
            self.assertGreater(data['residual_m_s'],data['tolerance_m_s'])
            original=path.read_bytes()
            with self.assertRaises(ValueError):self.simulate(scene,rejected_contact_path=path)
            self.assertEqual(path.read_bytes(),original)
            with self.assertRaises(ValueError):run(scene,solver='sequential',rejected_contact_path=Path(directory)/'other.json')

    def test_gap_pose_policy_preserves_spin_and_accounts_gravity_repair(self):
        scene=floor_ball([0,0,0],duration=1e-4,gravity=9.81)
        scene['bodies'][1]['position'][2]=.1-5e-5
        scene['bodies'][1]['omega']=[0,0,10]
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'progress.json'
            result=self.simulate(scene,dt=1e-4,primary_steps=1,travel_fraction=0,
                                 position_stabilization='split_translation_gap',
                                 progress_checkpoint_path=checkpoint)
            prefix=json.loads(checkpoint.read_text())
        before,after=np.asarray(result['states'])[:,1]
        np.testing.assert_allclose(after[10:13],[0,0,10],atol=1e-12)
        np.testing.assert_allclose(after[7:10],0,atol=1e-12)
        self.assertAlmostEqual(energy(result)[1],energy(result)[0],delta=1e-12)
        delta_z=after[2]-before[2]
        self.assertGreater(delta_z,0)
        expected=result['mass'][1]*9.81*delta_z
        self.assertAlmostEqual(result['translation_pose_potential_change_J'],expected,delta=1e-12)
        self.assertLess(result['translation_split_residual_max_m_s'],1e-8)
        np.testing.assert_allclose(result['translation_pose_orbital_change_kg_m2_s'],0,atol=1e-12)
        self.assertTrue(prefix['complete'])
        for key in result:
            if key.startswith('translation_pose_'):
                self.assertEqual(prefix[key],result[key])
        self.assertEqual(result['numerical_model']['position_stabilization'],'split_translation_gap')

    def test_profile_rejects_unsupported_material_and_phase(self):
        scene=floor_ball([1,0,-2]);scene['bodies'][1]['restitution']=.5
        with self.assertRaises(ValueError):self.simulate(scene)
        with self.assertRaises(ValueError):run(floor_ball([1,0,-2]),solver='coulomb')
        with self.assertRaises(ValueError):self.simulate(floor_ball([1,0,-2]),contact_recovery='yes')


    def test_combined_pose_policy_does_not_spend_clearance_twice(self):
        radius=.1
        scene=dict(duration=.001,gravity=[0,0,0],bodies=[
            dict(type='static',position=[-.1095,0,0],friction=0,
                 shapes=[dict(kind='box',half_extents=[.01,1,1])]),
            dict(type='static',position=[.111000001,0,0],friction=0,
                 shapes=[dict(kind='box',half_extents=[.01,1,1])]),
            dict(position=[0,0,0],velocity=[.95,0,0],friction=0,
                 shapes=[dict(kind='sphere',radius=radius,
                              density=1/(4*np.pi*radius**3/3))])])
        old=self.simulate(scene,dt=.001,primary_steps=1,travel_fraction=0,
                          position_stabilization='split_translation_gap')
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'progress.json'
            new=self.simulate(scene,dt=.001,primary_steps=1,travel_fraction=0,
                              position_stabilization='split_translation_combined',
                              early_component_recovery=True,
                              progress_checkpoint_path=checkpoint)
            prefix=json.loads(checkpoint.read_text())
        self.assertAlmostEqual(new['mass'][2],1,delta=1e-12)
        old_state=np.asarray(old['states'])[-1,2]
        new_state=np.asarray(new['states'])[-1,2]
        right_face=.101000001
        self.assertLess(right_face-old_state[0]-radius,-4.9e-5)
        self.assertGreater(right_face-new_state[0]-radius,4.9e-5)
        np.testing.assert_allclose(new_state[:3],[.00095,0,0],atol=1e-12)
        np.testing.assert_allclose(new_state[7:13],[.95,0,0,0,0,0],atol=1e-12)
        np.testing.assert_allclose(old_state[7:13],new_state[7:13],atol=1e-12)
        self.assertAlmostEqual(energy(new)[-1],energy(new)[0],delta=1e-12)
        self.assertEqual(new['translation_pose_displacement_max_m'],0)
        self.assertLessEqual(new['coulomb_residual_max_m_s'],1e-8)
        self.assertLessEqual(new['translation_split_residual_max_m_s'],1e-8)
        self.assertEqual(new['early_component_policy']['attempts'],0)
        self.assertTrue(new['numerical_model']['early_component_recovery'])
        self.assertEqual(new['numerical_model']['shape_cache_margin_order'],'margin before recalc')
        for key in new:
            if key.startswith('translation_pose_'):
                self.assertEqual(prefix[key],new[key])

    def test_early_recovery_requires_a_boolean_and_enabled_coulomb_recovery(self):
        scene=floor_ball([0,0,0])
        for options in (dict(early_component_recovery=1),
                        dict(early_component_recovery=True,contact_recovery=False)):
            with self.assertRaises(ValueError):self.simulate(scene,**options)
        with self.assertRaises(ValueError):run(scene,early_component_recovery=True)
