"""Regression checks for the user's two-coefficient impact law."""
import unittest
import numpy as np
from spatial_engine import BINARY, run, energy
from research.spatial_scenes import sphere

@unittest.skipUnless(BINARY.is_file(), 'Build the spatial backend')
class TwoChannelRestitution(unittest.TestCase):
    def impact(self, en, et, friction=1., wall_velocity=(0,0,0)):
        scene={'duration':1e-6,'gravity':[0,0,0], 'bodies':[
            {'type':'kinematic','position':[0,0,-.1],'velocity':list(wall_velocity),'friction':1.,'shapes':[{'kind':'box','half_extents':[2,2,.1]}]},
            sphere([0,0,.1-1e-12],velocity=[1,0,-1],friction=friction)]}
        return run(scene,dt=1e-6,primary_steps=1,iterations=4096,travel_fraction=0,
                   solver='coulomb',kinematic_contact_phase='start',position_stabilization='split_translation_combined',
                   normal_restitution=en,tangential_restitution=et,record_contact_impacts=True)

    def test_positive_tangential_restitution_reverses_contact_slip(self):
        result=self.impact(.78,.49)
        c=result['restitution_contact_impacts'][0]
        before=np.array(c['contact_velocity_before_normal_tangent_m_s'])
        after=np.array(c['contact_velocity_after_normal_tangent_m_s'])
        np.testing.assert_allclose(after, -np.array([.78,.49,.49])*before,atol=1e-8)
        self.assertLessEqual(energy(result)[-1]-energy(result)[0],1e-8)

    def test_friction_limit_can_prevent_requested_tangential_rebound(self):
        result=self.impact(.78,.49,friction=.05)
        c=result['restitution_contact_impacts'][0];p=np.array(c['impulse_world_kg_m_s']);n=np.array(c['normal']);pn=p@n
        self.assertAlmostEqual(np.linalg.norm(p-pn*n),.05*pn,places=8)
        before=np.array(c['contact_velocity_before_normal_tangent_m_s']);after=np.array(c['contact_velocity_after_normal_tangent_m_s'])
        self.assertGreater(np.linalg.norm(after[1:]+.49*before[1:]),.1)

    def test_elastic_both_channels_preserve_pair_energy_and_momentum(self):
        scene={'duration':1e-6,'gravity':[0,0,0],'bodies':[
            sphere([-.1+5e-13,0,0],velocity=[1,.3,.2],friction=2.),
            sphere([.1-5e-13,0,0],velocity=[-1,-.3,-.2],friction=2.)]}
        result=run(scene,dt=1e-6,primary_steps=1,iterations=4096,travel_fraction=0,solver='coulomb',kinematic_contact_phase='start',position_stabilization='velocity_only',normal_restitution=1,tangential_restitution=1)
        states=np.array(result['states']);m=np.array(result['mass']);mom=np.einsum('i,tij->tj',m,states[:,:,7:10])
        np.testing.assert_allclose(mom[-1],mom[0],atol=1e-8)
        self.assertAlmostEqual(energy(result)[-1],energy(result)[0],places=8)

    def test_translating_wall_energy_includes_actuator_work(self):
        result=self.impact(.78,.49,wall_velocity=(.5,0,.1))
        c=result['restitution_contact_impacts'][0];p=np.array(c['impulse_world_kg_m_s'])*(1 if c['body_a']==1 else -1)
        self.assertAlmostEqual(result['boundary_work_J'],float(p@np.array([.5,0,.1])),places=8)
        self.assertLessEqual(energy(result)[-1]-energy(result)[0]-result['boundary_work_J'],1e-8)
