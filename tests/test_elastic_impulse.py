import unittest
import numpy as np
from research.elastic_patch import Material,Plane,simulate
from research.elastic_impulse import impact,resolve_impact,UnsupportedImpact


class ElasticImpulseTests(unittest.TestCase):
    def material(self,**changes):
        values=dict(normal_stiffness=1e5,tangent_stiffness=1e5*2/7,
                    twist_stiffness=400.,friction=3.,compression_exponent=0)
        values.update(changes);return Material(**values)

    def test_closed_form_torsion_and_tangential_exact_endpoints(self):
        for v,w,expected_v,expected_w in [([0,0,-1],[0,0,10],[0,0,1],[0,0,-10]),
                ([0,0,-1],[0,10,0],[4/7,0,1],[0,-30/7,0]),
                ([.2,0,-1],[0,-5,0],[-.2,0,1],[0,5,0])]:
            result=impact(self.material(),[0,0,1],v,w)
            np.testing.assert_allclose(result.outgoing_velocity,expected_v,atol=1e-14)
            np.testing.assert_allclose(result.outgoing_omega,expected_w,atol=1e-13)
            self.assertLess(abs(result.energy_residual_J),1e-14)
        self.assertAlmostEqual(result.duration_s,np.pi/np.sqrt(1e5),places=14)

    def test_whole_contact_trajectory_matches_refined_material_integration(self):
        m=self.material();v=np.array([.2,.1,-1.]);w=np.array([2.,-5.,10.])
        exact=impact(m,[0,0,1],v,w)
        # Start already touching; explicit event initialization tests the full
        # history trajectory, rather than merely matching chosen endpoints.
        result=simulate(m,position=[0,0,m.radius],velocity=v,omega=w,
                        duration=exact.duration_s,sample_dt=exact.duration_s/100,
                        max_step=exact.duration_s/200,rtol=1e-12,atol=1e-14)
        closed=exact.trajectory(result['times'],position=[0,0,m.radius])
        np.testing.assert_allclose(result['states'][:,:3],closed['position'],atol=1e-9)
        np.testing.assert_allclose(result['states'][:,3:6],closed['velocity'],atol=1e-7)
        np.testing.assert_allclose(result['states'][:,6:9],closed['omega'],atol=1e-6)
        np.testing.assert_allclose(result['stored_J'],closed['normal_stored_J']+closed['tangent_stored_J']+closed['twist_stored_J'],atol=1e-8)
        np.testing.assert_allclose(result['couple_impulse_N_m_s'][-1],exact.independent_angular_impulse,atol=1e-8)

    def test_force_and_torque_share_instantaneous_capacity(self):
        m=self.material();result=impact(m,[0,0,1],[.3,.2,-1.],[2.,-3.,10.])
        trace=result.trajectory(np.linspace(0,result.duration_s,301))
        Fn=trace['force_N'][:,2]
        Ft=np.linalg.norm(trace['force_N'][:,:2],axis=1)
        twist=np.linalg.norm(trace['independent_couple_N_m'],axis=1)/m.effective_length
        self.assertLessEqual(np.max(np.hypot(Ft,twist)-m.friction*Fn),1e-12)
        np.testing.assert_allclose(trace['kinetic_J']+trace['normal_stored_J']+trace['tangent_stored_J']+trace['twist_stored_J'],trace['kinetic_J'][0],atol=1e-14)

    def test_fast_path_rejects_unsupported_material_and_states(self):
        for m in [self.material(normal_damping=1),self.material(compression_exponent=2),
                  self.material(twist_stiffness=401),self.material(friction=.1)]:
            with self.assertRaises(UnsupportedImpact):impact(m,[0,0,1],[0,0,-1],[0,0,10])
        for options in [dict(gravity=[0,0,-9.81]),dict(history=[.01,0,0])]:
            with self.assertRaises(UnsupportedImpact):impact(self.material(),[0,0,1],[0,0,-1],[0,0,10],**options)
        with self.assertRaises(UnsupportedImpact):impact(self.material(),[0,0,1],[0,0,1],[0,0,10])
        with self.assertRaises(UnsupportedImpact):impact(self.material(),[0,0,1],[0,0,-100],[0,0,0])

    def test_dispatcher_preserves_material_and_falls_back_to_elastic_history(self):
        fast=resolve_impact(self.material(),[0,0,1],[0,0,-1],[0,0,10])
        self.assertEqual(fast['method'],'exact-matched-elastic')
        material=Material(friction=1.)
        resolved=resolve_impact(material,[0,0,1],[0,0,-1],[0,0,1])
        self.assertEqual(resolved['method'],'resolved-compliant')
        self.assertIn('weighted',resolved['exact_rejection'])
        # Independently refined, archived weighted material benchmark.
        self.assertAlmostEqual(resolved['omega'][2],-1.3268,places=4)
        self.assertAlmostEqual(resolved['velocity'][2],.998478,places=5)
        self.assertLess(resolved['energy_residual_J'],1e-8)
        self.assertLess(resolved['dissipated_J'],1e-9)
        self.assertEqual(material.compression_exponent,2)

    def test_same_floor_bounces_reverse_horizontal_motion_and_spin(self):
        m=self.material(normal_stiffness=1e8,tangent_stiffness=1e8*2/7,twist_stiffness=4e5)
        r=simulate(m,position=[0,0,.1001],velocity=[.2,0,-1.],omega=[0,-5.,0],
                   gravity=[0,0,-9.81],duration=.46,sample_dt=.0005,
                   max_step=.0001,rtol=1e-11,atol=1e-13)
        lifts=[e for e in r['events'] if e['kind']=='lift_off']
        self.assertGreaterEqual(len(lifts),3)
        for i,e in enumerate(lifts):
            sign=(-1)**i
            self.assertAlmostEqual(e['velocity_m_s'][0],-.2*sign,places=4)
            self.assertAlmostEqual(e['omega_rad_s'][1],5*sign,places=4)
        self.assertLess(max(abs(r['energy_residual_J'])),1e-8)
        # Gravity perturbs the contact half-period, so a small residual
        # shear store is dissipated explicitly rather than silently deleted.
        self.assertGreaterEqual(r['dissipated_J'][-1],0.)
        self.assertLess(r['dissipated_J'][-1],1e-5)

    def test_same_material_slow_rapid_with_compression_budget(self):
        m=self.material(normal_stiffness=1e8,tangent_stiffness=1e8*2/7,twist_stiffness=4e5)
        for speed in [.01,100.]:
            result=impact(m,[0,0,1],[0,0,-speed],[0,0,10*speed])
            np.testing.assert_allclose(result.outgoing_velocity,[0,0,speed],atol=1e-12)
            np.testing.assert_allclose(result.outgoing_omega,[0,0,-10*speed],atol=1e-12)
            self.assertLessEqual(result.peak_compression_m,.1*m.radius+1e-15)


if __name__=='__main__':unittest.main()
