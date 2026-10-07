import unittest
import numpy as np
from research.elastic_patch import Material, Plane, contact, simulate


class ElasticPatchTests(unittest.TestCase):
    def matched(self, **changes):
        values = dict(tangent_stiffness=1e5*2/7, twist_stiffness=400.,
                      friction=10., compression_exponent=0)
        values.update(changes)
        return Material(**values)

    def run_drop(self, material, **changes):
        values = dict(position=(0.,0.,.11), velocity=(0.,0.,-1.),
                      omega=(0.,0.,10.), duration=.04, sample_dt=.0002,
                      max_step=.0002, rtol=1e-10, atol=1e-12)
        values.update(changes)
        return simulate(material, **values)

    def assert_energy(self, result, tolerance=2e-8):
        self.assertLess(max(abs(result['energy_residual_J'])), tolerance)
        self.assertGreaterEqual(np.min(result['dissipated_J']), -tolerance)
        self.assertGreaterEqual(np.min(np.diff(result['dissipated_J'])), -tolerance)

    def test_normal_spin_reversal_requires_independent_couple(self):
        r=self.run_drop(self.matched())
        np.testing.assert_allclose(r['states'][-1,3:9],[0.,0.,1.,0.,0.,-10.], atol=1e-7)
        np.testing.assert_allclose(r['linear_impulse_N_s'][-1],[0.,0.,2.],atol=1e-7)
        np.testing.assert_allclose(r['couple_impulse_N_m_s'][-1],[0.,0.,-.08],atol=1e-8)
        self.assertGreater(np.max(r['history_stored_J']), .19)
        self.assertLess(r['stored_J'][-1], 1e-12)
        self.assert_energy(r)

    def test_tangent_spin_and_com_translation_match_exact_oscillator(self):
        m=self.matched(); r=self.run_drop(m,omega=(0.,10.,0.))
        np.testing.assert_allclose(r['states'][-1,3:9],[4/7,0.,1.,0.,-30/7,0.],atol=1e-7)
        np.testing.assert_allclose(r['couple_impulse_N_m_s'][-1],0.,atol=1e-12)
        # Angular momentum from the offset force is kept distinct from the couple.
        P=r['linear_impulse_N_s'][-1]
        np.testing.assert_allclose(m.inertia*(r['states'][-1,6:9]-[0.,10.,0.]),
                                   np.cross([0.,0.,-m.radius],P),atol=1e-8)
        self.assert_energy(r)

    def test_floor_and_ceiling_retrace_for_specific_oblique_spin(self):
        for ceiling in (False, True):
            plane=Plane((0.,0.,-1.),-.3,'ceiling') if ceiling else Plane()
            incoming=np.array([.2,0.,1. if ceiling else -1.])
            spin=np.array([0.,5. if ceiling else -5.,0.])
            z=.19 if ceiling else .11
            r=self.run_drop(self.matched(),planes=(plane,),position=(0.,0.,z),velocity=incoming,omega=spin)
            np.testing.assert_allclose(r['states'][-1,3:6],-incoming,atol=1e-7)
            np.testing.assert_allclose(r['states'][-1,6:9],-spin,atol=1e-7)
            self.assert_energy(r)

    def test_floor_ceiling_repeated_vertical_bounces_reverse_spin(self):
        for initial_sign in [-1.,1.]:
            r=self.run_drop(self.matched(),omega=(0.,0.,10*initial_sign),
                            planes=(Plane(),Plane((0.,0.,-1.),-.23,'ceiling')),duration=.18)
            lifts=[e for e in r['events'] if e['kind']=='lift_off']
            self.assertGreaterEqual(len(lifts),4)
            for i,e in enumerate(lifts):
                sign=(-1.)**i
                self.assertEqual(e['plane'],'floor' if i%2==0 else 'ceiling')
                self.assertAlmostEqual(e['velocity_m_s'][2],sign,places=6)
                self.assertAlmostEqual(e['omega_rad_s'][2],-10*sign*initial_sign,places=6)
                self.assertLess(e['separation_loss_J'],1e-10)
            self.assert_energy(r)

    def test_hybrid_yield_solves_high_spin_with_original_work_budget(self):
        for sign in [-1.,1.]:
            m=Material(friction=.5)
            r=self.run_drop(m,omega=(0.,0.,10*sign),max_rhs_evaluations=20000,
                            rtol=1e-12,atol=1e-14,max_step=.0001)
            self.assertLess(r['rhs_evaluations'],20000)
            self.assertEqual([e['kind'] for e in r['material_events']],['yield','release'])
            self.assertGreater(r['states'][-1,8]*sign,0.)
            self.assertLess(r['states'][-1,8]*sign,10.)
            self.assertGreater(r['dissipated_J'][-1],.1)
            L=r['couple_impulse_N_m_s'][-1]
            np.testing.assert_allclose(m.inertia*(r['states'][-1,6:9]-[0,0,10*sign]),L,atol=1e-10)
            self.assertLessEqual(abs(L[2]),m.friction*m.effective_length*r['linear_impulse_N_s'][-1,2]+1e-9)
            self.assertLess(r['max_yield_excess_N'],1e-5)
            self.assert_energy(r)

    def test_grazing_contacts_have_no_spurious_zero_time_transitions(self):
        for v in [(0.,0.,0.),(1.,0.,0.)]:
            r=self.run_drop(self.matched(),position=(0.,0.,.1),velocity=v,
                            duration=.02,max_rhs_evaluations=1)
            self.assertEqual(r['events'],[])
            self.assertEqual(r['rhs_evaluations'],0)
            np.testing.assert_allclose(r['states'][:,:3],np.array([0.,0.,.1])+r['times'][:,None]*v,atol=1e-15)
            np.testing.assert_allclose(r['couple_impulse_N_m_s'],0.,atol=1e-15)
            self.assert_energy(r)

    def test_ballistic_gravity_uses_exact_motion_without_contact_evaluations(self):
        g=np.array([0.,0.,-9.81]);v=np.array([2.,1.,0.]);x=np.array([0.,0.,1.])
        r=self.run_drop(self.matched(),position=x,velocity=v,gravity=g,duration=.05,max_rhs_evaluations=1)
        t=r['times'][:,None]
        np.testing.assert_allclose(r['states'][:,:3],x+t*v+.5*t*t*g,atol=1e-15)
        np.testing.assert_allclose(r['states'][:,3:6],v+t*g,atol=1e-15)
        self.assertEqual(r['rhs_evaluations'],0)
        self.assert_energy(r)

    def test_small_friction_does_not_reverse_spin(self):
        r=self.run_drop(self.matched(friction=.1),rtol=1e-12,atol=1e-14)
        self.assertAlmostEqual(r['states'][-1,8],9.,places=5)
        self.assertAlmostEqual(r['dissipated_J'][-1],.038,places=6)
        self.assertLess(r['max_yield_excess_N'],1e-5)
        self.assert_energy(r)

    def test_no_friction_preserves_spin(self):
        r=self.run_drop(self.matched(friction=0.))
        self.assertAlmostEqual(r['states'][-1,8],10.,places=8)
        np.testing.assert_allclose(r['couple_impulse_N_m_s'],0.,atol=1e-12)
        self.assert_energy(r)

    def test_compression_weighted_contact_releases_energy_conservatively(self):
        r=self.run_drop(Material(twist_stiffness=1e6,friction=100.))
        self.assertLess(r['states'][-1,8],-7.)
        self.assertGreater(r['states'][-1,5],1.)
        self.assertGreater(np.max(r['history_stored_J']),.1)
        self.assertLess(r['stored_J'][-1],1e-12)
        self.assert_energy(r)

    def test_compression_damping_dissipates_without_force_attraction(self):
        m=self.matched(normal_damping=100.)
        r=self.run_drop(m,omega=(0.,0.,0.))
        self.assertGreater(r['dissipated_J'][-1],.1)
        self.assertGreater(r['states'][-1,5],0.)
        self.assertLess(r['states'][-1,5],1.)
        self.assertGreaterEqual(np.min(r['normal_force_N']),0.)
        self.assert_energy(r)

    def test_refinement_converges_sliding_spin_and_energy(self):
        m=self.matched(friction=.1)
        values=[self.run_drop(m,max_step=step,rtol=tol,atol=tol*.01)
                for step,tol in [(.0008,1e-7),(.0004,1e-9),(.0002,1e-11)]]
        coarse=abs(values[0]['states'][-1,8]-9.)
        fine=abs(values[-1]['states'][-1,8]-9.)
        self.assertLess(fine,coarse/20)
        self.assertLess(fine,1e-6)
        self.assert_energy(values[-1])

    def test_combined_force_couple_capacity_is_shared(self):
        m=self.matched(friction=.2)
        r=self.run_drop(m,velocity=(.2,0.,-1.),omega=(0.,10.,10.),rtol=1e-12,atol=1e-14)
        both_active=False
        for state,h in zip(r['states'],r['strain']):
            F,L,_,_,_,N,_=contact(m,Plane(),state[:3],state[3:6],state[6:9],h)
            Ft=np.linalg.norm(F[:2]); twist=np.linalg.norm(L)/m.effective_length
            if Ft>1e-3 and twist>1e-3: both_active=True
            self.assertLessEqual(np.hypot(Ft,twist),m.friction*N+1e-5)
        self.assertTrue(both_active)
        self.assertGreater(r['dissipated_J'][-1],.01)
        self.assert_energy(r)

    def test_three_dimensional_rotation_covariance(self):
        # Rotate z-normal into x-normal and rotate all initial velocities.
        Q=np.array([[0.,0.,1.],[0.,1.,0.],[-1.,0.,0.]])
        m=self.matched()
        a=self.run_drop(m,velocity=(.2,.1,-1.),omega=(2.,-5.,10.))
        b=self.run_drop(m,planes=(Plane((1.,0.,0.)),),
                        position=Q@np.array([0.,0.,.11]),
                        velocity=Q@np.array([.2,.1,-1.]),omega=Q@np.array([2.,-5.,10.]))
        np.testing.assert_allclose(b['states'][-1,:3],Q@a['states'][-1,:3],atol=1e-7)
        np.testing.assert_allclose(b['states'][-1,3:6],Q@a['states'][-1,3:6],atol=1e-7)
        np.testing.assert_allclose(b['states'][-1,6:9],Q@a['states'][-1,6:9],atol=1e-7)
        self.assert_energy(b)

    def test_exhausted_accuracy_budget_rejects_result(self):
        with self.assertRaisesRegex(RuntimeError,'no accepted result'):
            self.run_drop(self.matched(),max_rhs_evaluations=1)

    def test_invalid_materials_and_planes_rejected(self):
        for field in ('mass','radius','normal_stiffness','effective_length'):
            with self.assertRaises(ValueError): Material(**{field:0})
        with self.assertRaises(ValueError): Material(compression_exponent=1)
        with self.assertRaises(ValueError): Plane((0.,0.,2.))


if __name__=='__main__': unittest.main()
