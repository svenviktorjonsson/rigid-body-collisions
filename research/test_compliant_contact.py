import unittest
import numpy as np
from compliant_contact import elastic_slider,integrate_contact,calibrate_normal_damping

class CompliantTests(unittest.TestCase):
    def test_slider_local_passivity(self):
        rng=np.random.default_rng(4)
        for _ in range(500):
            rate,state=rng.normal(size=2);normal=rng.uniform(0,10)
            force,zrate,D,plastic=elastic_slider(rate,state,50,3,normal,.6,.4)
            self.assertGreaterEqual(D,-1e-10)
            self.assertAlmostEqual(force*rate+50*state*zrate,-D,places=9)
            self.assertLessEqual(abs(force),.6*normal+1e-10)

    def test_normal_restitution_is_calibrated(self):
        kn=10000.;e=.6;cn=calibrate_normal_damping(e,kn)
        result=integrate_contact(np.eye(3),[-1,0,0],kn,cn,100,2,100,2,.6,.4,.03,.02)
        self.assertAlmostEqual(result['effective_normal_restitution'],e,places=6)
        self.assertLess(np.max(np.abs(result['energy_accounting_residual'])),1e-6)

    def test_coupled_total_energy_accounting(self):
        K=np.array([[2,-.3,.2],[-.3,3,-.4],[.2,-.4,4.]])
        result=integrate_contact(K,[-1,.7,2],10000,80,4000,20,30,2,.6,.4,.03,.02)
        self.assertLess(np.max(np.abs(result['energy_accounting_residual'])),2e-6)
        initial=result['kinetic'][0]
        self.assertLessEqual(result['kinetic'][-1]+result['stored'][-1],initial+2e-6)
        self.assertGreaterEqual(result['dissipation'][-1],0)

    def test_invalid_mobility_is_rejected(self):
        for K in (np.diag([1,1,-1]),np.array([[1,1,0],[0,1,0],[0,0,1.]])):
            with self.assertRaises(ValueError):
                integrate_contact(K,[-1,.7,2],10000,80,4000,20,30,2,.6,.4,.03,.02)

    def test_retained_slip_modes_and_transition_convergence(self):
        K=np.array([[2,-.3,.2],[-.3,3,-.4],[.2,-.4,4.]])
        args=(K,[-1,.7,2],10000,36.1716614,4000,20,30,2,.6,.4,.03,.02)
        coarse=integrate_contact(*args,rtol=1e-9)
        fine=integrate_contact(*args,rtol=1e-10)
        self.assertLessEqual(len(fine['phase_history']),10)
        self.assertTrue(any(mode['modes'][0]!=0 for mode in fine['phase_history']))
        self.assertTrue(any(mode['modes'][1]!=0 for mode in fine['phase_history']))
        np.testing.assert_allclose(coarse['post_velocity'],fine['post_velocity'],atol=2e-7)
        self.assertLess(np.max(np.abs(fine['energy_accounting_residual'])),2e-6)

if __name__=='__main__':unittest.main()
