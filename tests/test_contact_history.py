import unittest
import numpy as np
from contact_history import ContactHistory


class ContactHistoryTests(unittest.TestCase):
    def test_elastic_stick_preserves_body_plus_stored_energy(self):
        law=ContactHistory([[2.,.2],[.2,1.]],[200.,100.],.01)
        u=np.array([1.,-.3]);eta=np.array([0.,.01]);inverse=np.linalg.inv(law.G)
        energy=.5*u@inverse@u+.5*law.K@(eta*eta)
        for _ in range(200):
            step=law.step(u,eta,[100,100],[80,80]);u=step.motion;eta=step.history
            self.assertFalse(step.sliding)
            self.assertAlmostEqual(.5*u@inverse@u+step.stored_energy,energy,places=12)

    def test_distinct_static_dynamic_capacities_and_passive_yield(self):
        law=ContactHistory([[1.]],[1000.],.1)
        stick=law.step([1.],[0.],[2.],[.1]);slide=law.step([1.],[0.],[.2],[.1])
        self.assertFalse(stick.sliding);self.assertTrue(slide.sliding)
        self.assertAlmostEqual(slide.impulse[0],-.1)
        self.assertGreater(slide.plastic_loss,0.)
        self.assertAlmostEqual(slide.energy_residual,0.,places=13)

    def test_zero_capacity_and_detachment_do_not_apply_phantom_impulses(self):
        law=ContactHistory([[1.]],[1000.],.1)
        zero=law.step([2.],[0.],[0.],[0.])
        np.testing.assert_array_equal(zero.impulse,[0.]);np.testing.assert_array_equal(zero.motion,[2.])
        opened=law.step([2.],[.1],[10.],[8.],active=False)
        np.testing.assert_array_equal(opened.impulse,[0.]);np.testing.assert_array_equal(opened.history,[0.])
        self.assertAlmostEqual(opened.released_mode_energy,5.)
        self.assertEqual(opened.plastic_loss,0.)

    def test_history_release_can_change_velocity_without_refitting_restitution(self):
        law=ContactHistory([[1.]],[1000.],.05)
        step=law.step([0.],[.1],[10.],[10.])
        self.assertLess(step.motion[0],0.)
        self.assertAlmostEqual(.5*step.motion[0]**2+step.stored_energy,5.,places=12)

    def test_coupled_modes_permutations_and_300_random_passivity_cases(self):
        rng=np.random.default_rng(20261007)
        for _ in range(300):
            A=rng.normal(size=(3,3));G=A@A.T;K=10**rng.uniform(0,4,3);h=10**rng.uniform(-3,-1)
            u=rng.normal(size=3);eta=rng.normal(size=3)*.01;dynamic=10**rng.uniform(-3,0,3);static=dynamic*1.5
            law=ContactHistory(G,K,h);a=law.step(u,eta,static,dynamic)
            perm=np.array([2,0,1]);b=ContactHistory(G[np.ix_(perm,perm)],K[perm],h).step(u[perm],eta[perm],static[perm],dynamic[perm])
            np.testing.assert_allclose(b.impulse,a.impulse[perm],rtol=1e-10,atol=1e-12)
            self.assertGreaterEqual(a.plastic_loss,-1e-10)
            self.assertLess(abs(a.energy_residual),1e-9)

    def test_invalid_mobility_and_capacity_rejected(self):
        with self.assertRaises(ValueError):ContactHistory([[1,1],[0,1]],[1,1],.1)
        with self.assertRaises(ValueError):ContactHistory([[-1]],[1],.1)
        with self.assertRaises(ValueError):ContactHistory([[1]],[0],.1)
        with self.assertRaises(ValueError):ContactHistory([[1]],[1],.1).step([1],[0],[.1],[.2])
