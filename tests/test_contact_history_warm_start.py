import unittest
import numpy as np
from contact_history import ContactHistory


class ContactHistoryWarmStartTests(unittest.TestCase):
    def test_changing_loads_reversals_and_zero_capacities_revalidate_hint(self):
        law=ContactHistory([[2.,.7,.1],[.7,1.,-.2],[.1,-.2,1.]],[1000.,200.,500.],.02)
        hint=(1,1,1)
        rng=np.random.default_rng(20261008)
        for k in range(150):
            u=rng.normal(size=3)*3;eta=rng.normal(size=3)*.03
            cd=10**rng.uniform(-3,0,3)
            if k%3==0:cd[k%3]=0
            cs=cd*1.5
            cold=law.step(u,eta,cs,cd)
            warm=law.step(u,eta,cs,cd,active_set=hint)
            np.testing.assert_allclose(warm.impulse,cold.impulse,rtol=1e-11,atol=1e-13)
            np.testing.assert_allclose(warm.motion,cold.motion,rtol=1e-11,atol=1e-13)
            self.assertAlmostEqual(warm.plastic_loss,cold.plastic_loss,places=10)
            hint=warm.active_set

    def test_hint_does_not_override_static_friction_or_opening(self):
        law=ContactHistory([[1.]],[1000.],.1)
        a=law.step([1.],[0.],[2.],[.1],active_set=(-1,))
        self.assertFalse(a.sliding)
        b=law.step([1.],[.1],[2.],[.1],active=False,active_set=(-1,))
        self.assertIsNone(b.active_set)
        np.testing.assert_array_equal(b.impulse,[0.])
        self.assertAlmostEqual(b.released_mode_energy,5.)

    def test_hint_is_caller_owned_and_no_cross_contact_state_is_retained(self):
        law=ContactHistory([[1.]],[1000.],.1)
        a=law.step([10.],[0.],[.1],[.1])
        b=law.step([-10.],[0.],[.1],[.1],active_set=a.active_set)
        self.assertEqual(a.active_set,(-1,));self.assertEqual(b.active_set,(1,))
        repeat=law.step([10.],[0.],[.1],[.1],active_set=b.active_set)
        np.testing.assert_array_equal(repeat.impulse,a.impulse)

    def test_warm_start_follows_actual_motion_history_and_recontact(self):
        law=ContactHistory([[2.,.3],[.3,1.]],[400.,800.],.02)
        cold_u=warm_u=np.array([3.,-2.]);cold_eta=warm_eta=np.array([.03,-.02]);hint=None
        for k in range(100):
            cd=np.array([.05,.01])*(1.+.3*np.sin(k));cs=cd*1.5
            if k%13==0:cd[1]=cs[1]=0
            active=k%17!=0
            a=law.step(cold_u,cold_eta,cs,cd,active=active)
            b=law.step(warm_u,warm_eta,cs,cd,active=active,active_set=hint)
            np.testing.assert_allclose(a.motion,b.motion,rtol=1e-11,atol=1e-12)
            np.testing.assert_allclose(a.history,b.history,rtol=1e-11,atol=1e-12)
            self.assertAlmostEqual(a.released_mode_energy,b.released_mode_energy,places=12)
            cold_u,cold_eta=a.motion,a.history;warm_u,warm_eta=b.motion,b.history;hint=b.active_set

    def test_singular_body_mobility_still_has_unique_regularized_mode_solve(self):
        law=ContactHistory(np.ones((3,3)),[100.,100.,100.],.01)
        cold=law.step([3.,-4.,2.],[0.,0.,0.],[.01]*3,[.005]*3)
        warm=law.step([3.,-4.,2.],[0.,0.,0.],[.01]*3,[.005]*3,active_set=(1,-1,0))
        np.testing.assert_allclose(warm.impulse,cold.impulse,atol=1e-14)

    def test_coupled_unconstrained_face_guess_can_be_wrong_and_must_fall_back(self):
        law=ContactHistory([[1.,.8],[.8,1.]],[4.,4.],1.)
        # The unconstrained solution suggests lower/upper. At that face the
        # upper mode violates KKT; the correct coupled solution is lower/free.
        result=law.step([1.,.1],[0.,0.],[.15,.15],[.1,.1],active_set=(1,1))
        np.testing.assert_allclose(result.impulse,[-.1,-.06],atol=1e-14)
        self.assertEqual(result.active_set,(-1,0))

    def test_bad_hints_and_mutating_prepared_coefficients_rejected(self):
        law=ContactHistory([[1.]],[1000.],.1)
        for hint in [[0],(3,),(),(True,)]:
            with self.assertRaises(ValueError):law.step([1.],[0.],[1.],[1.],active_set=hint)
        for array in [law.G,law.K,law.A,*law.factors.values()]:
            with self.assertRaises(ValueError):array.flat[0]=0
