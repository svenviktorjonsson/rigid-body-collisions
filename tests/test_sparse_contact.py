import unittest
import numpy as np
from scipy.optimize import LinearConstraint, minimize

from research.contact_solver import assemble_planar
from research.container_scenes import packed_row_contact_system
from research.sparse_contact import assemble_sparse, normal_solve, friction_solve


class SparseContactTests(unittest.TestCase):
    def test_sparse_geometry_matches_full_point_mobility_with_units_and_prescribed_spin(self):
        centers = [[0, .2], [1, -.1], [2, .3]]
        contacts = [(0, 1, [.5, 0], [-1, 0]), (1, 2, [1.5, 0], [.6, .8]), (0, 1, [.5, .4], [-1, 0])]
        for ell in (.1, 1, 10):
            inverse, G, K = assemble_planar(centers, [1, 2, np.inf], [.2, .5, np.inf], contacts, ell)
            system = assemble_sparse(centers, [1, 2, np.inf], [.2, .5, np.inf], contacts, ell)
            np.testing.assert_allclose(system.inverse_mass, inverse.diagonal())
            np.testing.assert_allclose(system.contact_map.toarray(), G)
            _, mobility = system.mobility((0, 1, 2))
            np.testing.assert_allclose(mobility.toarray(), K, atol=1e-12)

    def test_ten_thousand_ball_row_propagates_impulse_and_balances_actuator_work(self):
        system, velocity = packed_row_contact_system(10000, sparse=True)
        post, p, stats = normal_solve(system, velocity)
        np.testing.assert_allclose(post[:-3].reshape(-1, 3), np.tile([1, 0, 0], (10000, 1)), atol=1e-8)
        self.assertLessEqual(stats['active_normal_velocity_m_s'], 1e-8)
        self.assertLess(stats['normal_mobility_nnz'], 4 * 10001)
        self.assertAlmostEqual(p[0] - p[-3], 10000, delta=1e-5)
        kinetic = .5 * np.sum(post[:-3:3]**2)
        self.assertAlmostEqual(p[0] - p[-3] - kinetic, 5000, delta=1e-5)

    def test_redundant_contacts_do_not_require_artificial_softness(self):
        contact = (0, 1, [0, 0], [1, 0])
        system = assemble_sparse([[0, 0], [0, 0]], [1, np.inf], [1, np.inf], [contact, contact])
        post, p, stats = normal_solve(system, [-1, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(post, 0, atol=1e-10)
        self.assertAlmostEqual(p[0] + p[3], 1, places=10)
        self.assertGreater(stats['singular_solves'], 0)

    def test_separating_contacts_release_without_spurious_impulse(self):
        system = assemble_sparse([[0, 0], [0, 0]], [1, np.inf], [1, np.inf], [(0, 1, [0, 0], [1, 0])])
        velocity = np.array([2, .5, .7, 0, 0, 0])
        post, p, _ = normal_solve(system, velocity)
        np.testing.assert_allclose(post, velocity); np.testing.assert_allclose(p, 0)

    def test_incompatible_prescribed_walls_are_rejected(self):
        system = assemble_sparse([[0, 0]]*3, [1, np.inf, np.inf], [1, np.inf, np.inf],
                                 [(0, 1, [0, 0], [1, 0]), (0, 2, [0, 0], [-1, 0])])
        with self.assertRaises(RuntimeError): normal_solve(system, [0, 0, 0, 1, 0, 0, -1, 0, 0])

    def test_random_offcentre_systems_match_independent_primal_projection(self):
        rng = np.random.default_rng(481)
        for _ in range(30):
            centers = rng.normal(size=(4, 2))
            masses = np.r_[rng.uniform(.3, 3, 3), np.inf]
            inertias = np.r_[rng.uniform(.2, 2, 3), np.inf]
            contacts = [(int(rng.integers(3)), 3, rng.normal(size=2), rng.normal(size=2)) for _ in range(7)]
            system = assemble_sparse(centers, masses, inertias, contacts)
            velocity = rng.normal(size=12); velocity[-3:] = 0
            post, _, _ = normal_solve(system, velocity)
            N, _ = system.mobility(); N = N[:, :-3].toarray(); H = 1 / system.inverse_mass[:-3]
            reference = minimize(lambda z: .5*np.sum(H*(z-velocity[:-3])**2), np.zeros(9),
                jac=lambda z: H*(z-velocity[:-3]), constraints=[LinearConstraint(N, 0, np.inf)],
                method='SLSQP', options={'ftol': 1e-12, 'maxiter': 500})
            self.assertTrue(reference.success)
            np.testing.assert_allclose(post[:-3], reference.x, atol=3e-6)

    def test_internal_momentum_and_energy_are_preserved_or_dissipated(self):
        centers = np.array([[0, .2], [1, -.1], [2, .3]])
        mass, inertia = np.array([1, 2, 3]), np.array([.2, .5, .7]); ell = .3
        contacts = [(0, 1, [.5, 0], [-1, 0]), (1, 2, [1.5, 0], [-1, 0])]
        system = assemble_sparse(centers, mass, inertia, contacts, ell)
        velocity = np.array([1., .3, .2, 0, 0, 0, -1, -.2, .1])
        post, _, _ = normal_solve(system, velocity)
        change = (post-velocity).reshape(3, 3)
        linear = mass[:, None]*change[:, :2]
        np.testing.assert_allclose(linear.sum(axis=0), 0, atol=1e-10)
        angular = np.sum(centers[:, 0]*linear[:, 1]-centers[:, 1]*linear[:, 0]+inertia*change[:, 2]/ell)
        self.assertAlmostEqual(angular, 0, places=10)
        H = 1 / system.inverse_mass
        self.assertLessEqual(np.sum(H*(post**2-velocity**2)), 1e-9)

    def test_invalid_contact_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            assemble_sparse([[0, 0]], [1], [1], [(0, 0, [0, 0], [0, 0])])

    def test_no_contacts_preserve_free_motion_without_a_linear_solve(self):
        system = assemble_sparse([[0, 0]], [1], [1], [])
        post, impulse, stats = normal_solve(system, [1, 2, 3])
        np.testing.assert_allclose(post, [1, 2, 3])
        self.assertEqual(len(impulse), 0); self.assertEqual(stats['iterations'], 0)

    def test_friction_matches_isolated_sliding_and_sticking_impulses(self):
        system = assemble_sparse([[0, 0], [0, 0]], [1, np.inf], [.005, np.inf], [(0, 1, [-.1, 0], [1, 0])])
        for mu, expected_t in ((0, 0), (.1, -.1), (1, -2/3)):
            post, p, stats = friction_solve(system, [-1, 2, 0, 0, 0, 0], mu)
            self.assertAlmostEqual(p[0], 1, places=9)
            self.assertAlmostEqual(p[1], expected_t, places=9)
            self.assertLessEqual(stats['normal_residual_m_s'], 1e-8)
            self.assertLessEqual(stats['slip_law_residual_kg_m_s'], 1e-8)
            self.assertLessEqual(stats['contact_energy_change_minus_boundary_work_J'], 0)

    def test_frictional_moving_wall_work_and_post_contact_velocity(self):
        system = assemble_sparse([[0, 0], [0, 0]], [1, np.inf], [.005, np.inf], [(0, 1, [-.1, 0], [1, 0])])
        initial = np.array([0., 0, 0, 1, .2, 0])
        post, p, stats = friction_solve(system, initial, .4)
        np.testing.assert_allclose(system.contact_map @ post, [0, 0, post[2]], atol=1e-9)
        self.assertAlmostEqual(p[0], 1, places=9); self.assertAlmostEqual(p[1], .2/3, places=9)
        work = p[0]+.2*p[1]
        kinetic = .5*np.sum(post[:2]**2)+.5*.005*post[2]**2
        self.assertAlmostEqual(kinetic-work, stats['contact_energy_change_minus_boundary_work_J'], places=10)

    def test_friction_handles_general_normal_tangent_coupling_and_internal_momentum(self):
        centers = np.array([[0, .2], [1, -.1], [2, .3]])
        mass, inertia = np.array([1, 2, 3]), np.array([.2, .5, .7])
        system = assemble_sparse(centers, mass, inertia, [(0, 1, [.5, 0], [-1, 0]), (1, 2, [1.5, 0], [-1, 0])])
        _, K = system.mobility((0, 1))
        self.assertGreater(np.max(np.abs(K[::2, 1::2].toarray())), .01)
        initial = np.array([1., .3, .2, 0, 0, 0, -1, -.2, .1])
        post, p, stats = friction_solve(system, initial, .3)
        delta = (post-initial).reshape(3, 3); linear = mass[:, None]*delta[:, :2]
        np.testing.assert_allclose(linear.sum(axis=0), 0, atol=1e-10)
        angular = np.sum(centers[:, 0]*linear[:, 1]-centers[:, 1]*linear[:, 0]+inertia*delta[:, 2])
        self.assertAlmostEqual(angular, 0, places=10)
        self.assertLessEqual(stats['contact_energy_change_minus_boundary_work_J'], 1e-10)

    def test_large_frictional_row_satisfies_normal_cone_slip_and_energy_gates(self):
        system, v = packed_row_contact_system(1000, sparse=True)
        v[1:-3:3] = 2*np.sin(np.arange(1000)*.37); v[-2] = .2
        post, p, stats = friction_solve(system, v, .4)
        np.testing.assert_allclose(post[:-3:3], 1, atol=1e-8)
        self.assertLessEqual(stats['normal_residual_m_s'], 1e-8)
        self.assertLessEqual(stats['friction_capacity_residual_kg_m_s'], 1e-8)
        self.assertLessEqual(stats['slip_law_residual_kg_m_s'], 1e-8)
        self.assertLessEqual(stats['contact_energy_change_minus_boundary_work_J'], 0)

    def test_friction_free_flight_and_invalid_coefficients(self):
        system = assemble_sparse([[0, 0]], [1], [1], [])
        post, p, _ = friction_solve(system, [1, 2, 3], .4)
        np.testing.assert_allclose(post, [1, 2, 3]); self.assertEqual(len(p), 0)
        system, v = packed_row_contact_system(4, sparse=True)
        with self.assertRaises(ValueError): friction_solve(system, v, -.1)

    def test_dispatch_preserves_physics_and_general_coupling(self):
        system, v = packed_row_contact_system(100, sparse=True)
        outputs = [normal_solve(system, v, strategy=s)[0] for s in ('sparse', 'dense', 'auto')]
        for out in outputs: np.testing.assert_allclose(out, outputs[0], atol=1e-8)
        v[1:-3:3] = 2*np.sin(np.arange(100)*.37); v[-2] = .2
        a, pa, sa = friction_solve(system, v, .02, strategy='auto')
        b, pb, sb = friction_solve(system, v, .02, strategy='general')
        np.testing.assert_allclose(a, b, atol=1e-8); np.testing.assert_allclose(pa, pb, atol=1e-8)
        self.assertEqual(sa['method'], 'decoupled_box'); self.assertEqual(sb['method'], 'semismooth')
        self.assertGreater(sa['sliding_contacts'], 0)


if __name__ == '__main__': unittest.main()
