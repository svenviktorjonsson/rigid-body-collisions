import unittest

import numpy as np
from numpy.testing import assert_allclose

from physics import advance_disks


class CollisionTests(unittest.TestCase):
    def run_scene(self, positions, velocities, dt, masses=None, radius=0.05, e=1, g=0):
        x = np.array(positions, dtype=float)
        v = np.array(velocities, dtype=float)
        r = np.full(len(x), radius)
        m = np.ones(len(x)) if masses is None else np.array(masses, dtype=float)
        advance_disks(x, v, r, m, dt, e, g)
        return x, v

    def test_equal_mass_head_on(self):
        x, v = self.run_scene([[0.3, 0.5], [0.7, 0.5]], [[1, 0], [-1, 0]], 0.2)
        assert_allclose(x, [[0.4, 0.5], [0.6, 0.5]])
        assert_allclose(v, [[-1, 0], [1, 0]])

    def test_unequal_masses(self):
        _, v = self.run_scene([[0.2, 0.5], [0.4, 0.5]], [[1, 0], [0, 0]], 0.2, [1, 3])
        assert_allclose(v, [[-0.5, 0], [0.5, 0]])
        assert_allclose(np.sum(v * [[1], [3]], axis=0), [1, 0])
        assert_allclose(np.sum(v * v * [[1], [3]]) / 2, 0.5)

    def test_oblique_collision_conserves_momentum_and_energy(self):
        initial = np.array([[1.0, 0.2], [-0.5, -0.1]])
        m = np.array([2.0, 3.0])
        _, v = self.run_scene([[0.3, 0.4], [0.6, 0.49]], initial, 0.2, m)
        self.assertFalse(np.allclose(v, initial))
        assert_allclose(np.sum(m[:, None] * v, axis=0), np.sum(m[:, None] * initial, axis=0))
        assert_allclose(np.sum(m[:, None] * v ** 2), np.sum(m[:, None] * initial ** 2))

    def test_chain_in_same_step(self):
        x, v = self.run_scene(
            [[0.2, 0.5], [0.4, 0.5], [0.6, 0.5]], [[1, 0], [0, 0], [0, 0]], 0.3
        )
        assert_allclose(v, [[0, 0], [0, 0], [1, 0]])
        assert_allclose(x, [[0.3, 0.5], [0.5, 0.5], [0.7, 0.5]])

    def test_fast_disk_does_not_tunnel(self):
        _, v = self.run_scene([[0.3, 0.5], [0.7, 0.5]], [[10, 0], [-10, 0]], 0.025)
        assert_allclose(v, [[-10, 0], [10, 0]])

    def test_multiple_wall_hits(self):
        x, v = self.run_scene([[0.5, 0.5]], [[4, 0]], 1)
        assert_allclose(x, [[0.9, 0.5]])
        assert_allclose(v, [[4, 0]])

    def test_corner_at_step_endpoint(self):
        x, v = self.run_scene([[0.5, 0.5]], [[1, 1]], 0.375, radius=0.125)
        assert_allclose(x, [[0.875, 0.875]])
        assert_allclose(v, [[-1, -1]])

    def test_touching_approaching_disks(self):
        _, v = self.run_scene([[0.25, 0.5], [0.5, 0.5]], [[1, 0], [-1, 0]], 0.01, radius=0.125)
        assert_allclose(v, [[-1, 0], [1, 0]])

    def test_touching_separating_disks(self):
        _, v = self.run_scene([[0.25, 0.5], [0.5, 0.5]], [[-1, 0], [1, 0]], 0.01, radius=0.125)
        assert_allclose(v, [[-1, 0], [1, 0]])

    def test_pair_at_step_endpoint(self):
        _, v = self.run_scene([[0.25, 0.5], [0.625, 0.5]], [[1, 0], [-1, 0]], 0.125, radius=0.0625)
        assert_allclose(v, [[-1, 0], [1, 0]])

    def test_inelastic_pair(self):
        _, v = self.run_scene([[0.3, 0.5], [0.7, 0.5]], [[1, 0], [-1, 0]], 0.2, e=0.5)
        assert_allclose(v, [[-0.5, 0], [0.5, 0]])

    def test_inelastic_wall(self):
        x, v = self.run_scene([[0.8, 0.5]], [[1, 0]], 0.3, e=0.5)
        assert_allclose(v, [[-0.5, 0]])
        assert_allclose(x, [[0.875, 0.5]])

    def test_stationary_disks(self):
        x, v = self.run_scene([[0.2, 0.5], [0.8, 0.5]], [[0, 0], [0, 0]], 1)
        assert_allclose(x, [[0.2, 0.5], [0.8, 0.5]])
        assert_allclose(v, 0)

    def test_gravity_free_flight(self):
        x, v = self.run_scene([[0.5, 0.7]], [[0.2, 0.1]], 0.1, g=9.82)
        assert_allclose(x, [[0.52, 0.7 + 0.01 - 0.5 * 9.82 * 0.1 ** 2]])
        assert_allclose(v, [[0.2, 0.1 - 9.82 * 0.1]])

    def test_long_run_stays_inside_without_energy_drift(self):
        rng = np.random.default_rng(2026)
        x = np.array([[a, b] for a in np.linspace(0.15, 0.85, 5) for b in np.linspace(0.15, 0.85, 5)])
        v = rng.uniform(-2, 2, x.shape)
        r = np.full(len(x), 0.04)
        m = rng.uniform(0.5, 3, len(x))
        energy = np.sum(m[:, None] * v ** 2) / 2
        first, second = np.triu_indices(len(x), 1)
        for _ in range(300):
            advance_disks(x, v, r, m, 0.05)
            self.assertTrue(np.all(x >= r[:, None] - 1e-10))
            self.assertTrue(np.all(x <= 1 - r[:, None] + 1e-10))
            distances = np.linalg.norm(x[first] - x[second], axis=1)
            self.assertTrue(np.all(distances >= r[first] + r[second] - 1e-10))
        assert_allclose(np.sum(m[:, None] * v ** 2) / 2, energy, rtol=1e-11)


if __name__ == "__main__":
    unittest.main()
