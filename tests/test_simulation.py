import unittest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy.testing import assert_allclose

from test_v3 import Simulation


class SimulationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def circle(self, **changes):
        obj = dict(type="circle", position=[0.5, 0.5], velocity=[1, 0], radius=0.05, density=1, color="white")
        obj.update(changes)
        return obj

    def test_animation_initialization_does_not_advance_physics(self):
        sim = Simulation(dt=0.05, g=0)
        sim.add_object(self.circle())
        before = sim.get_all("x").copy()
        sim.run()
        assert_allclose(sim.get_all("x"), before)
        sim.update(0)
        assert_allclose(sim.get_all("x"), [[0.55, 0.5]])
        assert_allclose(sim.graph_objects[0].center, [0.55, 0.5])
        assert_allclose(sim.get_all("xx"), np.sum(sim.get_all("x") ** 2, axis=1))

    def test_invalid_bodies_are_rejected(self):
        sim = Simulation(dt=0.05)
        for changes in [dict(radius=0), dict(density=-1), dict(position=[0, 0]), dict(velocity=[np.nan, 0]), dict(type="line_segment")]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                sim.add_object(self.circle(**changes))
        sim.add_object(self.circle())
        with self.assertRaises(ValueError):
            sim.add_object(self.circle())
        self.assertEqual(sim.end_index, 1)

    def test_invalid_settings_are_rejected(self):
        for settings in [dict(dt=0), dict(dt=-1), dict(dt=np.nan), dict(dt=0.1, e=1.1), dict(dt=0.1, g=-1)]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                Simulation(**settings)

    def test_empty_simulation(self):
        sim = Simulation(dt=0.05)
        self.assertEqual(sim.update(0), ())


if __name__ == "__main__":
    unittest.main()
