import copy
import unittest
import numpy as np

from rigid_engine import BINARIES, DEFAULT_BINARY, run, validate_scene
from research.rigid_scenes import body, make_scene, rectangle, scenes


class SceneValidationTests(unittest.TestCase):
    def test_concavity_and_nonfinite_properties_are_rejected(self):
        scene = make_scene("test", [body(rectangle())])
        bad = copy.deepcopy(scene)
        bad["bodies"][0]["polygons"][0]["vertices"] = [[0, 0], [1, 0], [.2, .2], [0, 1]]
        with self.assertRaises(ValueError): validate_scene(bad)
        bad = copy.deepcopy(scene); bad["bodies"][0]["omega"] = float("nan")
        with self.assertRaises(ValueError): validate_scene(bad)
        bad = copy.deepcopy(scene); bad["bodies"][0]["polygons"][0]["restitution"] = 1.1
        with self.assertRaises(ValueError): validate_scene(bad)

    def test_declared_scenes_are_valid(self):
        for scene in scenes():
            with self.subTest(case=scene["id"]): validate_scene(scene)


@unittest.skipUnless(DEFAULT_BINARY.is_file(), "Build the pinned rigid backend for integration checks")
class RigidIntegrationTests(unittest.TestCase):
    def test_free_motion_and_mass(self):
        scene = make_scene("flight", [body(rectangle(), velocity=(2, -1))], duration=1, gravity=(0, 0))
        result = run(scene)
        np.testing.assert_allclose(np.asarray(result["states"])[-1, 0, :2], [2, -1], atol=2e-5)
        self.assertAlmostEqual(result["mass"][0], 1)
        self.assertAlmostEqual(result["inertia"][0], 1/6, places=6)

    def test_isolated_rebound_and_internal_momentum(self):
        scene = next(c for c in scenes() if c["id"] == "normal_rebound_boxes")
        result = run(scene, primary_steps=4, substeps=16)
        state = np.asarray(result["states"])
        np.testing.assert_allclose(state[-1, :, 3], [-1.2, 1.2], atol=.02)
        np.testing.assert_allclose(state[-1, :, 5], 0, atol=.02)
        mass, inertia = np.asarray(result["mass"]), np.asarray(result["inertia"])
        kinetic = .5 * np.sum(mass * np.sum(state[:, :, 3:5]**2, axis=2) + inertia * state[:, :, 5]**2, axis=1)
        self.assertAlmostEqual(kinetic[-1] / kinetic[0], .6**2, delta=1e-5)
        momentum = np.sum(state[:, :, 3:5] * np.asarray(result["mass"])[None, :, None], axis=1)
        np.testing.assert_allclose(momentum, 0, atol=2e-5)

    def test_frictional_stop_and_ccd(self):
        scene = next(c for c in scenes() if c["id"] == "sliding_box")
        result = run(scene, primary_steps=4, substeps=16)
        final = np.asarray(result["states"])[-1, 0]
        self.assertLess(abs(final[3]), .005)
        self.assertAlmostEqual(final[0], 3**2/(2*.3*9.81), delta=.025)
        scene = next(c for c in scenes() if c["id"] == "thin_wall_ccd")
        result = run(scene, primary_steps=1, substeps=1)
        self.assertLess(np.max(np.asarray(result["states"])[:, 0, 0]), 0)

    def test_adaptation_preserves_world_and_physical_identity(self):
        scene = next(c for c in scenes() if c["id"] == "triangle_drop")
        fixed = run(scene)
        adaptive = run(scene, policy={})
        self.assertEqual(fixed["physical_setup_id"], adaptive["physical_setup_id"])
        self.assertEqual(sum(adaptive["level_frames"]), 360)
        self.assertGreater(adaptive["switches"], 0)
        self.assertGreater(adaptive["controller_s"], 0)
        np.testing.assert_allclose(fixed["mass"], adaptive["mass"])

    def test_frequency_clipping_is_rejected(self):
        scene = make_scene("bad", [body(rectangle())], duration=1, contact_hertz=100)
        with self.assertRaises(ValueError): run(scene, primary_steps=1, substeps=1, backend="temporal")

    @unittest.skipUnless(BINARIES["temporal"].is_file(), "Build the temporal comparator")
    def test_temporal_rebound_counterexample_is_reported(self):
        scene = next(c for c in scenes() if c["id"] == "normal_rebound_boxes")
        result = run(scene, primary_steps=4, substeps=16, backend="temporal")
        final = np.asarray(result["states"])[-1]
        # A documented adverse result must not be treated as exact restitution.
        self.assertGreater(abs(final[0, 3] + 1.2), .1)
        self.assertGreater(abs(final[0, 5]), .1)
