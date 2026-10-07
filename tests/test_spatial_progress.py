"""Verify crash evidence preserves accepted states without changing physics."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
import numpy as np
from spatial_engine import BINARY, run


@unittest.skipUnless(BINARY.exists(), 'Build spatial backend')
class SpatialProgress(unittest.TestCase):
    def test_recording_preserves_world_state_and_final_impulse(self):
        scene = dict(duration=.03, gravity=[0, 0, -9.81], bodies=[dict(
            position=[0, 0, 1], velocity=[.2, 0, -1], omega=[1, 2, 3],
            shapes=[dict(kind='box', half_extents=[.2, .1, .05], density=1000)])])
        options = dict(dt=.01, solver='coulomb', kinematic_contact_phase='start')
        plain = run(scene, **options)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'progress.json'
            recorded = run(scene, progress_checkpoint_path=path, **options)
            progress = json.loads(path.read_text())
        self.assertEqual(plain['states'], recorded['states'])
        self.assertEqual(plain['boundary_work_J'], recorded['boundary_work_J'])
        self.assertTrue(progress['complete'])
        self.assertEqual(progress['completed_output_frames'], 3)
        np.testing.assert_array_equal(progress['times'], recorded['times'])
        # Position and velocities are world coordinates in both representations;
        # backend quaternions explicitly use principal inertia axes.
        a = np.asarray(progress['states_backend']); b = np.asarray(recorded['states'])
        np.testing.assert_array_equal(a[:, :, :3], b[:, :, :3])
        np.testing.assert_array_equal(a[:, :, 7:], b[:, :, 7:])
        self.assertEqual(progress['orientation_frame'], 'backend principal inertia axes')

    def test_later_guard_rejection_keeps_prior_accepted_frames(self):
        wall = dict(type='kinematic', position=[0, 0, -2], shapes=[dict(
            kind='box', half_extents=[1, 1, .05])], velocity_schedule=[dict(
            time_s=.02, velocity=[1e6, 0, 0], omega=[0, 0, 0])])
        ball = dict(position=[0, 0, 1], shapes=[dict(kind='sphere', radius=.1)])
        scene = dict(duration=.03, gravity=[0, 0, 0], bodies=[wall, ball])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'partial.json'
            with self.assertRaises(subprocess.CalledProcessError) as failed:
                run(scene, dt=.01, solver='coulomb', kinematic_contact_phase='start',
                    travel_fraction=1e-7, progress_checkpoint_path=path)
            self.assertIn('Travel guard exhausted', failed.exception.stderr)
            progress = json.loads(path.read_text())
        self.assertFalse(progress['complete'])
        self.assertEqual(progress['completed_output_frames'], 2)
        np.testing.assert_array_equal(progress['times'], [0, .01, .02])
        self.assertTrue(np.isfinite(progress['states_backend']).all())
        self.assertEqual(progress['collision_updates'], 8)


if __name__ == '__main__':
    unittest.main()
