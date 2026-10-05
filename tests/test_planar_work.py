"""Boundary-work observer must include signed normal and tangent impulses."""
from pathlib import Path
import unittest
import numpy as np
from rigid_engine import run, BINARIES
from research.rigid_scenes import rectangle
from research.container_scenes import ball

@unittest.skipUnless(BINARIES['block'].exists(), 'Build the block backend')
class PlanarBoundaryWork(unittest.TestCase):
    def test_translation_work_equals_prescribed_velocity_dot_dynamic_impulse(self):
        for wall_first in (True, False):
            for friction in (0., .4):
                with self.subTest(wall_first=wall_first, friction=friction):
                    velocity=np.array([.7,.5])
                    wall={'type':'kinematic','position':[0,0],'velocity':velocity.tolist(),
                          'polygons':[rectangle(.01,.4,friction=friction)]}
                    particle=ball((.12,0),friction=friction)
                    scene={'duration':.01,'gravity':[0,0],'collision_skin_m':.01,
                           'bodies':[wall,particle] if wall_first else [particle,wall]}
                    result=run(scene,dt=.01,backend='block',primary_steps=8,substeps=32)
                    states=np.asarray(result['states'])
                    impulse=result['mass'][0]*(states[-1,0,3:5]-states[0,0,3:5])
                    self.assertGreater(impulse[0],.5)
                    if friction:self.assertGreater(impulse[1],.01)
                    expected=float(velocity@impulse)
                    self.assertAlmostEqual(result['boundary_work_J'],expected,delta=2e-6)
                    self.assertGreaterEqual(result['absolute_boundary_work_J'],abs(result['boundary_work_J'])-1e-9)
                    self.assertGreater(result['boundary_impulse_points'],0)

if __name__=='__main__':unittest.main()
