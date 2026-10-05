from pathlib import Path
import unittest

import numpy as np

from rigid_engine import run, BINARIES
from research.build_precision_backend import convert
from research.random_shapes import scenes
from research.container_scenes import container
from research.rigid_scenes import body, rectangle, make_scene


class TransformTests(unittest.TestCase):
    def test_scalar_transform_preserves_headers_comments_and_strings(self):
        original = '#include <float.h>\n// float 0.5f\nconst char* s="float .3f";\nfloat x=sqrtf(0.5f);'
        result = convert(original)
        self.assertIn('#include <float.h>', result)
        self.assertIn('// float 0.5f', result)
        self.assertIn('"float .3f"', result)
        self.assertIn('double x=sqrt(0.5);', result)


@unittest.skipUnless(BINARIES['block'].is_file(), 'Build block comparator')
class NumericalControlsTests(unittest.TestCase):
    def test_analytic_prescribed_wall_path_does_not_accumulate_float32_drift(self):
        scene = make_scene('boundary', [container(2,2,velocity=(.6,0)),
            body(rectangle(.1,.1), velocity=(.6,0))], duration=.5, gravity=(0,0))
        scene['analytic_kinematics'] = True
        paths=[]
        for primary in (16,128,512):
            result=run(scene,primary_steps=primary,substeps=8)
            path=np.asarray(result['kinematic_states'])[:,0,:2]; paths.append(path)
            np.testing.assert_allclose(path[:,0], .6*np.asarray(result['times']), atol=5e-8)
            self.assertEqual(result['numerical_model']['analytic_kinematics'], True)
        np.testing.assert_array_equal(paths[0], paths[1]); np.testing.assert_array_equal(paths[1], paths[2])

    def test_pose_iterations_are_explicit_and_do_not_change_authored_physics(self):
        scene=make_scene('flight', [body(rectangle(), (0,3))], duration=.1)
        a=run(scene,position_iterations=3); b=run(scene,position_iterations=12)
        self.assertEqual(a['physical_setup_id'],b['physical_setup_id'])
        self.assertEqual(b['numerical_model']['position_iterations'],12)
        np.testing.assert_array_equal(a['states'],b['states'])
        with self.assertRaises(ValueError): run(scene,position_iterations=0)
        with self.assertRaises(ValueError): run(scene,backend='temporal',position_iterations=12)

    @unittest.skipUnless(BINARIES['temporal'].is_file(), 'Build temporal comparator')
    def test_union_filter_keeps_exterior_support_and_reports_hidden_contacts(self):
        scene=next(s for s in scenes() if s['id']=='random_42_concave_drop')
        scene['suppress_internal_edges']=True
        result=run(scene,primary_steps=8,substeps=32,backend='temporal')
        self.assertGreater(result['internal_contact_points_removed'],0)
        state=np.asarray(result['states'])[:,0]
        self.assertGreater(np.min(state[:,1]),.15)
        self.assertLess(np.linalg.norm(state[-1,3:]),.002)
        self.assertLess(np.max(np.abs(np.asarray(result['mass'])-1)),2e-6)


DOUBLE = Path(__file__).resolve().parents[1]/'build/rigid_double/rigid_runner'


@unittest.skipUnless(DOUBLE.is_file(), 'Build explicit Float64 diagnostic')
class Float64DiagnosticTests(unittest.TestCase):
    def test_free_fall_precision_is_not_just_output_conversion(self):
        scene=make_scene('precision',[body(rectangle(),(0,5))],duration=1)
        result=run(scene,primary_steps=512,substeps=8,binary=DOUBLE)
        self.assertEqual(result['numerical_model']['scalar_precision'],'float64')
        self.assertIn('precision_source_sha256',result['numerical_model'])
        self.assertLess(abs(result['states'][-1][0][4]+9.81),1e-10)
        self.assertAlmostEqual(result['mass'][0],1.,places=12)
        self.assertAlmostEqual(result['inertia'][0],1/6,places=12)

    def test_double_isolated_rebound_has_correct_velocity_and_no_spin(self):
        scene=make_scene('rebound',[body(rectangle(friction=0,restitution=.6),(-1.5,0),(2,0)),
            body(rectangle(friction=0,restitution=.6),(1.5,0),(-2,0))],duration=1.2,gravity=(0,0))
        result=run(scene,primary_steps=4,substeps=32,binary=DOUBLE)
        state=np.asarray(result['states'])
        np.testing.assert_allclose(state[-1,:,3],[-1.2,1.2],atol=1e-10)
        np.testing.assert_allclose(state[-1,:,5],0,atol=1e-10)


if __name__ == '__main__': unittest.main()
