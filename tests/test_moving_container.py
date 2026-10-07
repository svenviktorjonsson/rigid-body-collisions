import copy
import unittest
import numpy as np
from rigid_engine import BINARIES, run, validate_scene
from research.container_scenes import ball, container, grid, packed_row_contact_system
from research.contact_solver import inelastic_normal_solve


class ContainerValidationTests(unittest.TestCase):
    def test_circles_and_prescribed_boundary_validation(self):
        scene=grid(side=2,duration=1)
        validate_scene(scene)
        bad=copy.deepcopy(scene);bad['bodies'][1]['circles'][0]['radius']=-1
        with self.assertRaises(ValueError):validate_scene(bad)
        bad=copy.deepcopy(scene);bad['bodies'][1]['velocity_schedule']=[{'time_s':0,'velocity':[1,0]}]
        with self.assertRaises(ValueError):validate_scene(bad)
        bad=copy.deepcopy(scene);bad['bodies'][0]['velocity_schedule']=[{'time_s':.1},{'time_s':.05}]
        with self.assertRaises(ValueError):validate_scene(bad)

    def test_global_driven_contact_projection_moves_every_ball_and_accounts_for_work(self):
        for count in (4,16,64,100):
            with self.subTest(count=count):
                inverse,G,velocity=packed_row_contact_system(count)
                post,impulse,residual=inelastic_normal_solve(inverse,G,velocity)
                np.testing.assert_allclose(post[:-3].reshape(count,3),np.tile([1,0,0],(count,1)),atol=1e-8)
                self.assertLess(residual['complementarity_residual'],1e-7)
                wall_impulse=impulse[0]-impulse[-3]
                self.assertAlmostEqual(wall_impulse,count,delta=1e-7)
                kinetic=.5*np.sum(post[:-3:3]**2)
                self.assertAlmostEqual(wall_impulse-kinetic,count/2,delta=1e-7)


@unittest.skipUnless(BINARIES['block'].is_file(),'Build circle/kinematic backend')
class ContainerIntegrationTests(unittest.TestCase):
    def test_analytic_disk_mass_inertia_and_boosted_free_motion(self):
        r=run({'duration':.25,'gravity':[0,0],'bodies':[ball((0,0),radius=.2,velocity=(1,2))]})
        self.assertAlmostEqual(r['mass'][0],1,places=6)
        self.assertAlmostEqual(r['inertia'][0],.02,places=6)
        np.testing.assert_allclose(np.asarray(r['states'])[-1,0,:2],[.25,.5],atol=1e-5)

    def test_moving_wall_restitution_and_boundary_work(self):
        scene={'duration':.25,'gravity':[0,0],'bodies':[container(1,1,velocity=(1,0),friction=0,restitution=.5),
            ball((-.7,0),friction=0,restitution=.5)]}
        r=run(scene);a=np.asarray(r['states']);b=np.asarray(r['kinematic_states'])
        self.assertAlmostEqual(a[-1,0,3],1.5,delta=.01)
        self.assertAlmostEqual(a[-1,0,5],0,delta=1e-5)
        self.assertAlmostEqual(b[-1,0,0],.25,delta=1e-5)
        # For U=1, J=m*delta v=1.5. W=UJ=1.5, delta K=1.125, D=.375.
        work=a[-1,0,3];kinetic=.5*a[-1,0,3]**2
        self.assertAlmostEqual(work-kinetic,.375,delta=.01)

    def test_galilean_covariance_with_many_balls(self):
        scene=grid(side=6,motion='translate',duration=.25,gravity=(0,0),friction=0)
        boosted=grid(side=6,motion='translate',duration=.25,boost=(.4,-.2),gravity=(0,0),friction=0)
        a,b=run(scene),run(boosted);t=np.asarray(a['times']);sa,sb=np.asarray(a['states']),np.asarray(b['states'])
        np.testing.assert_allclose(sb[:,:,:2]-np.asarray([.4,-.2])[None,None,:]*t[:,None,None],sa[:,:,:2],atol=.003)
        np.testing.assert_allclose(sb[:,:,3:5]-[.4,-.2],sa[:,:,3:5],atol=.006)
        np.testing.assert_allclose(sb[:,:,5],sa[:,:,5],atol=.005)

    def test_schedule_reverses_box_without_teleporting_contents(self):
        scene=grid(side=2,motion='shake',duration=1,gravity=(0,0),friction=0)
        r=run(scene);box=np.asarray(r['kinematic_states'])[:,0]
        self.assertAlmostEqual(box[60,0],.3,delta=1e-5)
        self.assertAlmostEqual(box[-1,0],0,delta=1e-5)
        self.assertTrue(np.isfinite(r['states']).all())
