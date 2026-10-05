"""Mechanical regressions for opt-in translation-only 3D position repair."""
import subprocess,unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from spatial_engine import BINARY,run


def kinetic(result):
    state=np.asarray(result['states']);total=np.zeros(len(state))
    for i,m in enumerate(result['mass']):
        if m==0:continue
        frame=Rotation.from_quat(state[:,i,3:7]).as_matrix()
        inertia=frame@np.asarray(result['inertia_body_kg_m2'][i])@frame.transpose(0,2,1)
        total+=.5*m*np.einsum('ti,ti->t',state[:,i,7:10],state[:,i,7:10])
        total+=.5*np.einsum('ti,tij,tj->t',state[:,i,10:13],inertia,state[:,i,10:13])
    return total


def fixture(world=Rotation.identity(),kinematic=False):
    ext=np.array([.2,.1,.05]);local=Rotation.from_rotvec([0,.4,0])
    z=abs(local.as_matrix()[2])@ext-.005
    floor=dict(type='kinematic' if kinematic else 'static',position=world.apply([0,0,-.05]).tolist(),
               orientation=world.as_quat().tolist(),friction=0,shapes=[dict(kind='box',half_extents=[10,10,.05])])
    if kinematic:floor['omega']=world.apply([0,2,0]).tolist()
    ball=dict(position=world.apply([0,0,z]).tolist(),orientation=(world*local).as_quat().tolist(),
              omega=world.apply([0,0,10]).tolist(),friction=0,
              shapes=[dict(kind='box',half_extents=ext.tolist(),density=1000)])
    return dict(duration=1e-5,gravity=[0,0,0],bodies=[floor,ball])


def simulate(scene,mode):
    return run(scene,dt=1e-5,primary_steps=1,iterations=4096,solver='coulomb',travel_fraction=0,
               kinematic_contact_phase='start',position_stabilization=mode)


@unittest.skipUnless(BINARY.exists(),'Build spatial backend')
class TranslationOnlySplit3D(unittest.TestCase):
    def test_anisotropic_spin_repair_preserves_physical_velocities_and_inertia_orientation(self):
        reference=simulate(fixture(),'velocity_only');repair=simulate(fixture(),'split_translation')
        a=np.asarray(reference['states']);b=np.asarray(repair['states'])
        np.testing.assert_allclose(b[:,:,3:],a[:,:,3:],atol=1e-11)
        self.assertGreater(b[-1,1,2]-a[-1,1,2],.0009)
        self.assertLess(abs(np.diff(kinetic(repair))[0]),1e-7)
        np.testing.assert_allclose(kinetic(repair),kinetic(reference),atol=1e-11)
        self.assertEqual(repair['numerical_model']['position_stabilization'],'split_translation')

    def test_repair_is_covariant_under_world_rotation(self):
        Q=Rotation.from_rotvec([.6,-.2,.3]);base=simulate(fixture(),'split_translation')
        turned=simulate(fixture(Q),'split_translation');a=np.asarray(base['states']);b=np.asarray(turned['states'])
        for index in (0,1):
            for first,last in ((0,3),(7,10),(10,13)):
                np.testing.assert_allclose(b[:,index,first:last],Q.apply(a[:,index,first:last]),atol=1e-9)
        np.testing.assert_allclose(kinetic(base),kinetic(turned),atol=1e-9)

    def test_rotating_wall_impulse_and_work_are_independent_of_pose_repair(self):
        scene=fixture(kinematic=True);reference=simulate(scene,'velocity_only');repair=simulate(scene,'split_translation')
        np.testing.assert_allclose(np.asarray(repair['states'])[:,:,3:],np.asarray(reference['states'])[:,:,3:],atol=1e-11)
        np.testing.assert_allclose(kinetic(repair),kinetic(reference),atol=1e-11)
        self.assertAlmostEqual(repair['boundary_work_J'],reference['boundary_work_J'],delta=1e-11)
        self.assertLessEqual(repair['coulomb_residual_max_m_s'],1e-8)

    def test_native_redundant_and_infeasible_position_constraints(self):
        binary=Path(BINARY).parent/'translation_split_checks'
        if not binary.exists():self.skipTest('Build translation_split_checks target')
        result=subprocess.run([str(binary)],capture_output=True,text=True,check=True)
        self.assertIn('Translation split checks PASS',result.stdout)
