import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from spatial_engine import prepare,run,BINARY,energy


class MeasuredPropertiesTests(unittest.TestCase):
    def scene(self):
        Q=Rotation.from_rotvec([.3,-.2,.4]).as_matrix()
        I=Q@np.diag([.004,.006,.008])@Q.T
        return dict(duration=.01,gravity=[0,0,0],bodies=[dict(
            shapes=[dict(kind='box',half_extents=[.1,.2,.3])],
            mass_properties=dict(mass_kg=.121,center_of_mass_m=[.02,-.03,.01],inertia_body_kg_m2=I.tolist()))])

    def test_authoritative_mass_com_and_full_tensor_reach_native_frame(self):
        scene=self.scene();native,masses,I,axes,_=prepare(scene);body=scene['bodies'][0]
        self.assertEqual(masses,[.121])
        np.testing.assert_allclose(I[0],body['mass_properties']['inertia_body_kg_m2'],atol=1e-15)
        np.testing.assert_allclose(axes[0]@np.diag(native[0]['principal_inertia'])@axes[0].T,I[0],atol=1e-15)
        np.testing.assert_allclose(axes[0]@native[0]['shapes'][0]['center'],-np.array(body['mass_properties']['center_of_mass_m']),atol=1e-15)
        self.assertGreater(abs(I[0][0][1]),1e-4)

    def test_rejects_missing_com_asymmetry_nonphysical_inertia_and_nan(self):
        for change in [dict(mass_kg=-1),dict(inertia_body_kg_m2=[[1,0,0],[0,1,0],[0,0,3]]),
                       dict(inertia_body_kg_m2=[[1,.1,0],[0,1,0],[0,0,1]]),
                       dict(inertia_body_kg_m2=np.diag([1,1,float('nan')]).tolist())]:
            scene=self.scene();scene['bodies'][0]['mass_properties'].update(change)
            with self.assertRaises(ValueError):prepare(scene)
        scene=self.scene();del scene['bodies'][0]['mass_properties']['center_of_mass_m']
        with self.assertRaises(ValueError):prepare(scene)

    @unittest.skipUnless(BINARY.exists(),'Build native spatial backend')
    def test_native_free_rotation_uses_measured_tensor(self):
        scene=self.scene();scene['duration']=.05;scene['bodies'][0]['omega']=[1,2,3]
        result=run(scene,dt=.01,primary_steps=256)
        states=np.array(result['states'])[:,0];rot=Rotation.from_quat(states[:,3:7]).as_matrix();I=np.array(result['inertia_body_kg_m2'][0])
        L=np.einsum('tij,jk,tlk,tl->ti',rot,I,rot,states[:,10:13])
        self.assertLess(np.max(np.linalg.norm(L-L[0],axis=1)),1e-7)
        self.assertLess(np.ptp(energy(result)),1e-7)
        self.assertEqual(result['mass'],[.121])

    @unittest.skipUnless(BINARY.exists(),'Build native spatial backend')
    def test_native_offcenter_contact_uses_measured_effective_mass(self):
        I=np.diag([.0004,.0005,.0007]);m=.121;R=.1;com=np.array([.03,.02,.01])
        n=np.array([1.,0,0]);lever=-com-R*n;cross=np.cross(lever,n)
        impulse=2/(1/m+cross@np.linalg.solve(I,cross))
        sphere=dict(friction=0,mass_properties=dict(mass_kg=m,center_of_mass_m=com.tolist(),inertia_body_kg_m2=I.tolist()),shapes=[dict(kind='sphere',radius=R)])
        wall=dict(type='kinematic',position=[-.18,0,0],velocity=[2,0,0],friction=0,shapes=[dict(kind='box',half_extents=[.05,1,1])])
        scene=dict(duration=.00001,gravity=[0,0,0],bodies=[wall,sphere])
        result=run(scene,dt=.00001,primary_steps=1,travel_fraction=0,solver='normal_coupled',kinematic_contact_phase='start',position_stabilization='velocity_only')
        state=np.array(result['states'])[-1,1]
        np.testing.assert_allclose(state[7:10],impulse*n/m,atol=1e-7)
        np.testing.assert_allclose(state[10:13],np.linalg.solve(I,impulse*cross),atol=1e-7)


if __name__=='__main__':unittest.main()
