import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from supported_contact import Resistance,advance_planar,advance_spatial


class SupportedContactTests(unittest.TestCase):
    def test_small_radius_does_not_mix_linear_and_angular_acceleration_tolerances(self):
        for radius in [1e-12,1e-9,1e-6,1e-3,1.,1e3,1e6]:
            result=self.case(radius_m=radius,inertia_kg_m2=.4*radius*radius,
                             velocity_m_s=0.,omega_rad_s=0.,drive_force_N=1e-8,
                             material=Resistance(0.,0.,0.,0.))
            self.assertAlmostEqual(result['velocity_m_s'],1e-8,places=22)
            self.assertEqual(result['omega_rad_s'],0.)

    def test_spin_arrest_is_exact_for_400_nonbinary_mass_properties(self):
        rng=np.random.default_rng(19)
        for _ in range(400):
            result=self.case(inertia_kg_m2=rng.uniform(.001,.009),spin_rad_s=rng.uniform(-10,10),
                             velocity_m_s=0.,omega_rad_s=0.,duration_s=1e6,
                             material=Resistance(.5,.3,0.,0.,.1,.02))
            self.assertEqual(result['spin_rad_s'],0.)

    def case(self,**changes):
        args=dict(mass_kg=1.,inertia_kg_m2=.004,radius_m=.1,normal_load_N=9.81,
                  drive_force_N=0.,velocity_m_s=1.,omega_rad_s=10.,duration_s=1.,
                  material=Resistance(.5,.3,.02,.1))
        args.update(changes);return advance_planar(**args)

    def test_rolling_decelerates_at_zero_surface_slip_and_stops_without_reversal(self):
        for alpha in [.4,.5,2/3]:
            I=alpha*.1**2
            deceleration=.02*9.81/(1+alpha)
            a=self.case(inertia_kg_m2=I)
            self.assertAlmostEqual(a['velocity_m_s'],1-deceleration,places=13)
            self.assertAlmostEqual(a['omega_rad_s'],a['velocity_m_s']/.1,places=12)
            self.assertAlmostEqual(a['sliding_loss_J'],0,places=13)
            self.assertLess(a['independent_rolling_impulse_Nms'],0)
            stopped=self.case(inertia_kg_m2=I,duration_s=20.)
            self.assertEqual(stopped['velocity_m_s'],0.)
            self.assertEqual(stopped['omega_rad_s'],0.)
            self.assertAlmostEqual(stopped['distance_m'],1/(2*deceleration),places=12)

    def test_sliding_transitions_to_rolling_with_dynamic_friction(self):
        r=self.case(velocity_m_s=2.,omega_rad_s=0.,material=Resistance(.8,.3,0.,0.))
        self.assertAlmostEqual(r['velocity_m_s'],2/1.4,places=13)
        self.assertAlmostEqual(r['omega_rad_s'],r['velocity_m_s']/.1,places=12)
        self.assertAlmostEqual(r['events'][0]['force_N'],-.3*9.81,places=13)
        self.assertTrue(r['events'][0]['slip_arrest'])
        self.assertEqual(r['events'][-1]['slip_sign'],0)

    def test_static_and_dynamic_coefficients_choose_different_branches(self):
        # Holding rolling under this moment needs mu_r/(1+alpha) = .07143.
        enough=self.case(material=Resistance(.1,.02,.1,.1),duration_s=.1)
        insufficient=self.case(material=Resistance(.02,.02,.1,.1),duration_s=.1)
        self.assertEqual(enough['events'][0]['slip_sign'],0)
        self.assertNotEqual(insufficient['events'][0]['slip_sign'],0)
        self.assertAlmostEqual(enough['velocity_m_s']-.1*enough['omega_rad_s'],0,places=13)
        self.assertGreater(abs(insufficient['velocity_m_s']-.1*insufficient['omega_rad_s']),.01)

    def test_pure_axial_spin_changes_only_by_independent_angular_impulse(self):
        law=Resistance(.5,.3,0.,0.,.1,.02)
        r=self.case(velocity_m_s=0.,omega_rad_s=0.,spin_rad_s=5.,material=law,duration_s=10.)
        self.assertEqual(r['spin_rad_s'],0.)
        self.assertEqual(r['tangent_impulse_Ns'],0.)
        self.assertAlmostEqual(r['independent_spin_impulse_Nms'],-.004*5,places=14)
        self.assertAlmostEqual(r['spin_loss_J'],.5*.004*25,places=14)

    def test_drive_onset_static_hold_and_power_balance(self):
        hold=self.case(velocity_m_s=0.,omega_rad_s=0.,drive_force_N=.05,
                       material=Resistance(.2,.1,.1,.1))
        self.assertEqual(hold['velocity_m_s'],0.)
        self.assertEqual(hold['omega_rad_s'],0.)
        roll=self.case(velocity_m_s=0.,omega_rad_s=0.,drive_force_N=1.,
                       material=Resistance(.5,.3,0.,0.))
        self.assertAlmostEqual(roll['velocity_m_s'],1/1.4,places=13)
        self.assertAlmostEqual(roll['velocity_m_s'],.1*roll['omega_rad_s'],places=13)
        self.assertAlmostEqual(roll['external_work_J'],roll['final_kinetic_J'],places=13)

    def test_rotated_spatial_impulses_include_lever_moment_plus_free_couple(self):
        rotation=Rotation.from_rotvec([.3,-.7,.4]).as_matrix();n=rotation[:,2];d=rotation[:,0];axis=np.cross(n,d)
        U=.7*d+.2*axis
        r=advance_spatial(normal=n,direction=d,velocity=U+d,omega=10*axis+5*n,
                          plane_velocity=U,mass_kg=1.,inertia_kg_m2=.004,radius_m=.1,
                          normal_load_N=9.81,drive_force_N=0.,duration_s=1.,
                          material=Resistance(.5,.3,.02,.1,.1,.02))
        J=np.array(r['linear_impulse']);L=np.array(r['independent_angular_impulse'])
        expected=np.cross(-.1*n,J)+L
        np.testing.assert_allclose(r['body_angular_change'],expected,atol=1e-14)
        np.testing.assert_allclose(.004*(np.array(r['omega'])-10*axis-5*n),expected,atol=1e-14)
        self.assertAlmostEqual(r['support_work_J'],U@J,places=14)
        self.assertAlmostEqual(r['final_kinetic_J']-r['initial_kinetic_J']-r['support_work_J']+
                               r['sliding_loss_J']+r['rolling_loss_J']+r['spin_loss_J'],0,places=13)
        for frame in r['directional_branch_frames']:
            if frame['s'] is not None:
                M=np.array(frame['independent_moment']);B=np.column_stack([frame['s'],n])
                np.testing.assert_allclose(B@np.linalg.lstsq(B,M,rcond=None)[0],M,atol=1e-13)

    def test_small_initial_motion_is_not_erased_by_absolute_stop_threshold(self):
        for scale in [1e-14,1e-9,1e-4]:
            r=self.case(velocity_m_s=scale,omega_rad_s=0.,material=Resistance(.5,.3,0.,0.))
            self.assertAlmostEqual(r['velocity_m_s']/scale,1/1.4,places=12)
            self.assertGreater(r['velocity_m_s'],0.)
            self.assertLess(abs(r['energy_residual_J'])/r['initial_kinetic_J'],1e-12)

    def test_partial_angular_arrest_with_transverse_static_couple_is_not_relabelled(self):
        with self.assertRaisesRegex(ValueError,'s/n span'):
            advance_spatial(normal=[0,0,1],direction=[1,0,0],velocity=[1,0,0],omega=[0,0,5],
                            mass_kg=1.,inertia_kg_m2=.004,radius_m=.1,normal_load_N=9.81,
                            drive_force_N=0.,duration_s=.01,material=Resistance(.5,.3,.5,.1))

    def test_semigroup_and_sign_symmetry(self):
        law=Resistance(.4,.2,.06,.07,.03,.01)
        whole=self.case(material=law,velocity_m_s=1.7,omega_rad_s=-3.,spin_rad_s=-8.,duration_s=3.)
        first=self.case(material=law,velocity_m_s=1.7,omega_rad_s=-3.,spin_rad_s=-8.,duration_s=.8)
        second=self.case(material=law,velocity_m_s=first['velocity_m_s'],omega_rad_s=first['omega_rad_s'],spin_rad_s=first['spin_rad_s'],duration_s=2.2)
        for field in ['velocity_m_s','omega_rad_s','spin_rad_s']:self.assertAlmostEqual(whole[field],second[field],places=12)
        for field in ['distance_m','tangent_impulse_Ns','independent_rolling_impulse_Nms','independent_spin_impulse_Nms']:
            self.assertAlmostEqual(whole[field],first[field]+second[field],places=12)
        inverse=self.case(material=law,velocity_m_s=-1.7,omega_rad_s=3.,spin_rad_s=8.,duration_s=3.)
        for field in ['velocity_m_s','omega_rad_s','spin_rad_s','distance_m']:self.assertAlmostEqual(whole[field],-inverse[field],places=12)

    def test_unsupported_states_and_invalid_coefficients_are_explicit(self):
        with self.assertRaises(ValueError):Resistance(.1,.2,0.,0.)
        with self.assertRaises(ValueError):Resistance(.2,.1,.02,0.)
        with self.assertRaises(ValueError):advance_spatial(normal=[0,0,1],direction=[1,0,0],velocity=[1,1,0],omega=[0,10,0],mass_kg=1.,inertia_kg_m2=.004,radius_m=.1,normal_load_N=9.81,drive_force_N=0.,duration_s=1.,material=Resistance(.5,.3,.02,.1))


if __name__=='__main__':unittest.main()
