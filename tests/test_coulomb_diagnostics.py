import unittest
import json
from pathlib import Path
import numpy as np
from research.coulomb_diagnostics import System,solve,analyze,recover


def system(A,b,mu=.5):
    n=len(b);dep=np.repeat(3*np.arange(n//3),3);dep[::3]=-1
    lo=np.tile([0.,-mu,-mu],n//3);hi=np.tile([1e30,mu,mu],n//3)
    return System.from_dump(dict(A=np.asarray(A).tolist(),b=np.asarray(b).tolist(),lo=lo.tolist(),hi=hi.tolist(),dep=dep.tolist()))


class CoulombDiagnosticTests(unittest.TestCase):
    def test_sticking_and_sliding_with_full_coupling(self):
        A=np.array([[2.,.3,-.2],[.3,1.8,.4],[-.2,.4,1.5]])
        for p,w in [(np.array([2.,.3,-.4]),np.zeros(3)),(np.array([2.,-.6,-.8]),np.array([0.,.3,.4]))]:
            S=system(A,A@p-w);r=solve(S,tolerance=1e-10)
            self.assertTrue(r['accepted']);np.testing.assert_allclose(r['impulse'],p,atol=1e-8)
            self.assertLess(r['passive_change_bound_J'],0.)

    def test_exact_jacobian_matches_directional_difference_on_smooth_face(self):
        A=np.array([[2.,.3,-.2],[.3,1.8,.4],[-.2,.4,1.5]])
        S=system(A,[1.,2.,-1.]);p=np.array([.6,.1,-.2]);d=np.array([.3,-.4,.1])
        F,J=S.equations(p,True);numeric=(S.equations(p+1e-7*d)-S.equations(p-1e-7*d))/(2e-7)
        np.testing.assert_allclose(J@d,numeric,atol=1e-8)

    def test_redundant_contacts_preserve_law_without_compliance(self):
        G=np.vstack([np.eye(3),np.eye(3)]);A=G@G.T;b=np.tile([2.,3.,0.],2)
        r=solve(system(A,b),tolerance=1e-10)
        self.assertTrue(r['accepted']);self.assertLess(r['residual_m_s'],1e-10)
        self.assertEqual(analyze(system(A,b))['rank'],3)

    def test_tangent_rotation_covariance(self):
        A=np.array([[2.,.3,-.2],[.3,1.8,.4],[-.2,.4,1.5]])
        p=np.array([2.,-.6,-.8]);w=np.array([0.,.3,.4]);b=A@p-w
        angle=.73;Q=np.eye(3);Q[1:,1:]=[[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]]
        r=solve(system(Q@A@Q.T,Q@b),tolerance=1e-10)
        self.assertTrue(r['accepted']);np.testing.assert_allclose(r['impulse'],Q@p,atol=1e-8)

    def test_normal_complementarity_is_not_associated_cone_qp(self):
        r=solve(system(np.eye(3),[1.,4.,0.],mu=1),tolerance=1e-10)
        self.assertTrue(r['accepted']);np.testing.assert_allclose(r['impulse'],[1.,1.,0.],atol=1e-10)
        self.assertGreater(system(np.eye(3),[1.,4.,0.],mu=1).residual(np.array([2.5,2.5,0.])),1.)

    def test_incompatible_normal_targets_are_rejected_with_witness(self):
        G=np.zeros((6,5));G[0,0]=1;G[3,0]=-1;G[1,1]=G[2,2]=G[4,3]=G[5,4]=1
        S=system(G@G.T,[1.,0.,0.,1.,0.,0.]);report=analyze(S)
        self.assertIsNotNone(report['normal_target_witness'])
        self.assertEqual(report['normal_target_witness']['left_null_residual'],0.)
        r=solve(S,tolerance=1e-10,max_newton=5,max_nfev=30)
        self.assertFalse(r['accepted']);self.assertGreater(r['residual_m_s'],.5)

    def test_captured_nearly_singular_hull_recovers_with_cold_newton(self):
        path=Path(__file__).resolve().parents[1]/'research/coulomb-diagnostics/hull42-rejected.json'
        data=json.loads(path.read_text());S=System.from_dump(data)
        r=recover(S,data['p'],tolerance=data['tolerance_m_s'],max_nfev=0)
        self.assertTrue(r['accepted']);self.assertEqual(r['method'],'semismooth-svd+cold-restart')
        self.assertLess(r['residual_m_s'],1e-8)
        self.assertLess(r['passive_change_bound_J'],0.)

    def test_captured_inconsistent_stick_face_escapes_via_mechanical_nullspace(self):
        path=Path(__file__).resolve().parents[1]/'research/coulomb-diagnostics/hull7301-rejected.json'
        data=json.loads(path.read_text());S=System.from_dump(data)
        r=recover(S,data['p'],tolerance=data['tolerance_m_s'],max_nfev=0)
        self.assertTrue(r['accepted']);self.assertEqual(r['method'],'semismooth-svd+mechanical-null-gauge')
        self.assertLess(r['residual_m_s'],1e-8)
        proof=r['gauge_certificate']
        self.assertLess(proof['mobility_null_residual'],1e-12)
        self.assertLess(proof['velocity_change_m_s'],1e-12)
        self.assertEqual(proof['boundary_normal_row'],6)
        self.assertLess(r['passive_change_bound_J'],0.)


if __name__=='__main__':unittest.main()
