import unittest
import numpy as np
from numpy.testing import assert_allclose
from contact_solver import assemble_planar, inelastic_normal_solve, passive_target_projection

class ContactTests(unittest.TestCase):
    def test_global_chain_coupling(self):
        centers=[[0,0],[1,0],[2,0]]
        contacts=[(0,1,[.5,0],[-1,0]),(1,2,[1.5,0],[-1,0])]
        inverse,G,K=assemble_planar(centers,[1,1,1],[1,1,1],contacts)
        assert_allclose(K[np.ix_([0,3],[0,3])],[[2,-1],[-1,2]])
        initial=np.array([1,0,0,0,0,0,-1,0,0.])
        after,p,_=inelastic_normal_solve(inverse,G,initial)
        assert_allclose(after,0,atol=1e-10)
        assert_allclose(p[[0,3]],[1,1])

    def test_internal_momentum_and_global_energy(self):
        centers=np.array([[0,.2],[1,-.1],[2,.3]])
        masses=np.array([1,2,3.]);inertias=np.array([.2,.5,.7]);ell=.3
        contacts=[(0,1,[.5,0],[-1,0]),(1,2,[1.5,0],[-1,0]),(0,1,[.5,.4],[-1,0])]
        inverse,G,K=assemble_planar(centers,masses,inertias,contacts,ell)
        rng=np.random.default_rng(1);initial=rng.normal(size=9);p=rng.normal(size=9)
        change=inverse@G.T@p;rows=change.reshape(-1,3)
        linear=masses[:,None]*rows[:,:2]
        assert_allclose(np.sum(linear,axis=0),0,atol=1e-12)
        angular=np.sum(centers[:,0]*linear[:,1]-centers[:,1]*linear[:,0]+inertias*rows[:,2]/ell)
        self.assertAlmostEqual(angular,0,places=11)
        H=np.linalg.inv(inverse)
        before=.5*initial@H@initial;after=.5*(initial+change)@H@(initial+change)
        self.assertAlmostEqual(after-before,(G@initial)@p+.5*p@K@p,places=10)
        self.assertLess(np.linalg.matrix_rank(K),len(K))

    def test_static_resting_contacts(self):
        inverse,G,_=assemble_planar([[0,0],[1,0]],[1,1],[.5,.5],[(0,1,[.5,0],[-1,0])])
        initial=np.zeros(6)
        after,p,_=inelastic_normal_solve(inverse,G,initial)
        assert_allclose(after,initial);assert_allclose(p,0)

    def test_projection_passivity_and_normal_target(self):
        K=np.array([[5.2631578947,-4.7368421053,0],[-4.7368421053,5.2631578947,0],[0,0,1.]])
        p,details=passive_target_projection(K,np.array([-1,1,.4]),[1,0,.5],1,.5,.3,.2)
        self.assertLessEqual(details['energy_change'],1e-7)
        self.assertAlmostEqual((np.array([-1,1,.4])+K@p)[0],1,places=7)

    def test_projection_reference_length_invariance(self):
        inverse,G,K=assemble_planar([[0,0],[1,.3]],[1,2],[.2,.4],[(0,1,[.4,.2],[-1,0])])
        physical_u=np.array([-1,.7,.4])
        outputs=[]
        for ell in (.1,1,10):
            S=np.diag([1,1,ell])
            p,details=passive_target_projection(S@K@S,S@physical_u,[.6,.3,.2],.8,.5,.3,.15,ell)
            outputs.append(S@p)
            self.assertLessEqual(details['energy_change'],1e-7)
        assert_allclose(outputs[0],outputs[1],atol=1e-7)
        assert_allclose(outputs[1],outputs[2],atol=1e-7)

if __name__=='__main__':unittest.main()
