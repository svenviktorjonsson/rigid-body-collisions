import copy
import unittest
import numpy as np

from research.convex_partition import merge, scene_partition
from research.random_shapes import generate, moments, scenes
from rigid_engine import validate_scene


def integrals(fixtures):
    mass=0.; first=np.zeros(2); polar=0.
    for f in fixtures:
        a,c,i=moments(f['vertices']); d=f.get('density',1)
        mass+=a*d; first+=a*d*c; polar+=d*(i+a*(c@c))
    return np.r_[mass,first,polar]


class PartitionTests(unittest.TestCase):
    def test_star_boundary_and_mass_moments_are_preserved(self):
        for seed in range(20):
            fixtures,geometry=generate(np.random.default_rng(seed),True)
            new,record=merge(fixtures)
            self.assertEqual(len(new),len(fixtures)//2)
            np.testing.assert_allclose(integrals(new),integrals(fixtures),atol=1e-14)
            outline=np.asarray(geometry['outline'])
            for p in new:
                # Every hull vertex comes from the original fan; tips/notches retained.
                for vertex in p['vertices']:
                    allvertices=np.concatenate([np.array(f['vertices']) for f in fixtures])
                    self.assertTrue(np.any(np.all(allvertices==vertex,axis=1)))
            for vertex in outline:
                self.assertTrue(any(np.any(np.all(np.asarray(p['vertices'])==vertex,axis=1)) for p in new))

    def test_heterogeneous_material_and_patch_boundaries_are_not_merged(self):
        fixtures,_=generate(np.random.default_rng(42),True)
        for i,f in enumerate(fixtures): f['friction']=.2+.01*i
        new,_=merge(fixtures);self.assertEqual(len(new),len(fixtures))
        for i,f in enumerate(fixtures): f['friction']=.4;f['material_patch']=i
        new,_=merge(fixtures);self.assertEqual(len(new),len(fixtures))

    def test_all_scenes_valid_and_input_is_unchanged(self):
        for scene in scenes():
            original=copy.deepcopy(scene);new=scene_partition(scene);validate_scene(new)
            self.assertEqual(scene,original)
            for a,b in zip(scene['bodies'],new['bodies']):
                np.testing.assert_allclose(integrals(a['polygons']),integrals(b['polygons']),atol=1e-12)


if __name__=='__main__':unittest.main()
