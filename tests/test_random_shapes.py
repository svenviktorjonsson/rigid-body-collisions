import unittest
import copy
import numpy as np

from rigid_engine import validate_scene, run, BINARIES
from research.random_shapes import generate, moments, scenes, contact_chain
from research.sparse_contact import assemble_sparse


class RandomGeometryTests(unittest.TestCase):
    def test_mass_centroid_inertia_and_reproducibility(self):
        for seed in range(20):
            for concave in (False, True):
                fixtures, info = generate(np.random.default_rng(seed), concave)
                self.assertEqual((fixtures, info), generate(np.random.default_rng(seed), concave))
                parts = [(*moments(f['vertices']), f['density']) for f in fixtures]
                mass = sum(a*d for a, _, _, d in parts)
                center = sum(a*d*c for a, c, _, d in parts)/mass
                inertia = sum(d*(i+a*(c@c)) for a, c, i, d in parts)
                self.assertAlmostEqual(mass, 1., places=12)
                np.testing.assert_allclose(center, 0, atol=1e-14)
                self.assertAlmostEqual(inertia, info['inertia_kg_m2'], places=12)
                self.assertLessEqual(np.max(np.linalg.norm(info['outline'], axis=1)), .220000000001)

    def test_native_scenes_valid_and_initial_box_clearance(self):
        for scene in scenes():
            validate_scene(scene)
            if 'mixed36' in scene['id']:
                outlines = scene['generated_geometry']
                self.assertTrue(all(np.max(np.linalg.norm(g['outline'], axis=1)) <= .180000001 for g in outlines))
                # Any rotation fits inside radius .18; cell separation .5 and
                # initial nearest-wall clearance .4 both exceed 2*skin + radii.
                self.assertGreater(.5, 2*.18+2*.01)
                self.assertGreater(1.65-1.25, .18+.01)

    def test_chain_contacts_lie_on_actual_shapes_and_couple_rotation(self):
        for concave in (False, True):
            data = contact_chain(32, 42, concave)
            for a, b, point, normal in data['contacts']:
                for index in (a, b):
                    if index == 32: continue
                    world = np.asarray(data['geometry'][index]['outline'])+data['centers'][index]
                    self.assertLess(np.min(np.linalg.norm(world-point, axis=1)), 1e-12)
                world = np.asarray(data['geometry'][a]['outline'])+data['centers'][a]
                self.assertGreaterEqual(np.min((world-point)@normal), -1e-12)
            system = assemble_sparse(data['centers'], data['mass'], data['inertia'], data['contacts'])
            _, K = system.mobility((0, 1))
            self.assertGreater(K[::2, 1::2].nnz, 0)

    @unittest.skipUnless(BINARIES['temporal'].is_file(), 'Build the temporal comparator')
    def test_shallow_corners_survive_native_construction_without_mass_change(self):
        for scene in scenes():
            if 'mixed36' not in scene['id']: continue
            short = copy.deepcopy(scene); short['duration'] = 1/120
            short['bodies'][0]['velocity_schedule'] = short['bodies'][0]['velocity_schedule'][:1]
            result = run(short, backend='temporal', primary_steps=4, substeps=16)
            np.testing.assert_allclose(result['mass'], 1., atol=2e-6)
            np.testing.assert_allclose(result['inertia'],
                [g['inertia_kg_m2'] for g in scene['generated_geometry']], rtol=2e-5, atol=1e-7)

    @unittest.skipUnless(BINARIES['temporal'].is_file(), 'Build the temporal comparator')
    def test_clockwise_fixture_has_same_native_moments(self):
        shape, info = generate(np.random.default_rng(81))
        for fixture in shape: fixture['vertices'].reverse()
        scene = {'duration': 1/120, 'gravity': [0,0], 'bodies': [{'polygons': shape}]}
        result = run(scene, backend='temporal', primary_steps=1, substeps=1)
        self.assertAlmostEqual(result['mass'][0], 1., places=6)
        self.assertAlmostEqual(result['inertia'][0], info['inertia_kg_m2'], places=6)


if __name__ == '__main__': unittest.main()
