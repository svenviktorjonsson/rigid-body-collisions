import copy
import importlib.util
import json
from pathlib import Path
import unittest

DIRECTORY = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('certificate_verify', DIRECTORY / 'verify.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def pair():
    return (dict(schema='normal-only-position-rejection-v1', A=[[1., -1.], [-1., 1.]],
                 b=[1., 1.], lo=[0., 0.], hi=[10., 10.], tolerance_m_s=1e-8),
            dict(support=[0, 1], weights=[dict(numerator='1', denominator='2')] * 2))


class CertificateTests(unittest.TestCase):
    def test_inconsistent_opposing_rows(self):
        data, witness = pair()
        result = module.verify(data, witness)
        self.assertTrue(result['certified'])
        self.assertEqual(result['residual_lower_bound_m_s'], 1)

    def test_feasible_pair_has_no_certificate(self):
        data, witness = pair()
        data['b'] = [1., -1.]
        self.assertFalse(module.verify(data, witness)['certified'])

    def test_finite_upper_bound_is_essential(self):
        data, witness = pair()
        data['A'] = [[1.125, -1.], [-1., 1.125]]
        data['hi'] = [1., 1.]
        self.assertTrue(module.verify(data, witness)['certified'])
        data['hi'] = [100., 100.]
        self.assertFalse(module.verify(data, witness)['certified'])

    def test_real_capture_and_corrupt_witness(self):
        witness = json.loads((DIRECTORY / 'witness.json').read_text())
        data = json.loads((DIRECTORY.parents[1] / witness['capture']).read_text())
        self.assertTrue(module.verify(data, witness)['certified'])
        corrupted = copy.deepcopy(witness)
        corrupted['weights'][0]['numerator'] = '-1'
        with self.assertRaises(ValueError):
            module.verify(data, corrupted)


if __name__ == '__main__':
    unittest.main()
