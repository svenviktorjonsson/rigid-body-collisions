import copy
import json
from pathlib import Path
import unittest
import zipfile

from fidelity import select

ROOT=Path(__file__).resolve().parents[1]
MODES=[(4,64),(8,64),(16,16),(16,32),(16,64)]
EDGES=[((4,64),(8,64)),((8,64),(16,64)),((16,16),(16,32)),((16,32),(16,64))]


class ArchivedFidelityTests(unittest.TestCase):
    def traces(self,scene):
        with zipfile.ZipFile(ROOT/'research/random-shapes/results/traces.zip') as z:
            read=lambda p,s:json.loads(z.read(f'{scene}__block_p{p}_s{s}.json'))
            references={mode:read(*mode) for mode in MODES}
            candidates={f'p{p}s{s}':read(p,s) for p,s in [(1,1),(1,8),(4,16),(8,32)]}
        return references,candidates

    def test_failed_packed_reference_does_not_select_a_fast_setting(self):
        refs,candidates=self.traces('random_42_mixed36_shake')
        result=select(refs,(16,64),EDGES,candidates)
        self.assertEqual(result['status'],'unqualified_reference')
        self.assertIsNone(result['choice'])

    def test_choice_meets_actual_archived_convex_error_budget(self):
        refs,candidates=self.traces('random_42_convex_drop')
        result=select(refs,(16,64),EDGES,candidates)
        self.assertEqual(result['status'],'qualified')
        chosen=next(c for c in result['comparisons'] if c['candidate']==result['choice'])
        self.assertLessEqual(chosen['normalized_error'],1)
        rejected=next(c for c in result['comparisons'] if c['candidate']=='p1s1')
        self.assertGreater(rejected['normalized_error'],1)

    def test_physical_changes_cannot_be_used_as_fidelity_changes(self):
        refs,candidates=self.traces('random_42_convex_drop')
        candidates=copy.deepcopy(candidates);candidates['p1s8']['physical_setup_id']='different-material'
        with self.assertRaisesRegex(ValueError,'Different physical setup'):
            select(refs,(16,64),EDGES,candidates)


if __name__=='__main__':unittest.main()
