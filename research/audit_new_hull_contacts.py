"""Recompute every retained later-capture diagnostic, including failed trials."""
import hashlib
import json
from pathlib import Path

import numpy as np
from research.audit_large_contact_completion import check


def audit():
    directory = Path('research/new-hull-contact-review')
    manifest = json.loads((directory / 'manifest.json').read_text())
    for entry in manifest['files']:
        raw = Path(entry['path']).read_bytes()
        assert len(raw) == entry['bytes']
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
    packages = sorted(directory.glob('*reference*.json'))
    assert len(packages) == 4
    trials = 0
    for path in packages:
        package = json.loads(path.read_text())
        capture = Path(package['capture'])
        assert hashlib.sha256(capture.read_bytes()).hexdigest() == package['capture_sha256']
        for trial in package['trials']:
            saved = trial['original_gate']
            checked = check(capture, saved['p'])
            assert checked['accepted'] == saved['accepted']
            for key in ('independent_full_original_residual_m_s', 'passive_change_bound_J',
                        'passivity_scale', 'maximum_cone_excess_N_s'):
                np.testing.assert_allclose(checked[key], saved[key], rtol=1e-9, atol=1e-12)
            trials += 1
        assert package['trials'][-1]['original_gate']['accepted']
    print('Later frozen-contact audit PASS:', len(packages), 'strict final solutions;',
          trials, 'retained successful and failed trials; no trajectory qualification')


if __name__ == '__main__':
    audit()
