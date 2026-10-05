"""Audit the explicitly prospective translation-only six-run hull protocol."""
import argparse
from pathlib import Path
import subprocess

from research.audit_shared_hulls import audit
from research.audit_hull_active_completion import audit_progress

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / 'research/hull-translation-completion'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    source = subprocess.check_output(['git', 'rev-parse', args.source_commit],
                                     cwd=ROOT, text=True).strip()
    audit(STUDY, source, position_stabilization='split_translation')
    audit_progress(STUDY, source, require_all=True)
