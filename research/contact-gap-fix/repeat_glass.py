"""Replay the unchanged glass protocol into a new evidence directory."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    target = Path(args.output).resolve()
    if target.exists():
        raise FileExistsError(target)
    source = ROOT / 'research/documented-materials/compare_glass_worksheet.py'
    original = source.read_text()
    old = "D=P/'glass-worksheet-comparison'"
    assert original.count(old) == 1
    replay = original.replace(old, f'D=Path({str(target)!r})')
    # Preserve the protocol's original __file__ so its imports/path semantics
    # are unchanged. The sole edit is its exclusive output destination.
    exec(compile(replay, str(source), 'exec'), {'__file__': str(source), '__name__': '__main__'})
    receipt = dict(source=str(source.relative_to(ROOT)),
                   source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   change='Output directory only; no input, coefficient or metric change.',
                   output=str(target))
    (target / 'replay-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
