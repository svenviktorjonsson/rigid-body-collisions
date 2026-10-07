"""Create/verify a separate research extension; preserve earlier import bundles."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
DEST=Path('/home/viktor/Projects/Vektor Flow/Physics Reports/2026-10-07')


def digest(data):return hashlib.sha256(data).hexdigest()


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    archive=DEST/'modern-contact-optimization-extension-20261007.zip'
    pdf=DEST/'modern-contact-optimization-report-20261007.pdf'
    if archive.exists() or pdf.exists():raise FileExistsError('Preserve existing downloads; choose a new artifact version')
    names=['contact_history.py','contact_backend/schedule.h','contact_backend/README.md',
           'research/scaled-contact-article/MODEL-FIDELITY-AUDIT.md',
           'tests/test_contact_history.py','tests/test_contact_history_warm_start.py','tests/test_contact_schedule.py']
    names.extend(str(path.relative_to(ROOT)) for path in sorted(HERE.rglob('*'))
                 if path.is_file() and '__pycache__' not in path.parts and path.name not in ('downloads.json',))
    payload={name:(ROOT/name).read_bytes() for name in names}
    payload['tests/__init__.py']=b''
    payload['EXTENSION-README.txt']=b'''Research extension, 7 October 2026

This is a standalone local contact-history component and C++17 contact
topology scheduler, not a complete world engine or compiler port. The main
physics-supported-contact import package v2 remains a separate artifact.
Read contact_backend/README.md and research/modern-contact-optimization/README.md.
The current simulation module is the ROOT contact_history.py. Frozen/candidate
modules under research are historical references; preliminary candidates retain
a documented static-threshold defect and must not be selected for simulations.

Requires Python 3.11+, NumPy, g++ and OpenMP for the 15 component tests:
  python -m unittest tests.test_contact_history tests.test_contact_history_warm_start tests.test_contact_schedule -v
SciPy is additionally required for the independent optimizer/history experiment.
Run the boundary audit and benchmark commands given in the research README.
The 63 focused full-repository tests require the full repository and are not
claimed as standalone tests. This extension does not include experimental raw
datasets or demonstrate improved agreement with measurements.
manifest.json records every payload member SHA-256; verify before extraction.
'''
    manifest=dict(physics_source_sha256=digest(payload['contact_history.py']),
                  files={name:digest(data) for name,data in sorted(payload.items())})
    payload['manifest.json']=(json.dumps(manifest,indent=2)+'\n').encode()
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as bundle:
        for name,data in sorted(payload.items()):bundle.writestr(name,data)
    with zipfile.ZipFile(archive) as bundle:
        for name,expected in manifest['files'].items():
            if digest(bundle.read(name))!=expected:raise RuntimeError(f'Payload mismatch: {name}')
        with tempfile.TemporaryDirectory(prefix='physics-modern-extension-') as temporary:
            extracted=Path(temporary);bundle.extractall(extracted)
            command=[sys.executable,'-m','unittest','tests.test_contact_history','tests.test_contact_history_warm_start','tests.test_contact_schedule','-v']
            result=subprocess.run(command,cwd=extracted,capture_output=True,text=True)
            log=HERE/'verification/extracted-extension-tests.log';log.write_text(result.stdout+result.stderr)
            result.check_returncode()
    shutil.copy2(HERE/'report/report.pdf',pdf)
    receipt=dict(archive=dict(path=str(archive),bytes=archive.stat().st_size,sha256=digest(archive.read_bytes()),members=len(payload)),
                 report=dict(path=str(pdf),bytes=pdf.stat().st_size,sha256=digest(pdf.read_bytes())),
                 payload_verified=True,extracted_component_tests=15,extracted_tests_exit_code=result.returncode,
                 source_revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                 physics_source_sha256=manifest['physics_source_sha256'],scope='Experimental extension; no world/compiler acceptance or empirical accuracy claim')
    (HERE/'downloads.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
