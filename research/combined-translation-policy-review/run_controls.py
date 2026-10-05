"""Research-only unit controls against frozen read-only existing Bullet libs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = Path(__file__).resolve().parent
SOURCE_FILES = ['plan.json', 'combined_translation.h', 'helper_checks.cpp', 'run_controls.py', 'README.md']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-commit', required=True)
    parser.add_argument('--receipt-directory', default='results-v1')
    args = parser.parse_args()
    assert re.fullmatch(r'[0-9a-f]{40}', args.source_commit)
    assert re.fullmatch(r'[a-zA-Z0-9_-]+', args.receipt_directory)
    source = subprocess.check_output(['git', 'rev-parse', args.source_commit], cwd=ROOT, text=True).strip()
    assert source == args.source_commit
    paths = [str((DIRECTORY / name).relative_to(ROOT)) for name in SOURCE_FILES]
    for path in paths:
        assert subprocess.check_output(['git', 'show', source + ':' + path], cwd=ROOT) == (ROOT / path).read_bytes()
    production_provenance = json.loads((ROOT / 'research/hull-gap-completion/results/checkpoints/provenance.json').read_text())
    assert production_provenance['execution_source_commit'] == '108a9bb4c7899f75d760b27b179cc56557904a08'
    binary = ROOT / 'build/spatial/spatial_runner'
    libraries = [ROOT / 'build/spatial/_deps/bullet-build/src' / name / ('lib' + name + '.a')
                 for name in ('BulletDynamics', 'BulletCollision', 'LinearMath')]
    headers = sorted((ROOT / 'build/bullet-inspect/src').rglob('*.h'))
    library_hashes = {str(p.relative_to(ROOT)): sha(p) for p in libraries}
    header_hashes = {str(p.relative_to(ROOT)): sha(p) for p in headers}
    def guard():
        assert sha(binary) == production_provenance['binary_sha256']
        for path, expected in production_provenance['source_hashes'].items():
            assert sha(ROOT / path) == expected
        for path, expected in production_provenance['runtime_library_hashes'].items():
            assert sha(path) == expected
        for path, expected in dict(library_hashes, **header_hashes).items():
            assert sha(ROOT / path) == expected
        for path in paths:
            assert subprocess.check_output(['git', 'show', source + ':' + path], cwd=ROOT) == (ROOT / path).read_bytes()
    guard()
    output = DIRECTORY / args.receipt_directory
    output.mkdir(exist_ok=False)
    build = ROOT / 'build/research' / ('combined-translation-' + source[:12] + '-' + args.receipt_directory)
    build.mkdir(parents=True, exist_ok=False)
    executable = build / 'helper_checks'
    command = ['c++', '-std=c++17', '-O2', '-DBT_USE_DOUBLE_PRECISION', '-I' + str(ROOT / 'build/bullet-inspect/src'),
               str(DIRECTORY / 'helper_checks.cpp'), '-o', str(executable), '-Wl,--start-group',
               *map(str, libraries), '-Wl,--end-group', '-pthread']
    provenance = dict(schema='combined-translation-research-native-controls-v1', execution_source_commit=source,
                      reviewed_production_source_commit=production_provenance['execution_source_commit'],
                      production_binary_sha256=sha(binary), source_hashes={p: sha(ROOT / p) for p in paths},
                      library_hashes=library_hashes, header_hashes=header_hashes,
                      compiler_version=subprocess.check_output(['c++', '--version'], text=True), compiler_command=command,
                      new_world_simulation=False, performance_comparison=False,
                      workload_note='Research unit compilation/execution may overlap frozen six-study CPU work; no controlled timing or speed claim.')
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    with zipfile.ZipFile(output / 'source.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for path in paths:
            archive.write(ROOT / path, path)
    compiled = subprocess.run(command, cwd=ROOT, text=True, capture_output=True)
    (output / 'compile.stdout.log').write_text(compiled.stdout)
    (output / 'compile.stderr.log').write_text(compiled.stderr)
    guard()
    result = dict(compile_exit_code=compiled.returncode, test_exit_code=None, controls_passed=False,
                  original_production_source_binary_libraries_unchanged=True)
    if compiled.returncode == 0:
        executed = subprocess.run([str(executable)], cwd=ROOT, text=True, capture_output=True)
        (output / 'test.stdout.log').write_text(executed.stdout)
        (output / 'test.stderr.log').write_text(executed.stderr)
        guard()
        result.update(test_exit_code=executed.returncode, controls_passed=executed.returncode == 0,
                      research_executable_sha256=sha(executable))
    (output / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['controls_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
