"""Build an explicitly experimental full-Float64 Box2D 2.4.1 diagnostic.

This transforms project-owned C/C++ scalar declarations, literals, elementary
math calls and precision constants, not system headers. It is not an upstream
Float64 release or a validated replacement for all Box2D features. The original
archive is SHA256 checked; source and transformed byte inventories are retained.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tarfile
import urllib.request

PIN = 'd6b4650ff897ee1ead27cf77a5933ea197cbeef6705638dd181adc2e816b23c2'
ROOT = Path(__file__).resolve().parents[1]


def _convert_code(text):
    text = re.sub(r'\bfloat\b', 'double', text)
    text = re.sub(r'(?<![\w.])((?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?)[fF]\b', r'\1', text)
    for name in ('sqrt', 'atan2', 'sin', 'cos', 'tan', 'fabs', 'acos', 'asin', 'atan', 'pow', 'exp', 'log'):
        text = re.sub(r'\b'+name+r'f\b', name, text)
    text = text.replace('FLT_EPSILON', 'DBL_EPSILON').replace('FLT_MAX', 'DBL_MAX')
    return text


def convert(text):
    tokens = r'(//[^\n]*|/\*.*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|^[ \t]*#[ \t]*include[^\n]*)'
    parts = re.split(tokens, text, flags=re.S|re.M)
    return ''.join(part if index%2 else _convert_code(part) for index, part in enumerate(parts))


def build(output, archive, linear_slop_m=.005):
    if linear_slop_m not in (.005, .000001):
        raise ValueError('Supported numerical slop: baseline .005 m or qualified .000001 m')
    output = output.resolve(); output.mkdir(parents=True, exist_ok=True)
    if (output/'precision-source.json').exists():
        raise ValueError('Preserve earlier precision builds; choose a new output directory')
    if not archive.is_file():
        urllib.request.urlretrieve('https://codeload.github.com/erincatto/box2d/tar.gz/v2.4.1', archive)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != PIN:
        raise ValueError('Pinned original archive SHA256 mismatch')
    source = output/'source'; source.mkdir(exist_ok=True)
    with tarfile.open(archive) as tar: tar.extractall(source, filter='data')
    box = source/'box2d-2.4.1'; inputs = {}; transformed = {}
    for path in sorted(box.rglob('*')):
        if path.suffix not in ('.h', '.cpp') or not path.is_file(): continue
        relative = str(path.relative_to(source)); original = path.read_bytes()
        inputs[relative] = hashlib.sha256(original).hexdigest()
        path.write_text(convert(original.decode()))
        transformed[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ('runner.cpp', 'compat2.h'):
        original = (ROOT/'rigid_backend'/name).read_bytes()
        inputs[name] = hashlib.sha256(original).hexdigest()
        code = convert(original.decode())
        if name == 'runner.cpp': code = code.replace('std::setprecision(10)', 'std::setprecision(17)')
        (source/name).write_text(code)
        transformed[name] = hashlib.sha256((source/name).read_bytes()).hexdigest()
    if linear_slop_m != .005:
        header=source/'box2d-2.4.1/include/box2d/b2_common.h'
        text=header.read_text()
        if text.count('(0.005 * b2_lengthUnitsPerMeter)') != 1:
            raise ValueError('Pinned numerical slop definition changed')
        header.write_text(text.replace('(0.005 * b2_lengthUnitsPerMeter)', '(0.000001 * b2_lengthUnitsPerMeter)'))
        transformed[str(header.relative_to(source))] = hashlib.sha256(header.read_bytes()).hexdigest()
    (source/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.22)
project(rigid_precision_probe LANGUAGES C CXX)
set(CMAKE_POLICY_VERSION_MINIMUM 3.8)
set(BOX2D_BUILD_TESTBED OFF CACHE BOOL "" FORCE)
set(BOX2D_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(box2d-2.4.1)
add_executable(rigid_runner runner.cpp)
target_compile_features(rigid_runner PRIVATE cxx_std_17)
target_compile_definitions(rigid_runner PRIVATE RIGID_BLOCK_BACKEND RIGID_DOUBLE_PRECISION)
target_compile_options(rigid_runner PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
target_compile_options(box2d PRIVATE -ffp-contract=off)
target_link_libraries(rigid_runner PRIVATE box2d)
''')
    provenance = {'upstream_archive_sha256': PIN, 'upstream_commit': '9ebbbcd960ad424e03e5de6e66a40764c16f51bc',
        'linear_slop_m': linear_slop_m,
        'collision_skin_scope': 'Numerical penetration slop; adapter installs authored polygon/circle radii explicitly',
        'scope': 'Experimental transformed Float64 polygon/circle comparator, not upstream-supported all-feature Box2D',
        'transform_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'inputs': inputs, 'transformed': transformed}
    (output/'precision-source.json').write_text(json.dumps(provenance, indent=2)+'\n')
    subprocess.run(['cmake', '-S', str(source), '-B', str(output), '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release'], check=True)
    subprocess.run(['cmake', '--build', str(output), '-j', '4'], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'build/rigid_double')
    parser.add_argument('--archive', type=Path, default=Path('/tmp/box2d-block.tar.gz'))
    parser.add_argument('--linear-slop-m', type=float, choices=[.005, .000001], default=.005)
    args = parser.parse_args(); build(args.output, args.archive, args.linear_slop_m)
