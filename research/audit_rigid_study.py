"""Audit pinned sources, bundled traces and published error computations offline."""
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile
import numpy as np
from research.run_rigid_study import BUDGET, normalized_error

ROOT=Path(__file__).parent/'rigid-benchmarks'


def pack(directory):
    """Bundle individual already-compressed NPZ traces without deleting originals."""
    manifest=json.loads((directory/'trace-manifest.json').read_text())
    archive=directory/'traces.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_STORED) as out:
        for record in manifest:
            path=directory/'traces'/record['path']
            if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
                raise ValueError(f'Trace changed: {path}')
            out.write(path,record['path'])
    (directory/'trace-archive.json').write_text(json.dumps({'path':archive.name,
        'sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'entries':len(manifest),
        'extract_to':'traces/','format':'ZIP containing compressed NumPy NPZ arrays; no pickle'},indent=2)+'\n')


def audit(directory):
    lock=json.loads((ROOT/'source-lock.json').read_text())
    sources=lock['source_files']+lock['block_comparator']['source_files']
    for item in sources:
        if hashlib.sha256((ROOT/item['path']).read_bytes()).hexdigest()!=item['sha256']:
            raise ValueError(f"Pinned source changed: {item['path']}")
    archive_record=json.loads((directory/'trace-archive.json').read_text())
    archive=directory/archive_record['path']
    if hashlib.sha256(archive.read_bytes()).hexdigest()!=archive_record['sha256']:
        raise ValueError('Trace archive checksum mismatch')
    manifest=json.loads((directory/'trace-manifest.json').read_text()); traces={}
    with zipfile.ZipFile(archive) as stream:
        if set(stream.namelist())!={m['path'] for m in manifest}:
            raise ValueError('Unexpected or missing trace entries')
        for record in manifest:
            data=stream.read(record['path'])
            if hashlib.sha256(data).hexdigest()!=record['sha256']:raise ValueError('Trace digest mismatch')
            with np.load(io.BytesIO(data),allow_pickle=False) as arrays:
                trace={k:arrays[k].copy() for k in arrays.files}
            if not all(np.all(np.isfinite(v)) for v in trace.values()):raise ValueError('Nonfinite trace')
            if len(trace['states'])!=len(trace['times']):raise ValueError('History/sample mismatch')
            traces[record['case_id'],record['mode']]=trace
    with (directory/'comparisons.csv').open() as file:rows=list(csv.DictReader(file))
    for row in rows:
        reference=traces[row['case_id'],'reference_block'];candidate=traces[row['case_id'],row['mode']]
        for k in ('times','mass','inertia'):
            if not np.allclose(reference[k],candidate[k],rtol=1e-5,atol=1e-7):raise ValueError(f'{k} changed')
        delta=candidate['states']-reference['states']
        computed={'rms_position_m':np.sqrt(np.mean(np.sum(delta[:,:,:2]**2,axis=2))),
                  'rms_velocity_m_s':np.sqrt(np.mean(np.sum(delta[:,:,3:5]**2,axis=2))),
                  'rms_spin_rad_s':np.sqrt(np.mean(delta[:,:,5]**2))}
        for k,v in computed.items():
            if not np.isclose(v,float(row[k]),rtol=1e-9,atol=1e-12):raise ValueError(f'Error changed: {k}')
        if (normalized_error(computed)<=1)!=(row['within_budget']=='True'):raise ValueError('Budget verdict changed')
        if not np.isclose(normalized_error(computed),float(row['normalized_error'])):raise ValueError('Normalized error changed')
    summary=json.loads((directory/'summary.json').read_text())
    qualified={r['case_id'] for r in rows if r['block_reference_qualified']=='True'}
    if len(qualified)!=summary['qualified_block_references']:raise ValueError('Qualification count changed')
    print(json.dumps({'source_files_verified':len(sources),'traces_verified':len(traces),'comparisons_recomputed':len(rows),
        'physically_validated_cases':0,'scope':'Numerical reproduction and file integrity, not material or reference truth'},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT/'results')
    parser.add_argument('--pack',action='store_true')
    args=parser.parse_args()
    if args.pack:pack(args.directory)
    audit(args.directory)

if __name__=='__main__':main()
