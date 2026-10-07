"""Source provenance and flat spatial pose import, without guessed coefficients."""
import argparse,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
from study import trajectory,write_csv


def run(cache,out):
    rows=[];poses=[];counts=Counter();max_norm_error=0.;case=0
    for task in ['slope contact','nonsmooth contact']:
        metadata=json.loads((cache/f'gauge/metadata/rigid/{task}.json').read_text())
        for path in sorted((cache/f'gauge/data/rigid/{task}/json').rglob('*.json')):
            obj=json.loads(path.read_text());material=path.relative_to(cache/f'gauge/data/rigid/{task}/json').parts[0]
            names=[name for name in ['teh','pyramid','wedge','squarebase','tribase'] if name in obj]
            assert names
            for body,name in enumerate(names):
                time,p,_=trajectory(path,name);q=np.array([obj[name][x] for x in ['qx','qy','qz','qw']]).T
                assert q.shape==(len(p),4) and np.all(np.isfinite(q))
                norm_error=float(np.max(np.abs(np.linalg.norm(q,axis=1)-1)));max_norm_error=max(max_norm_error,norm_error)
                if norm_error>1e-4:raise ValueError('unexpected source quaternion normalization error')
                moving=name in ['teh','pyramid','wedge'];props=metadata['assets'][name]['material'][material if moving else 'wood']
                offset=sum(len(x) for x in poses);poses.append(np.column_stack([time,p,q]))
                rows.append(dict(case_index=case,body_index=body,task=task,source_trial=path.stem,source_subtask=path.parent.name if path.parent.name.startswith('task-') else 'default',
                                 material=material if moving else 'wood',asset=name,moving=moving,pose_offset=offset,pose_count=len(p),
                                 mass_kg=props['mass'],friction_source=props['friction'],restitution_source=json.dumps(props['restitution'],sort_keys=True),
                                 source_path=str(path.relative_to(cache)),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
            counts[task]+=1;case+=1
    write_csv(out/'spatial-bodies.csv',rows)
    # One flat pose index; entity ranges are explicit. Quaternions retain source signs.
    np.savez_compressed(out/'spatial-poses.npz',poses=np.concatenate(poses),columns=np.array(['time_s','x_m','y_m','z_m','qx','qy','qz','qw']))
    result=dict(trials=case,counts=dict(counts),body_records=len(rows),pose_records=sum(len(x) for x in poses),
                maximum_source_quaternion_norm_error=max_norm_error,
                indexing='Flat case_index; one body_index per case; explicit pose_offset/pose_count. Arrays contain observations, not predictions.',
                limits=['Released poses are 30 Hz; instantaneous velocities/spin and contact intervals cannot be assumed exact.',
                        'No independently measured full 3D inertia tensor provided by the metadata; uniform-density mesh inertia would be an assumption.',
                        'Nonsmooth JSON folders task-1/2/3 do not match metadata task-3/4/5 labels. Restitution dictionaries are preserved without guessing a mapping.',
                        'No separate mu_s, mu_d, mu_r or tangential restitution fields. Do not fill missing values with unrelated material data.'])
    (out/'spatial-results.json').write_text(json.dumps(result,indent=2)+'\n')
    sources=[
      dict(id='gauge',title='GAUGE: A Measurement-Grounded Benchmark for Physical Fidelity in Simulation Engines and Video World Models',
           url='https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset',paper='https://arxiv.org/html/2608.05948v1',license='MIT (dataset)',
           revision='9e0acb70fecc0d4161660264d9a4b08d8f56d45a',local_trials=dict(bounce=22,sliding_task2=59,spatial_impact=160),
           tested='Conditional normal bounce and prefix-conditioned sliding branches; no full spatial endpoint prediction.',
           parameter_provenance='Published calibrated metadata fixed unchanged. Material-pair and calibration/evaluation split are incompletely specified; not certified independent material validation.'),
      dict(id='mit_ellipse',title='Learning Data-Efficient Rigid-Body Contact Models: Case Study of Planar Impact',
           url='https://github.com/mcubelab/planar-impact-dataset',paper='https://proceedings.mlr.press/v78/fazeli17a.html',
           revision='f24a7e3b31ad0b53652d6b2a6b26a702cc4362da',license='No repository LICENSE found; originals stay outside this package.',
           local_records=1718,tested='Complete signed planar pre/post-state import and impulse reconstruction, not constitutive prediction.',
           parameter_provenance='Published mass and radius of gyration supplied. Published model coefficients fitted on these outcomes; independent matched friction/restitution absent.'),
      dict(id='limestone_rocking',title='Experimental dataset on the free-rocking response of masonry blocks',
           url='https://experiments.builtenvdata.eu/datasets/92/',doi='10.60756/uminho-jh25',paper='https://link.springer.com/article/10.1007/s10518-025-02224-8',license='CC BY 4.0',
           archive_sha256='a2423df131acf78692a18f8732d4054b1e87b5d798f8d322adb5d96d773952f4',release_date='2026-06-17',
           local_trials=135,unique_processed_pairs=134,processed_rows=7545,primary_events=134,
           tested='Nominal geometry-only ideal rocking comparator; no full rigid-body spatial prediction.',
           parameter_provenance='Nominal dimensions/mass/geometric inertia supplied. Processed angular ratios/effective geometry are outcomes. fc=.7 is an assumed energy correction, not independent friction.',
           discrepancy='Source paper describes 120 trials; released archive contains 135. Preserve release count and exact duplicate separately.'),
      dict(id='silicone_shell',title='Effect of a compliant substrate on the rebound of a spherical shell',
           url='https://data.hal.science/document/hal-05532284v1',pdf='https://hal.science/hal-05532284v1/file/EY12181.pdf',doi='10.1103/mmdr-2mm3',license='CC BY-NC-ND 4.0 paper metadata; original PDF outside repository.',
           tested='Source assessment only. Figure 8 spin data not digitized; no new prediction claimed.',
           experiment='40 mm diameter, 2.7 g ABS shell against bare glass or 0.50/0.96/1.25/1.78 mm silicone layers.',
           parameter_provenance='mu~.92, local stiffness/effective mass and derived shear modulus are fitted from rebound/spin results, not independently measured input properties.',
           model_relevance='Compliant local tangential displacement/pressure history changes spin transfer while keeping global bodies rigid; promising comparison for reduced local memory, pending independent inputs and data extraction.')
    ]
    fingerprints=[]
    for name in ['silicone-paper','mit-paper','gauge-paper','RockMasonry1.rar']:
        path=cache/name
        if path.exists():fingerprints.append(dict(cache_name=name,bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (out/'sources.json').write_text(json.dumps(dict(sources=sources,extra_source_fingerprints=fingerprints,catalog_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);print(json.dumps(run(Path(a.cache),out),indent=2))
