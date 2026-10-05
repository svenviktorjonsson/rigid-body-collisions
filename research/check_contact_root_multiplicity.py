"""Compare independently accepted roots through the original mobility A.

Impulse differences alone do not establish nonunique physical motion. A delta-p
compares captured post-contact velocities; exact mobility-null changes are
neutral impulse gauges. The declared screening threshold is ten capture gates.
No new solve, timing experiment, or material change is performed here.
"""
import hashlib,itertools,json
from pathlib import Path
import numpy as np
from research.audit_large_contact_completion import check


def run():
    roots={};rejected=[]
    def add(capture,p,label):
        capture=Path(capture);raw=capture.read_bytes();d=json.loads(raw);gate=check(capture,p)
        if not gate['accepted']:rejected.append(dict(label=label,capture=str(capture),gate=gate));return
        key=hashlib.sha256(raw).hexdigest();group=roots.setdefault(key,dict(capture=str(capture),data=d,roots=[]));group['roots'].append(dict(label=label,p=np.array(gate['p']),residual=gate['independent_full_original_residual_m_s']))
    for path in Path('research/completion-large-contact-review').glob('reference*.json'):
        d=json.loads(path.read_text())
        if 'trials' in d:
            for i,r in enumerate(d['trials']):
                if r.get('original_equations_accepted'):add(d['capture'],r['p'],str(path)+'#'+str(i))
        elif d.get('original_equations_accepted'):add(d['capture'],d['p'],str(path))
    for path in Path('research/new-hull-contact-review').glob('*.json'):
        d=json.loads(path.read_text())
        for i,r in enumerate(d.get('trials',[])):
            g=r['original_gate']
            if g['accepted']:add(d['capture'],g['p'],str(path)+'#'+str(i))
    for filename in ['active-final-four-native.jsonl','integrated-sixteen-replays.json']:
        path=Path('research/coulomb-normal')/filename
        records=json.loads(path.read_text()) if path.suffix=='.json' else [json.loads(x) for x in path.read_text().splitlines()]
        for i,r in enumerate(records):
            if 'result' in r:r=dict(r['result'],capture=r['input'])
            if r['accepted']:add(r['capture'],r['p'],str(path)+'#'+str(i))
    cert=Path('research/active-next-review/nonnegative-certificates.json')
    for r in json.loads(cert.read_text()):
        source=cert.parent/r['source'];d=json.loads(source.read_text());add(d['capture'],r['clipped_impulse'],str(source)+'#clipped')
    for path in [Path('research/active-next-review/native-normal-replays.jsonl')]:
        for i,r in enumerate(json.loads(x) for x in path.read_text().splitlines()):
            if not r['accepted']:continue
            d=json.loads(Path(r['capture']).read_text());ns=np.flatnonzero(np.array(d['dependencies'])<0);p=np.zeros(len(d['b']));p[ns]=r['normal_impulse'];add(r['capture'],p,str(path)+'#'+str(i))
    records=[]
    for digest,group in roots.items():
        if len(group['roots'])<2:continue
        A=np.array(group['data']['A']);tol=group['data']['tolerance_m_s'];pairs=[]
        for left,right in itertools.combinations(group['roots'],2):
            delta=left['p']-right['p'];response=A@delta;maximum=float(np.max(abs(response)));quadratic=float(delta@response)
            pairs.append(dict(left=left['label'],right=right['label'],left_residual_m_s=left['residual'],right_residual_m_s=right['residual'],
                maximum_impulse_difference_N_s=float(np.max(abs(delta))),maximum_contact_velocity_difference_m_s=maximum,
                mobility_quadratic_J=quadratic,screening_threshold_m_s=10*tol,distinct_contact_velocity_detected=maximum>10*tol))
        records.append(dict(capture=group['capture'],capture_sha256=digest,accepted_root_count=len(group['roots']),pairs=pairs))
    out=Path('research/contact-root-multiplicity');out.mkdir(exist_ok=True)
    package=dict(schema='accepted-root-mobility-multiplicity-review-v1',pairs=sum(len(r['pairs']) for r in records),
        distinct_contact_velocity_pairs=sum(p['distinct_contact_velocity_detected'] for r in records for p in r['pairs']),cases=records,rejected_inputs=rejected,
        interpretation='This compares retained accepted roots only. Mobility-null impulse differences are not distinct physical branches. Absence of detected distinct contact velocities neither proves uniqueness nor explains full-trajectory refinement failure.')
    (out/'accepted-root-comparison.json').write_text(json.dumps(package,indent=2)+'\n')
    print('Compared',package['pairs'],'accepted-root pairs;',package['distinct_contact_velocity_pairs'],'distinct-velocity screens')
    for r in records:print(r['capture'],max(p['maximum_contact_velocity_difference_m_s'] for p in r['pairs']))


if __name__=='__main__':run()
