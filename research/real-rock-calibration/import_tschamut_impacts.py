"""Import measured natural-rock impact speeds without mislabelling them restitution."""
import csv,hashlib,json,math
from pathlib import Path
import argparse


def number(value):
    try:
        x=float(value)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('cache',type=Path);args=parser.parse_args()
    impact=args.cache/'tschamut2014-jumpsandimpacts/JumpsAndImpacts_commentedForEnvidat.txt'
    overview=args.cache/'tschamut2014-overviewalltests/OverviewAllTests.txt'
    tests={}
    for row in csv.reader(overview.open(encoding='utf-8',errors='replace'),delimiter='\t'):
        if len(row)>9 and row[0].isdigit():
            tests[int(row[0])]={'specimen_number':row[4],'specimen_name':row[5],'mass_kg':number(row[6]),'extent_cm':[number(x) for x in row[7:10]]}
    rows=list(csv.reader(impact.open(encoding='utf-8',errors='replace'),delimiter='\t'))
    assert rows[2][30]=='Rotation before [rps]' and rows[2][31]=='Rotation after [rps]'
    current=None;records=[]
    for line,row in enumerate(rows[3:],4):
        if row and row[0].strip().startswith('v'):
            label=row[0].strip();digits=''.join(c for c in label[1:] if c.isdigit());current=(label,int(digits) if digits else None)
            continue
        if current is None or len(row)<33 or number(row[16]) is None:continue
        before,after=number(row[30]),number(row[31]);specimen=tests.get(current[1])
        records.append({'test':current[0],'source_line':line,'specimen':specimen,
          'specimen_kind':'manufactured EOTA block' if specimen and specimen['specimen_name']=='EOTA' else 'natural rock',
          'impact_start_s':number(row[16]),'impact_end_s':number(row[17]),'contact_duration_s':number(row[18]),
          'impact_position_m':[number(row[k]) for k in (27,28,29)],
          'rotation_before_rad_s':None if before is None else 2*math.pi*before,
          'rotation_after_rad_s':None if after is None else 2*math.pi*after,
          'rotation_speed_ratio':None if before in (None,0) or after is None else after/before,
          'rotation_speed_ratio_is_tangential_restitution':False,
          'LPS_speed_m_s':number(row[32]),
          'full_state_prediction_qualified':False,
          'missing_inputs':['signed three-component angular velocities','body attitude and sensor alignment',
                            'contact point and surface normal','independent same-pair friction and restitution',
                            'impact-resolved incoming/outgoing linear vectors']})
    finite=[r for r in records if r['rotation_before_rad_s'] is not None and r['rotation_after_rad_s'] is not None]
    result={'source':'https://www.envidat.ch/metadata/tschamut2014','license':'ODbL+DbCL per dataset metadata',
            'source_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (impact,overview)},
            'impact_count':len(records),'finite_before_after_rotation_count':len(finite),
            'test_count':len({r['test'] for r in records}),'specimen_mass_join_count':sum(r['specimen'] is not None for r in records),
            'natural_rock_impact_count':sum(r['specimen_kind']=='natural rock' for r in records),
            'independent_material_validation_passed':False,'records':records}
    Path(__file__).with_name('tschamut-impact-fixtures.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({k:result[k] for k in ('impact_count','finite_before_after_rotation_count','test_count','specimen_mass_join_count')})
