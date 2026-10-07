"""Fetch every pinned non-spherical GAUGE impact trajectory, not just pilots."""
import argparse,concurrent.futures,json
from pathlib import Path
from urllib.parse import quote
from fetch import download


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--inventory',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cache=Path(a.cache);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    rev=json.loads(Path(__file__).with_name('PROTOCOL.json').read_text())['gauge_revision'];jobs=[]
    for task in ['slope contact','nonsmooth contact']:
        inventory=json.loads((Path(a.inventory)/(task+'-inventory.json')).read_text())
        paths=[r['path'] for r in inventory if r['type']=='file' and r['path'].endswith('.json')]
        for path in paths:jobs.append(('https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset/resolve/'+rev+'/'+quote(path),cache/'gauge'/path))
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(download,*job) for job in jobs];receipts=[f.result() for f in concurrent.futures.as_completed(futures)]
    result=dict(gauge_revision=rev,scope='All spatial impact poses imported; no impact endpoints predicted here.',downloads=sorted(receipts,key=lambda r:r['path']))
    (out/'downloads.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(trajectories=len(receipts),bytes=sum(r['bytes'] for r in receipts))))
