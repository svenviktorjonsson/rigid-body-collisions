"""Pinned public downloads; large third-party originals stay outside Git."""
import argparse, concurrent.futures, hashlib, json, urllib.request
from pathlib import Path
from urllib.parse import quote


def download(url,path):
    if not path.exists():
        path.parent.mkdir(parents=True,exist_ok=True)
        with urllib.request.urlopen(url,timeout=30) as response:
            data=response.read()
        path.write_bytes(data)
    return dict(url=url,path=str(path),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    root=Path(__file__).resolve().parent;protocol=json.loads((root/'PROTOCOL.json').read_text())
    cache=Path(args.cache);out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    jobs=[];rev=protocol['gauge_revision']
    # Fetch task trees separately, avoiding 1000-entry recursive truncation.
    for task in ['bouncing ball','slope slider','slope contact','nonsmooth contact','turntable']:
        prefix='data/rigid/'+task+'/json'
        url='https://huggingface.co/api/datasets/InternRobotics/GAUGE-Dataset/tree/'+rev+'/'+quote(prefix)+'?recursive=true&limit=1000'
        tree_receipt=download(url,cache/'trees'/(task+'.json'))
        tree=json.loads(Path(tree_receipt['path']).read_text())
        if len(tree)>=1000:raise RuntimeError('tree pagination required; refuse silent truncation')
        (out/(task+'-inventory.json')).write_text(json.dumps(tree,indent=2)+'\n')
        # Material/sliding prediction trials plus 3D contact schema pilots.
        paths=[x['path'] for x in tree if x['type']=='file' and x['path'].endswith('.json')]
        if task=='slope slider':paths=[p for p in paths if '/task-2/' in p]
        elif task not in ['bouncing ball']:paths=[p for p in paths if p.endswith('/1.json')]
        paths += ['metadata/rigid/'+task+'.json']
        for path in paths:
            jobs.append(('https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset/resolve/'+rev+'/'+quote(path),cache/'gauge'/path))
    for path in ['README.md','assets/obj/board.obj','assets/obj/ball-6.obj','assets/obj/teh.obj','assets/obj/pyramid.obj','assets/obj/wedge.obj','assets/obj/tribase.obj','assets/obj/squarebase.obj']:
        jobs.append(('https://huggingface.co/datasets/InternRobotics/GAUGE-Dataset/resolve/'+rev+'/'+quote(path),cache/'gauge'/path))
    for path in ['processed_data/README.md','processed_data/data_ellipse.mat']:
        jobs.append(('https://raw.githubusercontent.com/mcubelab/planar-impact-dataset/'+protocol['mit_revision']+'/'+path,cache/'mit'/path))
    receipts=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(download,*job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):receipts.append(future.result())
    receipts.sort(key=lambda x:x['path'])
    result=dict(pinned_gauge_revision=rev,pinned_mit_revision=protocol['mit_revision'],downloads=receipts,
                protocol_sha256=hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest())
    (out/'downloads.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(downloads=len(receipts),bytes=sum(x['bytes'] for x in receipts),revision=rev)))
