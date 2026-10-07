"""Content-verified rocking download and selected RAR6 extraction, outside Git."""
import argparse,hashlib,json,shutil,subprocess
from pathlib import Path
from fetch import download

URL='https://experiments.builtenvdata.eu/api/v1/datasets/92/download/?filename=RockMasonry1.rar'
SHA='a2423df131acf78692a18f8732d4054b1e87b5d798f8d322adb5d96d773952f4'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    cache=Path(a.cache);out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    archive=cache/'RockMasonry1.rar';receipt=download(URL,archive)
    if receipt['sha256']!=SHA:raise RuntimeError('source archive changed; refuse unreviewed data revision')
    extracted=cache/'masonry-selected-v2'
    if not extracted.exists():
        if not shutil.which('unrar'):raise RuntimeError('unrar required for RAR6; 7z on this host cannot decode it')
        extracted.mkdir(parents=True)
        process=subprocess.run(['unrar','x','-y','-idq',str(archive),'DATA_FreeRocking/*/Singular_data.txt','DATA_FreeRocking/*/HalfCycle_and_Impact_data.txt',str(extracted)+'/'],capture_output=True,text=True)
        (out/'extract.stdout').write_text(process.stdout);(out/'extract.stderr').write_text(process.stderr)
        process.check_returncode()
    files=sorted(extracted.rglob('*.txt'))
    if len(files)!=270 or any(f.stat().st_size==0 for f in files):raise RuntimeError('incomplete/empty selected extraction; do not use failed 7z output')
    result=dict(archive=receipt,doi='10.60756/uminho-jh25',license='CC BY 4.0',
                selected_files=[dict(path=str(f.relative_to(extracted)),bytes=f.stat().st_size,sha256=hashlib.sha256(f.read_bytes()).hexdigest()) for f in files],
                interpretation='Processed angular ratios are outcomes; energy correction fc=.7 is not an independent friction input.')
    (out/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(archive_bytes=receipt['bytes'],selected_files=len(files))))
