from pathlib import Path
import hashlib,json,os,subprocess,sys,time,zipfile
BASE=Path(__file__).resolve().parent;ROOT=BASE.parents[1];sys.path.insert(0,str(BASE))
from audit import guard,sha
p=guard();binary=Path('/tmp/translation-first256-replay')
libraries=[Path('/lib/x86_64-linux-gnu/liblapack.so.3'),Path('/lib/x86_64-linux-gnu/libblas.so.3')]+list((ROOT/'build/spatial/_deps/bullet-build/src').glob('*/lib*.a'))
launch={'plan_sha256':sha(BASE/'plan.json'),'binary_sha256':sha(binary),'pid':os.getpid(),'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'libraries':{str(q):sha(q)for q in libraries},'threads':p['threads'],'complete':False}
(BASE/'launch.json').write_text(json.dumps(launch,indent=2)+'\n')
with zipfile.ZipFile(BASE/'execution-source.zip','w',zipfile.ZIP_DEFLATED)as z:
 for f in p['source_hashes']:z.write(ROOT/f,f)
 z.write(BASE/'plan.json','plan.json')
env=os.environ.copy();env.update(p['threads'])
with (BASE/'paired-native.jsonl').open('w')as out,(BASE/'stderr.log').open('w')as err:
 result=subprocess.run([str(binary),*p['corpus']],cwd=ROOT,env=env,stdout=out,stderr=err)
guard();assert sha(binary)==launch['binary_sha256']
for f,h in launch['libraries'].items():assert sha(Path(f))==h
launch.update(complete=True,exit_code=result.returncode,finished_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),rows=len((BASE/'paired-native.jsonl').read_text().splitlines()),post_execution_guards=True)
(BASE/'completion.json').write_text(json.dumps(launch,indent=2)+'\n');print(json.dumps(launch),flush=True)
