"""Copy frozen production headers and add a research-only QR trust trial."""
import hashlib,json,subprocess
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parent


def build():
    plan=json.loads((OUT/'plan.json').read_text());source=plan['baseline_source_commit'];headers=subprocess.check_output(['git','ls-tree','--name-only',source,'spatial_backend/'],cwd=ROOT,text=True).splitlines()
    headers=[p for p in headers if p.endswith('.h')];frozen={p:subprocess.check_output(['git','show',source+':'+p],cwd=ROOT) for p in headers}
    replay=subprocess.check_output(['git','show',source+':spatial_backend/coulomb_replay.cpp'],cwd=ROOT).decode()
    replay=replay.replace('#include <fstream>','#include <fstream>\n#include <chrono>')
    replay=replay.replace('bool solver_ok=coulombSolve(', 'auto solve_started=std::chrono::steady_clock::now();\n  bool solver_ok=coulombSolve(')
    replay=replay.replace('std::vector<double>w(n),impulses(n);','double solve_seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-solve_started).count();\n  std::vector<double>w(n),impulses(n);')
    replay=replay.replace('{"schema","native-circular-coulomb-replay-v1"}', '{"solve_time_s",solve_seconds},{"schema","native-circular-coulomb-replay-v1"}')
    old='''   if(remaining_svd>0){
    remaining_svd--;stats.svd_calls++;auto rhs=F;for(double& v:rhs)v=-v;
    auto linear=minimumNormNewton(J,rhs,n,1e-12);
    if(linear.converged&&trialStep(linear.step)){accepted=true;stats.newton_steps++;}
   }'''
    new='''   auto rhs=F;for(double& v:rhs)v=-v;
   auto qr=trial_qr::direction(J,rhs,n,1e-12);
   if(qr.converged&&trialStep(qr.step,true)){accepted=true;stats.newton_steps++;trial_qr::stats.accepted++;}
   else trial_qr::stats.rejected++;
   if(!accepted&&remaining_svd>0){
    remaining_svd--;stats.svd_calls++;
    auto linear=minimumNormNewton(J,rhs,n,1e-12);
    if(linear.converged&&trialStep(linear.step)){accepted=true;stats.newton_steps++;}
   }'''
    compiler=['g++',*plan['controls']['compiler_flags'],'-I'+str(ROOT/'build/bullet-inspect/src'),'-I'+str(ROOT/'build/spatial/_deps/json-src/include')]
    libs=[str(ROOT/'build/spatial/_deps/bullet-build/src'/name/('lib'+name+'.a')) for name in ['BulletDynamics','BulletCollision','LinearMath']]
    provenance=dict(plan_sha256=hashlib.sha256((OUT/'plan.json').read_bytes()).hexdigest(),source_commit=source,source_hashes={p:hashlib.sha256(raw).hexdigest() for p,raw in frozen.items()},compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],variants={})
    for variant in ['baseline','qr']:
        directory=OUT/variant;directory.mkdir(exist_ok=True)
        for name,raw in frozen.items():
            text=raw.decode()
            if variant=='qr' and Path(name).name in ['coulomb_trust.h','coulomb_active_trust.h']:
                assert text.count(old)==1;text=text.replace(old,new).replace('auto trialStep=[&](const std::vector<double>& step){','auto trialStep=[&](const std::vector<double>& step,bool meaningful=false){').replace('newMerit<=(1-1e-4*fraction)*oldMerit','newMerit<=(1-1e-4*fraction)*oldMerit&&(!meaningful||newMerit<=(1-1e-3)*oldMerit)').replace('#include "newton_linear.h"','#include "newton_linear.h"\n#include "qr_linear.h"')
            (directory/Path(name).name).write_text(text)
        copied=replay
        if variant=='qr':
            (directory/'qr_linear.h').write_bytes((OUT/'qr_linear.h').read_bytes())
            copied=copied.replace('{"solve_time_s",solve_seconds}', '{"qr_stats",{{"calls",trial_qr::stats.calls},{"accepted",trial_qr::stats.accepted},{"rejected",trial_qr::stats.rejected},{"budget_rejections",trial_qr::stats.budget_rejections},{"gram_calls",trial_qr::stats.gram_calls},{"gram_rejections",trial_qr::stats.gram_rejections},{"orthogonality_rejections",trial_qr::stats.orthogonality_rejections},{"model_rejections",trial_qr::stats.model_rejections}}},{"solve_time_s",solve_seconds}')
        (directory/'replay.cpp').write_text(copied);command=[*compiler,str(directory/'replay.cpp'),*libs,'-o',str(directory/'replay')]
        subprocess.run(command,cwd=ROOT,check=True)
        provenance['variants'][variant]=dict(command=command,binary_sha256=hashlib.sha256((directory/'replay').read_bytes()).hexdigest(),files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir() if p.name!='replay'})
    (OUT/'build-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n');print('Research-only baseline/QR replays built; no timing experiment executed')


if __name__=='__main__':build()
