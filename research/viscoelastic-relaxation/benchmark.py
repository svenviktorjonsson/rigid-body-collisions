"""Build/run local C++ response kernels and audit against Python mechanics."""
import argparse,hashlib,json,platform,subprocess
from pathlib import Path
import numpy as np
from model import NormalTable,rolling_advance

HERE=Path(__file__).resolve().parent


def main():
    p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(exist_ok=False)
    try:
        data=np.load(args.evidence/'normal-table.npz');table=NormalTable(data['beta'],data['restitution'])
        coefficients=args.output/'coefficients.txt';np.savetxt(coefficients,table.interpolant.c.T,fmt='%.17g')
        source=HERE/'benchmark.cpp';binary=args.output/'benchmark-local'
        command=['g++','-O3','-std=c++17','-Wall','-Wextra','-Werror',str(source),'-o',str(binary)]
        build=subprocess.run(command,capture_output=True,text=True)
        (args.output/'compile.stderr').write_text(build.stderr)
        if build.returncode:raise RuntimeError('benchmark build failed')
        completed=subprocess.run([str(binary),str(coefficients)],capture_output=True,text=True,check=True)
        (args.output/'native-results.json').write_text(completed.stdout)
        result=json.loads(completed.stdout);worst_rolling=0.;worst_lookup=0.
        for control in result['controls']:
            m,R,alpha,om,dt,A=control['rolling_input']
            py=rolling_advance(m,R,alpha,[0,1,0],[0,0,1],om,dt,A,1.,R)
            # In this orientation e=(-1,0,0), so scalar reaction is -p_x.
            expected=np.array([py['angular_velocity'][2],-py['linear_impulse'][0],py['angular_impulse'][2],py['dissipation']])
            error=float(np.max(np.abs(expected-control['rolling_output'])/(1+np.abs(expected))))
            worst_rolling=max(worst_rolling,error);assert error<1e-12
            err=abs(float(table.evaluate(control['beta']))-control['restitution'])
            worst_lookup=max(worst_lookup,err);assert err<1e-12
        for batch in result['batches']:
            by_name={k['kernel']:k for k in batch['kernels']}
            a=by_name['relaxation_exp']['checksum'];b=by_name['relaxation_cached_factor']['checksum']
            assert abs(a-b)<1e-12*(1+abs(a))
        cpu='unknown'
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):cpu=line.split(':',1)[1].strip();break
        result['audit']=dict(pass_=True,native_python_controls=16,
            worst_scaled_rolling_difference=worst_rolling,worst_restitution_difference=worst_lookup,
            cached_and_uncached_checksums_match=True)
        result['provenance']=dict(command=command,compiler=subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
            machine=platform.machine(),cpu=cpu,source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            coefficients_sha256=hashlib.sha256(coefficients.read_bytes()).hexdigest(),
            fast_math_used=False,cached_factor_requires_constant_load_geometry_timestep=True)
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        for name in ('benchmark.cpp','benchmark.py'):(args.output/name).write_bytes((HERE/name).read_bytes())
        print(json.dumps(dict(audit=result['audit'],batches=result['batches']),indent=2))
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,reason=str(error)),indent=2)+'\n');raise


if __name__=='__main__':main()
