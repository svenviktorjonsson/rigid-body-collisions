"""Replay archived wide-scale importer oracles against the actual native core."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from supported_batch import PreparedBatch

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--controls',required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    source=Path(a.controls);data=np.loadtxt(source);x=data[:,:15].T;expected=data[:,15:].T
    batch=PreparedBatch(x);one=batch.run(threads=1);eight=batch.run(threads=8)
    assert np.array_equal(one,eight)
    speed=np.maximum.reduce([abs(x[5]),x[2]*abs(x[6]),x[2]*abs(x[7]),abs(x[4]/x[0])*x[8]])
    scale=np.array([speed,speed/x[2],speed/x[2],speed*x[8],x[0]*speed,x[0]*x[2]*speed,x[0]*x[2]*speed,x[0]*speed**2,x[0]*speed**2,x[0]*speed**2,x[0]*speed**2])
    error=float(np.max(abs(one-expected)/scale));assert error<1e-10
    result=dict(cases=x.shape[1],max_scaled_error=error,worker_outputs_bit_identical=True,
                controls_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['supported_backend/kernel.h','supported_backend/batch.cpp','supported_backend/batch.h','supported_batch.py']})
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
