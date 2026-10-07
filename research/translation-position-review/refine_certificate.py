"""Bounded local Farkas-certificate search, exact final validation."""
import json
from pathlib import Path
from fractions import Fraction as F
import numpy as np
from scipy.optimize import linprog

def main():
 folder=Path(__file__).parent;d=json.loads(Path('research/translation-position-diagnostic/results/rejected-normal-system.json').read_text());A=np.array(d['A']);b=np.array(d['b']);r=json.loads((folder/'triple-null-guide.json').read_text());l=[F(v)for v in r['weights']];n=len(l);response=[sum(l[i]*F(d['A'][i][j])for i in range(n))for j in range(n)];scale=F('1e-15');records=[]
 for limit in [1.,10.,100.,1000.]:
  bounds=[(-limit,limit) if v>0 else (0,limit) for v in l];result=linprog(np.zeros(n),A_ub=A.T,b_ub=-np.array([float(v/scale)for v in response]),bounds=bounds,method='highs',options={'primal_feasibility_tolerance':1e-9,'dual_feasibility_tolerance':1e-9});record=dict(correction_limit=limit,lp_success=bool(result.success),message=result.message)
  if result.success:
   exact=[l[i]+scale*F(result.x[i])for i in range(n)];actual=[sum(exact[i]*F(d['A'][i][j])for i in range(n))for j in range(n)];target=sum(exact[i]*F(d['b'][i])for i in range(n));margin=target-F(d['tolerance_m_s'])*sum(exact);valid=all(v>=0 for v in exact) and all(v<=0 for v in actual) and margin>0;record.update(validated_farkas_certificate=valid,weights_exact=[dict(numerator=str(v.numerator),denominator=str(v.denominator))for v in exact],response_exact=[dict(numerator=str(v.numerator),denominator=str(v.denominator))for v in actual],maximum_response=float(max(actual)),weighted_target=float(target),tolerance_separation_margin=float(margin))
  records.append(record);print({k:v for k,v in record.items()if not k.endswith('_exact')},flush=True);(folder/'refined-certificates.json').write_text(json.dumps(records,indent=2)+'\n')
  if record.get('validated_farkas_certificate'):break
if __name__=='__main__':main()
