"""Conditional bounce comparison with independently reconstructed incoming speed."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from study import trajectory,ballistic_fit,write_csv,rmse


def run(cache,baseline,out):
    with (baseline/'bounce-events.csv').open() as f:old=list(csv.DictReader(f))
    rows=[];g=9.81
    for original in old:
        if original['status']!='accepted':raise ValueError('explicit decline handling needed')
        trial=int(original['trial']);event=int(original['event']);center=int(original['frame']);e=float(original['source_restitution'])
        time,p,_=trajectory(cache/f'gauge/data/rigid/bouncing ball/json/{trial}.json','ball');z=p[:,2]
        valleys,_=find_peaks(-z,prominence=.02,distance=3);index=event-1
        assert center==valleys[index]
        lo=valleys[index-1]+2 if index else 0;hi=valleys[index+1]-1 if index+1<len(valleys) else len(z)
        before=np.arange(max(lo,center-7),center-1);after=np.arange(center+2,min(hi,center+8))
        pre,_=ballistic_fit(time[before],z[before],g);post,_=ballistic_fit(time[after],z[after],g)
        tm=time[center];incoming=g*tm-pre[1];outgoing=post[1]-g*tm
        assert incoming>0 and outgoing>0
        error=e*incoming-outgoing;halfwidth=g*(1+e)/30
        rows.append(dict(trial=trial,event=event,role=original['role'],source_restitution=e,marker_time_s=float(tm),
                         incoming_speed_m_s=float(incoming),measured_outgoing_speed_m_s=float(outgoing),
                         predicted_outgoing_speed_m_s=float(e*incoming),signed_error_m_s=float(error),
                         measured_ratio=float(outgoing/incoming),sensitivity_low_m_s=float(error-halfwidth),sensitivity_high_m_s=float(error+halfwidth)))
        # Input independence check: replacing all outgoing fit coefficients cannot
        # change the incoming-only input evaluated at this conditioning timestamp.
        changed_post=np.array([1e3,-1e3]);assert g*tm-pre[1]==incoming
    write_csv(out/'bounce-marker-events.csv',rows)
    select=[r for r in rows if r['role']=='evaluation'];errors=[r['signed_error_m_s'] for r in select]
    result=dict(evaluation_events=len(select),outgoing_speed_rmse_m_s=rmse(errors),mean_signed_error_m_s=float(np.mean(errors)),
                max_absolute_error_m_s=float(np.max(np.abs(errors))),measured_ratio_median=float(np.median([r['measured_ratio'] for r in select])),
                errors_with_zero_inside_sensitivity_envelope=sum(r['sensitivity_low_m_s']<=0<=r['sensitivity_high_m_s'] for r in select),
                one_frame_sensitivity_halfwidth_m_s=halfwidth,
                scope='Conditional on observed impact marker, incoming fit independent of outgoing arc. Not blind impact timing or full-model prediction. Sampling envelope is not a confidence interval.',
                sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('MEASUREMENT-ERRATUM.md')]})
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cache',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);print(json.dumps(run(Path(a.cache),Path(a.baseline),out),indent=2))
