"""Independent analytic controls for state reconstruction and scene projection."""
import argparse, csv, hashlib, json
from pathlib import Path
import numpy as np
from study import bounce_events, ballistic_fit
from orientation import acceleration_from_normal


def run(evidence):
    ballistic_error=0.;bounce_error=0.;n_events=0
    for phase in [.001,.009,.018,.028]:
        for e in [.58,.70,.80]:
            g=9.81;h=.04;tc=.31+phase;incoming=3.;outgoing=e*incoming
            t=np.arange(50)/30.;z=np.zeros_like(t)
            second=tc+2*outgoing/g
            for j,time in enumerate(t):
                if time<tc:
                    u=time-tc;z[j]=h-incoming*u-.5*g*u*u
                elif time<second:
                    u=time-tc;z[j]=h+outgoing*u-.5*g*u*u
                else:
                    u=time-second;z[j]=h+e*outgoing*u-.5*g*u*u
            # Restrict after second bounce to avoid unmodelled further impacts.
            stop=second+2*e*outgoing/g
            rows=bounce_events(t[t<stop],z[t<stop],e)
            assert len(rows)==2 and all(r['status']=='accepted' for r in rows)
            bounce_error=max(bounce_error,max(abs(r['signed_error']) for r in rows));n_events+=len(rows)
    for velocity in [-3.,1.,4.]:
        t=np.arange(12)/30;z=.7+velocity*t-4.905*t*t
        c,res=ballistic_fit(t,z,9.81)
        ballistic_error=max(ballistic_error,float(np.max(np.abs(c-[.7,velocity]))),res)
    for theta in [.15,.4,.7]:
        normal=np.array([np.sin(theta),0.,np.cos(theta)]);axis=np.array([np.cos(theta),0.,-np.sin(theta)])
        for mu in [.15,.25,.4]:
            got=acceleration_from_normal(normal,mu)
            expect=9.81*(np.sin(theta)-mu*np.cos(theta))*axis
            assert np.max(np.abs(got-expect))<1e-12
            assert abs(np.dot(normal,got))<1e-12
    assert bounce_error<1e-10 and ballistic_error<1e-10
    with (evidence/'bounce-events.csv').open() as f:b=list(csv.DictReader(f))
    with (evidence/'limestone-rocking-events.csv').open() as f:r=list(csv.DictReader(f))
    primary=[x for x in r if x['primary_first_event']=='True']
    allunique=[x for x in r if not x['duplicate_of'] and x['measured_source_angular_ratio']]
    ratios=[float(x['measured_source_angular_ratio']) for x in primary]
    result=dict(analytic_bounce_events=n_events,max_analytic_bounce_velocity_error_m_s=bounce_error,
                max_ballistic_fit_error=ballistic_error,gravity_projection_controls=9,
                bounce_evaluation_events=sum(x['role']=='evaluation' for x in b),
                bounce_error_zero_inside_one_frame_bracket=sum(float(x['error_bracket_low'])<=0<=float(x['error_bracket_high']) for x in b if x['role']=='evaluation'),
                maximum_bounce_flight_fit_rmse_m=max(max(float(x['pre_fit_rmse_m']),float(x['post_fit_rmse_m'])) for x in b),
                primary_rocking_ratios_over_one=sum(x>1 for x in ratios),primary_rocking_ratios_negative=sum(x<0 for x in ratios),
                primary_rocking_ratio_range=[min(ratios),max(ratios)],
                unique_finite_rocking_events=len(allunique),
                unique_finite_rocking_ratios_over_one=sum(float(x['measured_source_angular_ratio'])>1 for x in allunique),
                caveat='Synthetic controls validate extraction under instantaneous fixed-height ideal ballistic assumptions. One-frame brackets are sensitivity envelopes, not statistical confidence intervals. Source rocking ratios >1 are retained estimates, not certified active material restitution.')
    result['audit_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);result=run(Path(a.evidence))
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
