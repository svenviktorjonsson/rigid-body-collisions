"""Real-data fixed-input branch comparisons and full-state readiness imports.

No engine adoption, no coefficient fitting, no redefinition of t/s directions.
"""
import argparse, csv, hashlib, json, math, re
from collections import defaultdict
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.signal import find_peaks


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def rmse(values):return float(np.sqrt(np.mean(np.square(values))))


def trajectory(path,asset):
    obj=json.loads(path.read_text())
    if obj['translation unit']!='mm' or obj['FPS']!=30:raise ValueError('undeclared source unit/fps change')
    p=np.array([obj[asset][axis] for axis in ['x','y','z']]).T*.001
    if not np.all(np.isfinite(p)):raise ValueError('nonfinite position')
    if len({len(v) for v in obj[asset].values()})!=1:raise ValueError('unequal source channel lengths')
    return np.arange(len(p))/obj['FPS'],p,obj


def ballistic_fit(time,z,g):
    design=np.column_stack((np.ones(len(time)),time))
    coefficients=np.linalg.lstsq(design,z+.5*g*time*time,rcond=None)[0]
    residual=z+.5*g*time*time-design@coefficients
    return coefficients,rmse(residual)


def bounce_events(time,z,e,g=9.81):
    valleys,_=find_peaks(-z,prominence=.02,distance=3)
    rows=[]
    for ordinal,center in enumerate(valleys[:2],1):
        index=ordinal-1
        lo=valleys[index-1]+2 if index else 0
        hi=valleys[index+1]-1 if index+1<len(valleys) else len(z)
        before=np.arange(max(lo,center-7),center-1)
        after=np.arange(center+2,min(hi,center+8))
        row=dict(event=ordinal,frame=int(center),status='declined',reason='',before_samples=len(before),after_samples=len(after),
                 incoming_speed=None,measured_outgoing_speed=None,predicted_outgoing_speed=None,
                 signed_error=None,measured_restitution=None,contact_time=None,contact_height=None,
                 pre_fit_rmse_m=None,post_fit_rmse_m=None,error_bracket_low=None,error_bracket_high=None)
        if min(len(before),len(after))<3:row['reason']='fewer than three clean flight samples'
        else:
            pre,pre_res=ballistic_fit(time[before],z[before],g)
            post,post_res=ballistic_fit(time[after],z[after],g)
            denominator=pre[1]-post[1]
            if abs(denominator)<1e-12:row['reason']='no unique flight intersection'
            else:
                tc=(post[0]-pre[0])/denominator
                incoming=-(pre[1]-g*tc);outgoing=post[1]-g*tc
                if abs(tc-time[center])>1/30:row['reason']='flight intersection outside one-frame contact bracket'
                elif min(incoming,outgoing)<=0:row['reason']='nonapproaching/nonseparating fitted state'
                else:
                    error=e*incoming-outgoing
                    row.update(status='accepted',incoming_speed=float(incoming),measured_outgoing_speed=float(outgoing),
                               predicted_outgoing_speed=float(e*incoming),signed_error=float(error),
                               measured_restitution=float(outgoing/incoming),contact_time=float(tc),
                               contact_height=float(pre[0]+pre[1]*tc-.5*g*tc*tc),
                               pre_fit_rmse_m=pre_res,post_fit_rmse_m=post_res,
                               error_bracket_low=float(error-g*(1+e)/30),error_bracket_high=float(error+g*(1+e)/30))
        rows.append(row)
    if not rows:rows.append(dict(event=0,frame=-1,status='declined',reason='no resolved valley',before_samples=0,after_samples=0,incoming_speed=None,measured_outgoing_speed=None,predicted_outgoing_speed=None,signed_error=None,measured_restitution=None,contact_time=None,contact_height=None,pre_fit_rmse_m=None,post_fit_rmse_m=None,error_bracket_low=None,error_bracket_high=None))
    return rows


def sliders(cache,out):
    metadata=json.loads((cache/'gauge/metadata/rigid/slope slider.json').read_text())
    rows=[];points=[];theta=math.radians(30);direction=np.array([math.cos(theta),0.,-math.sin(theta)])
    for material in ['wood','plastic','metal']:
        mu=metadata['assets']['cube']['material'][material]['friction']
        acceleration=9.81*(math.sin(theta)-mu*math.cos(theta))
        paths=sorted((cache/f'gauge/data/rigid/slope slider/json/{material}/task-2').glob('*.json'),key=lambda p:int(p.stem))
        for path in paths:
            time,p,_=trajectory(path,'cube');position=p@direction;distance=position-position[0]
            starts=np.flatnonzero(distance>=.002)
            row=dict(material=material,trial=int(path.stem),role='pilot' if path.stem=='1' else 'evaluation',mu_source=mu,
                     predicted_acceleration=acceleration,status='declined',reason='',prefix_samples=4,forecast_samples=0,
                     forecast_rmse_m=None,forecast_max_error_m=None,observed_acceleration=None,acceleration_error=None)
            if not len(starts):row['reason']='no resolved sliding onset'
            else:
                start=int(starts[0]);prefix=np.arange(start,min(start+4,len(p)))
                forecast=np.arange(start+4,len(p));forecast=forecast[distance[forecast]<=.30]
                if len(prefix)<4 or len(forecast)<3:row['reason']='insufficient fixed-window supported forecast'
                else:
                    t=time-time[start]
                    design=np.column_stack((np.ones(len(prefix)),t[prefix]))
                    c=np.linalg.lstsq(design,position[prefix]-.5*acceleration*t[prefix]**2,rcond=None)[0]
                    predicted=c[0]+c[1]*t[forecast]+.5*acceleration*t[forecast]**2
                    err=predicted-position[forecast]
                    observed=2*np.polyfit(t[np.r_[prefix,forecast]],position[np.r_[prefix,forecast]],2)[0]
                    row.update(status='accepted',forecast_samples=len(forecast),forecast_rmse_m=rmse(err),
                               forecast_max_error_m=float(np.max(np.abs(err))),observed_acceleration=float(observed),
                               acceleration_error=float(acceleration-observed))
                    for j,value,error in zip(forecast,predicted,err):points.append(dict(material=material,trial=int(path.stem),role=row['role'],frame=int(j),time=float(t[j]),measured_downhill_position=float(position[j]),predicted_downhill_position=float(value),signed_error_m=float(error)))
            rows.append(row)
    write_csv(out/'slider-trials.csv',rows);write_csv(out/'slider-forecast-points.csv',points)
    summary={}
    for material in ['wood','plastic','metal']:
        selected=[r for r in rows if r['material']==material and r['role']=='evaluation' and r['status']=='accepted']
        errors=[p['signed_error_m'] for p in points if p['material']==material and p['role']=='evaluation']
        summary[material]=dict(trials=len(selected),mu_source=metadata['assets']['cube']['material'][material]['friction'],
                               forecast_points=len(errors),forecast_rmse_m=rmse(errors) if errors else None,
                               maximum_forecast_error_m=float(np.max(np.abs(errors))) if errors else None,
                               mean_acceleration_error=float(np.mean([r['acceleration_error'] for r in selected])) if selected else None)
    return dict(trials=len(rows),accepted=sum(r['status']=='accepted' for r in rows),evaluation=summary,declines=[r for r in rows if r['status']!='accepted'])


def bounces(cache,out):
    metadata=json.loads((cache/'gauge/metadata/rigid/bouncing ball.json').read_text())
    e=metadata['assets']['ball']['material']['soft']['restitution'];rows=[]
    paths=sorted((cache/'gauge/data/rigid/bouncing ball/json').glob('*.json'),key=lambda p:int(p.stem))
    for path in paths:
        t,p,_=trajectory(path,'ball')
        for event in bounce_events(t,p[:,2],e):rows.append(dict(trial=int(path.stem),role='pilot' if path.stem=='1' else 'evaluation',source_restitution=e,**event))
    write_csv(out/'bounce-events.csv',rows)
    selected=[r for r in rows if r['status']=='accepted' and r['role']=='evaluation']
    errors=[r['signed_error'] for r in selected];ratios=[r['measured_restitution'] for r in selected]
    return dict(trials=len(paths),events=len(rows),accepted_events=sum(r['status']=='accepted' for r in rows),evaluation_events=len(selected),
                source_restitution=e,outgoing_speed_rmse_m_s=rmse(errors) if errors else None,
                mean_signed_error_m_s=float(np.mean(errors)) if errors else None,
                maximum_absolute_error_m_s=float(np.max(np.abs(errors))) if errors else None,
                measured_restitution_median=float(np.median(ratios)) if ratios else None,
                measured_restitution_range=[float(min(ratios)),float(max(ratios))] if ratios else None,
                declines=[r for r in rows if r['status']!='accepted'])


def ellipses(cache,out):
    data=loadmat(cache/'mit/processed_data/data_ellipse.mat',simplify_cells=True)['data']
    m=.0364;kg=.0192;I=m*kg*kg;metric=np.array([m,m,I]);rows=[]
    pre,post=data['states_i'],data['states_f'];n,d=data['n'],data['d']
    assert pre.shape==post.shape==(1718,6) and n.shape==d.shape==(1718,3)
    assert np.all(n[:,:2]==[0,1]) and np.all(d[:,:2]==[1,0])
    for i in range(len(pre)):
        if not all(np.all(np.isfinite(v)) for v in [pre[i],post[i],n[i],d[i]]):raise ValueError('nonfinite MIT source row')
        impulse=metric*(post[i,3:]-pre[i,3:])
        rx=n[i,2];ry=-d[i,2]
        lever=rx*impulse[1]-ry*impulse[0];free=impulse[2]-lever
        rows.append(dict(case_index=i,body_index=0,source_trial=i+1,mass_kg=m,nominal_I_kg_m2=I,rx=rx,ry=ry,
                         x_before=pre[i,0],y_before=pre[i,1],theta_before=pre[i,2],vx_before=pre[i,3],vy_before=pre[i,4],omega_before=pre[i,5],
                         x_after=post[i,0],y_after=post[i,1],theta_after=post[i,2],vx_after=post[i,3],vy_after=post[i,4],omega_after=post[i,5],
                         inferred_delta_px=impulse[0],inferred_delta_py=impulse[1],body_Delta_L=impulse[2],force_lever_angular_impulse=lever,inferred_independent_delta_L=free))
    write_csv(out/'mit-ellipse-states.csv',rows)
    # Geometry/Jacobian consistency against ellipse support, independent of outcomes.
    theta=pre[:,2];a=.035;b=.025
    support=np.sqrt((a*np.sin(theta))**2+(b*np.cos(theta))**2)
    expected_rx=-(a*a-b*b)*np.sin(theta)*np.cos(theta)/support
    geometry=max(float(np.max(np.abs(-d[:,2]+support))),float(np.max(np.abs(n[:,2]-expected_rx))))
    free=np.array([r['inferred_independent_delta_L'] for r in rows])
    return dict(records=len(rows),source_mass_kg=m,source_radius_of_gyration_m=kg,inertia_kg_m2=I,
                maximum_geometry_Jacobian_error_m=geometry,inferred_free_angular_impulse_rmse_Nms=rmse(free),
                note='Momentum reconstruction only; finite contact/pose changes, glass guide reactions and measurement uncertainty can explain residual. Not an independent torque measurement or constitutive prediction.')


def masonry(cache,out):
    root=cache/'masonry-selected-v2/DATA_FreeRocking';rows=[];trialrows=[];seen={};duplicate=[]
    for folder in sorted(root.glob('BlockGroup_*'),key=lambda p:tuple(map(int,re.fullmatch(r'BlockGroup_(\d+)_n(\d+)',p.name).groups()))):
        singular=folder/'Singular_data.txt';half=folder/'HalfCycle_and_Impact_data.txt'
        s=np.loadtxt(singular);x=np.atleast_2d(np.loadtxt(half));fingerprint=hashlib.sha256(singular.read_bytes()+half.read_bytes()).hexdigest()
        group,trial=map(int,re.fullmatch(r'BlockGroup_(\d+)_n(\d+)',folder.name).groups())
        predicted=1-1.5*s[1]**2/(s[1]**2+s[2]**2)
        assert abs(predicted-s[9])<1e-6
        duplicate_of=seen.get(fingerprint,'')
        if duplicate_of:duplicate.append(dict(trial=folder.name,duplicate_of=duplicate_of))
        else:seen[fingerprint]=folder.name
        finite=np.flatnonzero(np.isfinite(x[:,10]))
        for j in range(len(x)):
            measured=float(x[j,10]) if np.isfinite(x[j,10]) else None
            rows.append(dict(group=group,trial=trial,event=j+1,source_id=folder.name,duplicate_of=duplicate_of,nominal_mass_kg=float(s[0]),B_m=float(s[1]),H_m=float(s[2]),L_m=float(s[3]),nominal_I_cg_x_kg_m2=float(s[8]),predicted_ideal_angular_ratio=float(predicted),measured_source_angular_ratio=measured,signed_ratio_error=float(predicted-measured) if measured is not None else None,primary_first_event=bool(len(finite) and j==finite[0] and not duplicate_of)))
        errors=[float(predicted-x[j,10]) for j in finite]
        trialrows.append(dict(group=group,trial=trial,source_id=folder.name,duplicate_of=duplicate_of,finite_events=len(finite),missing_events=len(x)-len(finite),mass_kg=float(s[0]),B_m=float(s[1]),H_m=float(s[2]),L_m=float(s[3]),nominal_I_cg_x_kg_m2=float(s[8]),nominal_I_corner_kg_m2=float(s[5]),geometry_ideal_angular_ratio=float(predicted),first_measured_ratio=float(x[finite[0],10]) if len(finite) else None,first_ratio_error=errors[0] if errors else None,all_event_rmse=rmse(errors) if errors else None))
    write_csv(out/'limestone-rocking-events.csv',rows);write_csv(out/'limestone-rocking-trials.csv',trialrows)
    primary=[r for r in rows if r['primary_first_event']];errors=[r['signed_ratio_error'] for r in primary]
    groups={}
    for group in sorted({r['group'] for r in primary}):
        part=[r for r in primary if r['group']==group];err=[r['signed_ratio_error'] for r in part]
        groups[str(group)]=dict(trials=len(part),height_width_ratio=part[0]['H_m']/part[0]['B_m'],predicted_ratio=part[0]['predicted_ideal_angular_ratio'],measured_mean=float(np.mean([r['measured_source_angular_ratio'] for r in part])),rmse=rmse(err),max_error=float(np.max(np.abs(err))))
    return dict(source_trials=len(trialrows),unique_paired_trial_files=len(seen),exact_duplicate_pairs=duplicate,processed_event_rows=len(rows),primary_events=len(primary),first_event_ratio_rmse=rmse(errors),first_event_max_error=float(np.max(np.abs(errors))),groups=groups,
                note='Geometry-only ideal rocking comparator, not full directional model; measured source angular ratios are outcomes, never material inputs. Source nominal inertias are geometric.')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False);cache=Path(args.cache)
    result={}
    for name,function in [('bounce',bounces),('sliding',sliders),('ellipse_readiness',ellipses),('limestone_rocking',masonry)]:
        result[name]=function(cache,out)
        (out/'results.partial.json').write_text(json.dumps(result,indent=2)+'\n')
        print(name,json.dumps(result[name])[:1800],flush=True)
    result['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('PROTOCOL.json'),Path(__file__).with_name('ROCKING-PROTOCOL.json')]}
    result['scope']='Fixed published-metadata branch checks / ideal rocking comparator / full-state import; not full model or independent universal material validation.'
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
