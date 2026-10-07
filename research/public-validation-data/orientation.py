"""Measured scene input candidate; material coefficients stay fixed."""
import argparse, csv, hashlib, json, math
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from study import trajectory, rmse, write_csv


def acceleration_from_normal(normal, mu, g=9.81):
    gravity=np.array([0.,0.,-g])
    tangential=gravity-normal*np.dot(normal,gravity)
    downhill=tangential/np.linalg.norm(tangential)
    return tangential-mu*g*normal[2]*downhill


def run(cache, baseline, out):
    rows=[];points=[];axis=np.array([math.cos(math.pi/6),0.,-math.sin(math.pi/6)])
    with (baseline/'slider-trials.csv').open() as stream:trials=list(csv.DictReader(stream))
    max_euler_error=0.
    for old in trials:
        material=old['material'];trial=int(old['trial']);mu=float(old['mu_source'])
        if old['status']!='accepted':raise ValueError('baseline declined: explicit paired handling needed')
        time,p,obj=trajectory(cache/f'gauge/data/rigid/slope slider/json/{material}/task-2/{trial}.json','cube')
        board=obj['board'];q=np.array([board[x] for x in ['qx','qy','qz','qw']]).T
        rotations=Rotation.from_quat(q)
        euler=np.array([board[x] for x in ['ez','ey','ex']]).T
        euler_rotation=Rotation.from_euler('ZYX',euler,degrees=True)
        max_euler_error=max(max_euler_error,float(np.max((rotations.inv()*euler_rotation).magnitude())))
        position=p@axis;distance=position-position[0];start=int(np.flatnonzero(distance>=.002)[0])
        prefix=np.arange(start,start+4);forecast=np.arange(start+4,len(p));forecast=forecast[distance[forecast]<=.30]
        normals=rotations.apply([0.,0.,1.]);normal=np.mean(normals[prefix],axis=0);normal/=np.linalg.norm(normal)
        if normal[2]<=0:raise ValueError('unexpected board orientation')
        acceleration=float(acceleration_from_normal(normal,mu)@axis)
        t=time-time[start];design=np.column_stack((np.ones(4),t[prefix]))
        c=np.linalg.lstsq(design,position[prefix]-.5*acceleration*t[prefix]**2,rcond=None)[0]
        predicted=c[0]+c[1]*t[forecast]+.5*acceleration*t[forecast]**2
        error=predicted-position[forecast]
        max_drift=float(np.max(np.arccos(np.clip(normals[np.r_[prefix,forecast]]@normal,-1,1))))
        # Not a correction: distance of cube center from moving board plane.
        bp=np.array([board[x] for x in ['x','y','z']]).T*.001
        normal_distance=np.sum((p-bp)*normals,axis=1)
        normal_range=float(np.ptp(normal_distance[np.r_[prefix,forecast]]))
        rows.append(dict(material=material,trial=trial,role=old['role'],mu_source=mu,
                         measured_angle_deg=float(np.degrees(np.arccos(normal[2]))),
                         baseline_acceleration=float(old['predicted_acceleration']),candidate_acceleration=acceleration,
                         baseline_rmse_m=float(old['forecast_rmse_m']),candidate_rmse_m=rmse(error),
                         rmse_change_m=rmse(error)-float(old['forecast_rmse_m']),
                         max_board_normal_drift_rad=max_drift,cube_normal_distance_range_m=normal_range,
                         prefix_samples=4,forecast_samples=len(forecast)))
        for j,pred,err in zip(forecast,predicted,error):
            points.append(dict(material=material,trial=trial,role=old['role'],frame=int(j),
                               predicted_downhill_position=float(pred),signed_error_m=float(err)))
    write_csv(out/'orientation-trials.csv',rows);write_csv(out/'orientation-points.csv',points)
    result=dict(max_quaternion_Euler_angle_error_rad=max_euler_error,evaluation={})
    for mat in ['wood','plastic','metal']:
        select=[r for r in rows if r['material']==mat and r['role']=='evaluation']
        errors=[r['signed_error_m'] for r in points if r['material']==mat and r['role']=='evaluation']
        result['evaluation'][mat]=dict(trials=len(select),forecast_rmse_m=rmse(errors),
            improved_trials=sum(r['rmse_change_m']<0 for r in select),regressed_trials=sum(r['rmse_change_m']>0 for r in select),
            measured_angle_range_deg=[min(r['measured_angle_deg'] for r in select),max(r['measured_angle_deg'] for r in select)],
            max_board_normal_drift_rad=max(r['max_board_normal_drift_rad'] for r in select),
            maximum_normal_distance_range_m=max(r['cube_normal_distance_range_m'] for r in select))
    result['scope']='Exploratory input correction; same source material values, trials, windows and measurement coordinate. All regressions retained.'
    result['sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('ORIENTATION-PROTOCOL.json')]}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--cache',required=True);parser.add_argument('--baseline',required=True);parser.add_argument('--output',required=True);a=parser.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    print(json.dumps(run(Path(a.cache),Path(a.baseline),out),indent=2))
