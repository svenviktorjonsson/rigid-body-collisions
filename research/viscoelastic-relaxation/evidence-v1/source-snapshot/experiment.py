"""Reproducible exploratory data comparison and reduced-model cost measurement."""
import argparse
import csv
import hashlib
import json
import math
import platform
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from model import NormalTable, normal_reference

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scores(errors):
    errors=np.asarray(errors,float)
    return dict(count=len(errors),rmse=float(np.sqrt(np.mean(errors**2))),
                maximum_absolute_error=float(np.max(np.abs(errors))))


def tennis(source):
    result=[]
    for specimen in source['tennis']['datasets']:
        points=specimen['points']
        train=[p for p in points if p['split']=='train']
        x=np.array([p['apparent_omega_rad_s'] for p in train])
        y=np.array([p['measured_effective_mu_r'] for p in train])
        constant=float(y.mean());relaxation=float(x@y/(x@x))
        records=[]
        for p in points:
            records.append(dict(specimen=specimen['specimen'],split=p['split'],
                omega_rad_s=p['apparent_omega_rad_s'],observed_mu_r=p['measured_effective_mu_r'],
                constant_prediction=constant,relaxation_prediction=relaxation*p['apparent_omega_rad_s'],
                pdf_center_pt=p['pdf_center_pt']))
        test=[r for r in records if r['split']=='held_out']
        baseline=scores([r['constant_prediction']-r['observed_mu_r'] for r in test])
        candidate=scores([r['relaxation_prediction']-r['observed_mu_r'] for r in test])
        loo=[]
        for omit in range(len(points)):
            cohort=[p for i,p in enumerate(points) if i!=omit]
            xx=np.array([p['apparent_omega_rad_s'] for p in cohort])
            yy=np.array([p['measured_effective_mu_r'] for p in cohort])
            a=float(xx@yy/(xx@xx));c=float(yy.mean());p=points[omit]
            loo.append(dict(omitted_index=omit,relaxation_s=a,
                constant_error=c-p['measured_effective_mu_r'],
                relaxation_error=a*p['apparent_omega_rad_s']-p['measured_effective_mu_r']))
        # Propagate the same one-PDF-point axis sensitivity as the prior report.
        axis=specimen['calibration_axis'];dx=(axis['xmax']-axis['xmin'])/(axis['x1']-axis['x0'])
        dy=(axis['ymax']-axis['ymin'])/(axis['y0']-axis['y1']);estimates=[]
        for sx in (-1,1):
            for sy in (-1,1):
                xx=x+sx*dx*2*math.pi/60;yy=y+sy*dy
                estimates.append(float(xx@yy/(xx@xx)))
        result.append(dict(specimen=specimen['specimen'],parameter_count_each_model=1,
            estimated_effective_relaxation_s=relaxation,constant_mu_r=constant,
            training_count=len(train),evaluation_count=len(test),constant=baseline,relaxation=candidate,
            rmse_reduction_percent=100*(1-candidate['rmse']/baseline['rmse']),
            maximum_error_reduction_percent=100*(1-candidate['maximum_absolute_error']/baseline['maximum_absolute_error']),
            one_pdf_point_sensitivity_s=[min(estimates),max(estimates)],
            leave_one_out=dict(constant=scores([r['constant_error'] for r in loo]),
                              relaxation=scores([r['relaxation_error'] for r in loo]),records=loo),
            records=records))
    return dict(datasets=result,fit_is_validation=False,
        independent_confirmatory_validation=False,
        input_is_apparent_belt_speed_not_independent_ball_spin=True,
        actual_viscoelastic_relaxation_identified=False,
        scope='Exploratory effective law for three quasirolling tennis specimens; slight skid, shell construction and specimen age confound continuum A')


def rock_comparison(table,source):
    data=source['data'];historical=source['models']['fixed']['fit']['theta']
    _,et,mu=historical
    def fit(train,kind):
        vn=np.array([r['vn_before_m_s'] for r in train]);R=np.array([r['diameter_m']/2 for r in train])
        obs=np.array([r['vn_after_m_s'] for r in train])
        total=np.array([math.hypot(r['vn_before_m_s'],r['vt_before_m_s']) for r in train])
        if kind=='constant':
            parameter=float(np.clip(np.sum(vn*obs/total**2)/np.sum(vn**2/total**2),0,1))
        else:
            factors=(.05/R)*vn**.2
            upper=float(table.beta[-1]/max((.05/(r['diameter_m']/2))*r['vn_before_m_s']**.2 for r in data))
            def objective(b):return float(np.mean(((table.evaluate(b*factors)*vn-obs)/total)**2))
            search=minimize_scalar(objective,bounds=(0.,upper),method='bounded',options={'xatol':1e-10})
            if not search.success:raise RuntimeError('normal parameter search failed')
            candidates=[(objective(search.x),float(search.x)),(objective(0.),0.),(objective(upper),upper)]
            parameter=min(candidates)[1]
        return parameter
    def evaluate(train,test,kind):
        parameter=fit(train,kind);records=[]
        en_hist,et,mu=historical
        for r in test:
            vn,vt=r['vn_before_m_s'],r['vt_before_m_s'];R=r['diameter_m']/2
            en=parameter if kind=='constant' else float(table.evaluate(parameter*.05/R*vn**.2))
            jt=float(np.clip(-(1+et)*vt/3.5,-mu*(1+en)*vn,mu*(1+en)*vn))
            pred=np.array([en*vn,vt+jt,2.5*abs(jt)/R])
            obs=np.array([r['vn_after_m_s'],r['vt_after_m_s'],r['omega_after_rad_s']])
            kinetic=.5*(pred[0]**2+pred[1]**2)+.2*R**2*pred[2]**2
            initial=.5*(vn**2+vt**2)
            if kinetic>initial*(1+1e-12):raise AssertionError('proxy created kinetic energy')
            records.append(dict(source_row=r['row'],diameter_m=r['diameter_m'],plate_angle_deg=r['plate_angle_deg'],
                release_height_m=r['release_height_m'],incoming_normal_m_s=vn,effective_e_n=en,
                prediction=pred.tolist(),observed=obs.tolist(),energy_retained=kinetic/initial))
        errors=np.array([np.array(r['prediction'])-r['observed'] for r in records])
        return dict(estimated_parameter=parameter,parameter_count=1,training_count=len(train),evaluation_count=len(test),
            normal=scores(errors[:,0]),tangent=scores(errors[:,1]),spin=scores(errors[:,2]),records=records)
    train=[r for r in data if r['release_height_m']!=4.5];test=[r for r in data if r['release_height_m']==4.5]
    height={kind:evaluate(train,test,kind) for kind in ('constant','viscoelastic')}
    folds=[]
    for angle in sorted({r['plate_angle_deg'] for r in data}):
        tr=[r for r in data if r['plate_angle_deg']!=angle];te=[r for r in data if r['plate_angle_deg']==angle]
        folds.append(dict(angle=angle,models={k:evaluate(tr,te,k) for k in ('constant','viscoelastic')}))
    pooled={k:scores([r['prediction'][0]-r['observed'][0] for f in folds for r in f['models'][k]['records']])
            for k in ('constant','viscoelastic')}
    return dict(height_split=height,angle_folds=folds,pooled_angle_normal=pooled,
        fixed_historical_tangential_restitution=et,fixed_historical_sliding_friction=mu,
        documented_coefficients_changed=False,actual_rock_shapes_reproduced=False,
        normal_parameter_is_true_material_constant=False,
        scope='Normal-response hypothesis test under homogeneous-sphere proxy; missing actual inertia/contact attitude; secondary endpoints retain historical fitted tangent/sliding inputs')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    try:
        plan=json.loads((HERE/'plan.json').read_text())
        # Fixed grid and accuracy before the real-data model evaluation.
        beta=np.linspace(0,8,513);start=time.perf_counter()
        references=[normal_reference(float(b)) for b in beta]
        setup=time.perf_counter()-start
        table=NormalTable(beta,np.array([r['restitution'] for r in references]))
        np.savez(args.output/'normal-table.npz',beta=beta,restitution=table.restitution)
        rng=np.random.default_rng(20261007)
        independent_beta=rng.uniform(0,8,64)
        check=[normal_reference(float(b),rtol=2e-12) for b in independent_beta]
        maximum=float(np.max(np.abs(table.evaluate(independent_beta)-[r['restitution'] for r in check])))
        if maximum>1e-5:raise AssertionError(f'table error exceeds declared tolerance: {maximum}')
        table_control=dict(grid_nodes=len(beta),range=[0,8],build_seconds=setup,
            grid_array_bytes=int(beta.nbytes+table.restitution.nbytes),
            pchip_coefficients_bytes=int(table.interpolant.c.nbytes),
            independent_random_checks=64,maximum_restitution_error=maximum,
            maximum_reference_energy_error=max(abs(r['energy_balance_residual']) for r in references+check),
            validation_records=check)
        timings=[]
        for size in (1,100,10000):
            batch=rng.uniform(0,8,size);samples=[]
            for _ in range(5):
                t=time.perf_counter()
                for _ in range(max(1,10000//size)):table.evaluate(batch)
                samples.append((time.perf_counter()-t)/max(1,10000//size))
            timings.append(dict(batch_size=size,median_seconds=float(np.median(samples)),
                median_microseconds_per_lookup=float(np.median(samples))*1e6/size))
        batch=independent_beta[:16];start=time.perf_counter()
        direct=[normal_reference(float(b))['restitution'] for b in batch]
        direct_seconds=time.perf_counter()-start
        max_error=float(np.max(np.abs(table.evaluate(batch)-direct)))
        table_control['timing']=dict(batches=timings,direct_ode_count=16,
            direct_ode_seconds=direct_seconds,comparison_maximum_error=max_error,
            scope='Local Python isolated normal endpoint map only; no collision detection, simultaneous-contact solve or native engine timing')
        tennis_path=ROOT/'research/full-experimental-report/evidence-v3/results.json'
        rock_path=ROOT/'research/restitution-validation-report/fits.json'
        result=dict(plan=plan,tennis=tennis(json.loads(tennis_path.read_text())),
            rocks=rock_comparison(table,json.loads(rock_path.read_text())),normal_table=table_control,
            sources_sha256={str(p.relative_to(ROOT)):fingerprint(p) for p in (tennis_path,rock_path)},
            host=dict(python=platform.python_version(),machine=platform.machine()),
            production_adopted=False,complete_directional_model_validated=False)
        (args.output/'results.json').write_text(json.dumps(result,indent=2)+'\n')
        snapshot=args.output/'source-snapshot';snapshot.mkdir()
        for name in ('model.py','experiment.py','plan.json'):(snapshot/name).write_bytes((HERE/name).read_bytes())
        with (args.output/'rolling-comparisons.csv').open('w',newline='') as f:
            fields=['specimen','split','omega_rad_s','observed_mu_r','constant_prediction','relaxation_prediction']
            writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader()
            for d in result['tennis']['datasets']:
                for r in d['records']:writer.writerow({k:r[k] for k in fields})
        print(json.dumps(dict(rolling=[dict(specimen=d['specimen'],A=d['estimated_effective_relaxation_s'],
            reduction_percent=d['rmse_reduction_percent']) for d in result['tennis']['datasets']],
            rock_height={k:v['normal'] for k,v in result['rocks']['height_split'].items()},
            rock_angle=result['rocks']['pooled_angle_normal'],table_error=maximum),indent=2))
    except Exception as error:
        (args.output/'failure.json').write_text(json.dumps(dict(type=type(error).__name__,reason=str(error)),indent=2)+'\n')
        raise


if __name__=='__main__':main()
