"""Independent energy, wrench, quadrature and indexed-incidence controls."""
import argparse, hashlib, json, math
from pathlib import Path
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.integrate import solve_ivp
from model import Patch, evaluate, directional_residual, gather_motion, scatter_wrench


def footprint(kind, radial=12, azimuth=32, radius=.012):
    if kind == 'interval':
        nodes, weights = leggauss(radial)
        points = np.column_stack((radius*nodes, np.zeros(radial)))
        return Patch(points, weights/2)
    nodes, weights = leggauss(radial)
    theta = (nodes+1)*math.pi/4
    phi = (np.arange(azimuth)+.5)*2*math.pi/azimuth
    boundary = np.ones(azimuth)
    if kind == 'irregular':
        boundary += .22*np.cos(3*phi) + .13*np.sin(5*phi)
    elif kind == 'ellipse':
        boundary = 1/np.sqrt(np.cos(phi)**2 + 4*np.sin(phi)**2)
    elif kind != 'hertz':
        raise ValueError(kind)
    r = radius*np.sin(theta)[:, None]*boundary
    points = np.stack((r*np.cos(phi), r*np.sin(phi)), axis=-1).reshape(-1, 2)
    w = (3*np.sin(theta)*np.cos(theta)**2*weights*math.pi/4)[:, None] * boundary**2/azimuth
    return Patch(points, (w/w.sum()).ravel())


def audit():
    rng = np.random.default_rng(7102701)
    worst_balance = worst_incidence = worst_gradient = worst_gram = 0.
    counter = dict(loaded=0, partly_open_or_clipped=0)
    directional_examples = []
    for case in range(240):
        kind = ['interval', 'hertz', 'ellipse', 'irregular'][case % 4]
        patch = footprint(kind)
        compression = np.r_[rng.uniform(-.001,.003), rng.uniform(-.3,.3,2)]
        u, w = rng.normal(size=(2,3))
        if kind == 'interval':
            u[1] = w[0] = w[2] = compression[1] = 0
        K, C, mu = 10**rng.uniform(3,6), 10**rng.uniform(-1,2), rng.uniform(0,1)
        result = evaluate(patch, compression, u, w, K, C, mu)
        scale = max(1., abs(result['power']), abs(result['stored_rate']))
        error = abs(result['energy_rate_residual'])/scale
        worst_balance = max(worst_balance, float(error))
        assert error < 1e-12
        assert result['normal_dissipation'] >= -1e-11*scale and result['sliding_dissipation'] >= 0
        # Stored energy gradient independently checked by symmetric finite difference.
        h = 1e-8
        q = np.array([u[2],w[0],w[1]])
        ep = evaluate(patch, compression-h*q,u,w,K,C,mu)['stored_energy']
        em = evaluate(patch, compression+h*q,u,w,K,C,mu)['stored_energy']
        gradient_error = abs((ep-em)/(2*h)-result['stored_rate'])/scale
        worst_gradient = max(worst_gradient,float(gradient_error))
        assert gradient_error < 2e-7
        counter['loaded' if result['all_loaded'] else 'partly_open_or_clipped'] += 1
        if result['all_loaded']:
            exact = patch.gram @ (K*compression-C*q)
            normal = np.array([result['force'][2],result['moment'][0],result['moment'][1]])
            err = np.max(np.abs(normal-exact))/max(1.,np.max(np.abs(exact)))
            worst_gram = max(worst_gram,float(err)); assert err < 1e-12
        # Arbitrary world frames/arms and shared indexed body incidence.
        frame,_ = np.linalg.qr(rng.normal(size=(3,3)))
        if np.linalg.det(frame) < 0: frame[:,0] *= -1
        owners=np.array([2,0]); signs=np.array([1.,-1.])
        contact_point=rng.normal(size=3); centers=rng.normal(size=(3,3)); arms=contact_point-centers[owners]
        bv,bw=rng.normal(size=(2,3,3)); f=np.zeros((3,3)); m=np.zeros((3,3))
        local_u,local_w=gather_motion(bv,bw,owners,signs,arms,frame)
        scatter_wrench(result['force'],result['moment'],owners,signs,arms,frame,f,m)
        power=float(np.sum(f*bv+m*bw)); direct=result['force']@local_u+result['moment']@local_w
        err=max(abs(power-direct),np.linalg.norm(f.sum(axis=0)),np.linalg.norm((m+np.cross(centers,f)).sum(axis=0)))/max(1.,abs(power),np.linalg.norm(f),np.linalg.norm(m))
        worst_incidence=max(worst_incidence,float(err)); assert err < 2e-12
    # Circular Hertz pure-spin analytic moment; no separately fitted spin coefficient.
    patch=footprint('hertz',32,64); K=10000.; d=.002; mu=.4; omega=7.
    r=evaluate(patch,[d,0,0],[0,0,0],[0,0,omega],K,0,mu)
    analytic=-3*math.pi/16*mu*K*d*.012
    pure_error=abs(r['moment'][2]-analytic)/abs(analytic)
    assert pure_error < 1e-12 and np.linalg.norm(r['force'][:2]) < 1e-13
    # Pressure damping creates a rolling couple even with zero center sliding.
    rolling=evaluate(patch,[d,0,0],[0,0,0],[4,0,0],K,20,mu)
    expected=-20*patch.gram[1,1]*4
    assert abs(rolling['moment'][0]-expected)<1e-14 and expected < 0
    # Mixed regular/irregular shape state: force/couple need not lie in user's spans.
    for kind in ['hertz','ellipse','irregular']:
        r=evaluate(footprint(kind,24,128),[.002,.03,-.02],[.1,.2,-.03],[3,5,40],10000,20,.4)
        directional_examples.append(dict(kind=kind,force=r['force'].tolist(),moment=r['moment'].tolist(),**directional_residual(r['force'],r['moment'],np.array([.1,.2,-.03]),np.array([3,5,40]))))
    # Convergence against separately much finer polar integration, plus stability of reference.
    quadrature=[]
    for kind in ['hertz','ellipse','irregular']:
        for ratio in [.1, .75, 2., 8.]:
            # r=V/(a*spin); chosen ratios include a local slip-zero within the patch.
            args=([.002,.015,-.02],[ratio*.012*10,0,-.01],[.1,.2,10],10000,10,.4)
            refs=[evaluate(footprint(kind,n,4*n),*args) for n in [64,128]]
            N=refs[-1]['force'][2]; a=.012
            def error(value):
                return max(np.linalg.norm(value['force']-refs[-1]['force'])/(.4*N),np.linalg.norm(value['moment']-refs[-1]['moment'])/(.4*N*a))
            errors=[float(error(evaluate(footprint(kind,n,4*n),*args))) for n in [4,8,16,32]]
            quadrature.append(dict(kind=kind,velocity_spin_ratio=ratio,site_counts=[64,256,1024,4096],errors=errors,reference_32768_vs_131072=float(error(refs[0]))))
    # Pure normal transient: internal energy retained even after force-zero release.
    p=footprint('interval',8); K=10000.; C=20.
    def rhs(time,state):
        d,v,loss=state
        r=evaluate(p,[d,0,0],[0,0,v],[0,0,0],K,C,0)
        return [-v,r['force'][2],r['normal_dissipation']]
    sol=solve_ivp(rhs,(0,.08),[0.,-1.,0.],rtol=2e-10,atol=1e-12,max_step=1e-4)
    E=.5*sol.y[1]**2+.5*K*np.maximum(sol.y[0],0)**2+sol.y[2]
    ode_error=float(np.max(np.abs(E-.5)))
    assert sol.success and ode_error < 2e-8
    trace=dict(time=sol.t.tolist(),compression=sol.y[0].tolist(),normal_speed=sol.y[1].tolist(),dissipation=sol.y[2].tolist(),total_energy=E.tolist())
    return dict(pass_=True,controls=240,branches=counter,maximum_scaled_power_balance_error=worst_balance,
                maximum_stored_energy_finite_difference_error=worst_gradient,maximum_indexed_work_momentum_error=worst_incidence,
                maximum_cached_gram_error=worst_gram,pure_spin_analytic_relative_error=pure_error,
                rolling_pressure_moment=rolling['moment'].tolist(),rolling_expected=expected,
                normal_transient_energy_error=ode_error,normal_transient=trace,
                directional_examples=directional_examples,quadrature=quadrature,
                experimental_validation=False,production_adoption=False,
                note='Independent mechanical controls, not experimental/material validation; no static shear closure or moving patch state.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); args=parser.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=False)
    result=audit()
    result['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(__file__).with_name('model.py')]}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['normal_transient','quadrature','source_sha256']},indent=2))
