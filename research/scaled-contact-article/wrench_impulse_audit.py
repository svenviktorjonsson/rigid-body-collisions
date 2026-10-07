"""Independent 2D/3D torque-impulse checks; no fitted contact-moment law.

The controls prescribe force and free-couple impulses, rather than claiming a
prediction of an experimental contact wrench. Preserve outputs with --output.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from matrix_tools import Body, contact_wrench_map


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).with_name('wrench-impulse-v1'))
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    rng=np.random.default_rng(20261007)
    errors={name:0. for name in ['momentum','velocity','energy','duality','reference_shift','scale_invariance','pair_angular_balance','moving_boundary_work','planar']}
    def check(name,a,b):
        error=float(np.max(np.abs(np.asarray(a)-np.asarray(b))))
        errors[name]=max(errors[name],error)
        assert np.allclose(a,b,rtol=2e-11,atol=2e-11),(name,error)
    for _ in range(100):
        bodies=[];velocities=[];spins=[];maps=[]
        contact_ell=10**rng.uniform(-4,3)
        point=rng.normal(size=3);positions=rng.normal(size=(2,3))
        levers=point-positions
        j,k=rng.normal(size=(2,3));z=np.r_[j,k/contact_ell]
        for lever in levers:
            q=rng.normal(size=(3,3));body=Body(rng.uniform(.2,5),q.T@q+.4*np.eye(3),10**rng.uniform(-4,3))
            v,w=rng.normal(size=(2,3));bodies.append(body);velocities.append(v);spins.append(w)
            maps.append(contact_wrench_map(body,lever,contact_ell))
        before=[];after=[];physical_after=[];delta_energy=0.
        for sign,body,lever,v,w,H in zip((1,-1),bodies,levers,velocities,spins,maps):
            V=np.r_[v,body.ell*w];dP=H.T@(sign*z);Vp=V+body.mobility(dP)
            expected_v=v+sign*j/body.mass
            expected_w=w+np.linalg.solve(body.inertia,sign*(np.cross(lever,j)+k))
            check('momentum',dP,np.r_[sign*j,sign*(np.cross(lever,j)+k)/body.ell])
            check('velocity',np.r_[Vp[:3],Vp[3:]/body.ell],np.r_[expected_v,expected_w])
            check('duality',V@dP,sign*(j@(v+np.cross(w,lever))+k@w))
            shift=rng.normal(size=3)
            shifted=contact_wrench_map(body,lever+shift,contact_ell)
            check('reference_shift',shifted.T@np.r_[sign*j,sign*(k-np.cross(shift,j))/contact_ell],dP)
            other=Body(body.mass,body.inertia,body.ell*7)
            other_map=contact_wrench_map(other,lever,contact_ell/3)
            other_V=np.r_[v,other.ell*w]
            other_after=other_V+other.mobility(other_map.T@np.r_[sign*j,sign*k/(contact_ell/3)])
            check('scale_invariance',np.r_[other_after[:3],other_after[3:]/other.ell],np.r_[expected_v,expected_w])
            delta_energy+=.5*body.mass*(expected_v@expected_v-v@v)+.5*(expected_w@body.inertia@expected_w-w@body.inertia@w)
            before.append(V);after.append(Vp);physical_after.append((expected_v,expected_w))
        def mobility(force):return np.r_[bodies[0].mobility(force[:6]),bodies[1].mobility(force[6:])]
        G=np.column_stack((maps[0],-maps[1]));incoming=G@np.concatenate(before)
        W=np.column_stack([G@mobility(G.T@axis) for axis in np.eye(6)])
        check('energy',delta_energy,z@incoming+.5*z@W@z)
        total_delta=np.zeros(3)
        for body,x,v,w,(vp,wp) in zip(bodies,positions,velocities,spins,physical_after):
            total_delta+=np.cross(x,body.mass*(vp-v))+body.inertia@(wp-w)
        check('pair_angular_balance',total_delta,np.zeros(3))
        # Prescribed moving boundary contributes force work and torque work.
        b=bodies[0];H=maps[0];V=before[0];vp,wp=physical_after[0]
        vb,wb=rng.normal(size=(2,3));qrel=H@V-np.r_[vb,contact_ell*wb]
        Wa=np.column_stack([H@b.mobility(H.T@axis) for axis in np.eye(6)])
        dT=.5*b.mass*(vp@vp-velocities[0]@velocities[0])+.5*(wp@b.inertia@wp-spins[0]@b.inertia@spins[0])
        check('moving_boundary_work',dT-j@vb-k@wb,z@qrel+.5*z@Wa@z)
        # Planar scalar torque impulse and its dual velocity component.
        r=rng.normal(size=2);mass=float(rng.uniform(.2,5));I=float(rng.uniform(.1,3));ell=float(10**rng.uniform(-3,2))
        v=rng.normal(size=2);w=float(rng.normal());jp=rng.normal(size=2);kp=float(rng.normal())
        C=np.column_stack((np.eye(2),[-r[1]/ell,r[0]/ell]));K=np.vstack((C,[0,0,1]))
        V=np.r_[v,ell*w];zp=np.r_[jp,kp/ell];dp=K.T@zp
        cross=r[0]*jp[1]-r[1]*jp[0]
        check('planar',dp,np.r_[jp,(cross+kp)/ell])
        check('planar',V@dp,jp@(v+w*np.array([-r[1],r[0]]))+kp*w)
        delta=dp/np.array([mass,mass,I/ell**2]);energy=.5*((V+delta)@np.diag([mass,mass,I/ell**2])@(V+delta)-V@np.diag([mass,mass,I/ell**2])@V)
        check('planar',energy,zp@(K@V)+.5*zp@(K@np.diag([1/mass,1/mass,ell**2/I])@K.T)@zp)
    source=Path(__file__)
    input_path=source.with_name('torque_spin_inputs.json')
    inputs=json.loads(input_path.read_text())
    mass=inputs['mass_kg'];radius=inputs['radius_m'];ell=inputs['reference_length_m'];contact_ell=inputs['contact_reference_length_m']
    inertia=inputs['homogeneous_sphere_inertia_factor']*mass*radius**2
    body=Body(mass,inertia*np.eye(3),ell)
    normal=np.asarray(inputs['normal_world']);r=-radius*normal
    v=np.asarray(inputs['incoming_velocity_m_s']);w=np.asarray(inputs['incoming_angular_velocity_rad_s'])
    k=np.asarray(inputs['prescribed_free_torque_impulse_kg_m2_s'])
    j=-(1+inputs['normal_restitution'])*mass*(normal@v)*normal
    H=contact_wrench_map(body,r,contact_ell);V=np.r_[v,ell*w]
    eta=np.r_[j,k/contact_ell];Vp=V+body.mobility(H.T@eta)
    point_after=V+body.mobility(H.T@np.r_[j,np.zeros(3)])
    check('velocity',Vp[3:]/ell,w+k/inertia)
    check('momentum',np.cross(r,j),np.zeros(3))
    check('velocity',H[:3]@V,v)
    check('velocity',point_after[3:]/ell,w)
    W=np.column_stack([H@body.mobility(H.T@axis) for axis in np.eye(6)])
    before_energy=.5*V@body.momentum(V);after_energy=.5*Vp@body.momentum(Vp)
    check('energy',after_energy-before_energy,eta@(H@V)+.5*eta@W@eta)
    assert after_energy<before_energy
    central={'input_table':input_path.name,'input_sha256':hashlib.sha256(input_path.read_bytes()).hexdigest(),
             'inputs':inputs,'inertia_kg_m2':inertia,'force_impulse_kg_m_s':j.tolist(),
             'free_torque_impulse_kg_m2_s':k.tolist(),
             'force_only_outgoing_spin_rad_s':(point_after[3:]/ell).tolist(),
             'wrench_outgoing_spin_rad_s':(Vp[3:]/ell).tolist(),
             'outgoing_velocity_m_s':Vp[:3].tolist(),
             'angular_momentum_before_kg_m2_s':(inertia*w).tolist(),
             'angular_momentum_after_kg_m2_s':(inertia*Vp[3:]/ell).tolist(),
             'energy_before_J':float(before_energy),'energy_after_J':float(after_energy),
             'interpretation':inputs['purpose']}
    (args.output/'central-spin.json').write_text(json.dumps(central,indent=2)+'\n')
    rows=[
        (r'Mass $m$; radius $R$',f"${mass:g}\\,{{\\rm kg}}$; ${radius:g}\\,{{\\rm m}}$"),
        (r'Inertia $I$; lengths $\ell,\ell_c$',f"${inertia:g}\\,{{\\rm kg\\,m^2}}$; ${ell:g},{contact_ell:g}\\,{{\\rm m}}$"),
        (r'Incoming normal velocity; axial spin',f"${normal@v:g}\\,{{\\rm m/s}}$; ${normal@w:g}\\,{{\\rm rad/s}}$"),
        (r'Prescribed $e_n$; free axial impulse $k_n$',f"${inputs['normal_restitution']:g}$; ${normal@k:g}\\,{{\\rm kg\\,m^2/s}}$"),
    ]
    table=r'\begin{center}\small\begin{tabular}{ll}\hline Illustrative input & Value\\\hline'+'\n'
    table+='\n'.join(label+' & '+value+r'\\' for label,value in rows)
    table+=r'\hline\end{tabular}\end{center}'+'\n'
    table+=r'The torque impulse is prescribed solely to demonstrate the missing degree of freedom; it is not inferred from a measured outcome or claimed as a material prediction.'+'\n'
    table+=r'\begin{center}\small\begin{tabular}{lrr}\hline Computed quantity & Force impulse only & Force plus torque impulse\\\hline'+'\n'
    table+=r'Outgoing axial spin, rad/s & '+f'{normal@(point_after[3:]/ell):.3f} & {normal@(Vp[3:]/ell):.3f}'+r'\\'+'\n'
    table+=r'Outgoing axial angular momentum, kg m$^2$/s & '+f'{inertia*normal@(point_after[3:]/ell):.3f} & {inertia*normal@(Vp[3:]/ell):.3f}'+r'\\'+'\n'
    table+=r'Outgoing normal velocity, m/s & '+f'{normal@point_after[:3]:.3f} & {normal@Vp[:3]:.3f}'+r'\\'+'\n'
    table+=r'\hline\end{tabular}\end{center}'+'\n'
    table+=f'The full-wrench energy changes from ${before_energy:.3f}$ to ${after_energy:.3f}\\,{{\\rm J}}$. The axial angular-momentum change is exactly the prescribed torque impulse, while the central normal force impulse contributes no angular impulse.\n'
    (args.output/'spin-example.tex').write_text(table)
    result={'random_3d_pairs':100,'planar_controls':100,'max_absolute_errors':errors,
            'interpretation':'Prescribed synthetic force and independent torque impulses; algebra/energy/duality checks, not a measured contact-moment law.',
            'native_contact_moment_law_implemented':False,'central_spin_control':'central-spin.json',
            'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    (args.output/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
