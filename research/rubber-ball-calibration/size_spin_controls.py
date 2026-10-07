"""Exact synthetic inelastic disk/sphere impacts; no experimental rubber fit."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from rigid_engine import run as planar_run
from spatial_engine import run as spatial_run
from research.container_scenes import ball
from research.rigid_scenes import rectangle
from research.spatial_scenes import sphere

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).with_name('synthetic-size-spin-controls')
OUT.mkdir(exist_ok=False)
records = []
for dimension in (2, 3):
    for radius in (.023, .029, .05):
        for peripheral_spin in (-2., 0., 1., 2.):
            # Uniform 3D density for spheres; 2D areal density for disks.
            mass = (1200*4*math.pi*radius**3/3 if dimension == 3 else 12*math.pi*radius**2)
            factor = .4 if dimension == 3 else .5
            inertia = factor*mass*radius**2
            omega = peripheral_spin/radius
            if dimension == 2:
                dynamic = ball((0, radius+.01-1e-12), radius=radius, velocity=(1,-1), friction=.4)
                dynamic['circles'][0]['density'] = mass/(math.pi*radius**2)
                dynamic['omega'] = -omega
                wall = {'type':'kinematic','position':[0,-.1], 'polygons':[rectangle(2,.1,friction=.4,restitution=0)]}
                scene = {'duration':1e-6,'gravity':[0,0], 'collision_skin_m':.01,'bodies':[wall,dynamic]}
                binary = ROOT/'build/rigid_double_global_union_v1/rigid_runner'
                result = planar_run(scene,binary=binary,dt=1e-6,primary_steps=1,substeps=1,backend='block',position_iterations=12)
                states = np.asarray(result['states'])[:,0]
                measured = states[-1,3:6]
            else:
                dynamic = sphere([0,0,radius-1e-12],radius=radius,mass=mass,velocity=[1,0,-1],omega=[0,omega,0],friction=math.sqrt(.4))
                wall = {'type':'kinematic','position':[0,0,-.1],'friction':math.sqrt(.4),'shapes':[{'kind':'box','half_extents':[2,2,.1]}]}
                scene = {'duration':1e-6,'gravity':[0,0,0],'bodies':[wall,dynamic]}
                binary = ROOT/'build/spatial/spatial_runner'
                result = spatial_run(scene,binary=binary,dt=1e-6,primary_steps=1,iterations=4096,travel_fraction=0,solver='coulomb',kinematic_contact_phase='start',position_stabilization='split_translation_combined')
                states = np.asarray(result['states'])[:,1]
                measured = states[-1,[7,9,11]]
            # Independent impulse solution: normal impulse m; unconstrained tangential
            # impulse cancels contact slip, clipped at the exact Coulomb bound.
            tangent_impulse = np.clip(-(1-peripheral_spin)/(1/mass+radius**2/inertia),-.4*mass,.4*mass)
            final_spin = omega-radius*tangent_impulse/inertia
            expected = np.array([1+tangent_impulse/mass,0, -final_spin if dimension==2 else final_spin])
            error = float(np.max(abs(measured-expected)))
            initial_energy = .5*mass*2 + .5*inertia*omega**2
            final_energy = .5*mass*(measured[0]**2+measured[1]**2)+.5*inertia*measured[2]**2
            checks = {'dimension':dimension,'radius_m':radius,'peripheral_spin_m_s':peripheral_spin,
                      'mass_kg':mass,'inertia_kg_m2':inertia,'expected_velocity_spin':expected.tolist(),
                      'measured_velocity_spin':measured.tolist(),'max_velocity_spin_error':error,
                      'kinetic_change_J':final_energy-initial_energy,
                      'synthetic_only':True,'experimental_calibration':False,
                      'passed':bool(error<=1e-7 and final_energy-initial_energy<=1e-10)}
            records.append(checks)
            (OUT/f'{dimension}d_r{radius}_s{peripheral_spin}.json').write_text(json.dumps({'scene':scene,'result':result,'checks':checks},indent=2)+'\n')
            print(checks,flush=True)
summary={'passed':all(r['passed'] for r in records),'case_count':len(records),'records':records,
         'binaries':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'build/rigid_double_global_union_v1/rigid_runner',ROOT/'build/spatial/spatial_runner')},
         'scope':'Synthetic uniform disk/sphere radius and initial-spin controls. Inelastic rigid Coulomb model, no rubber compliance or measured friction fit. No global refinement/performance qualification.'}
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
assert summary['passed'], 'Retain every failed case; do not report control suite passing.'
