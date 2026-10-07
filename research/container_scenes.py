"""Moving-container cases: analytic driven constraints and dense granular motion."""
import math
from research.rigid_scenes import rectangle


def container(hx, hy, velocity=(0, 0), omega=0, schedule=None, friction=0.4, restitution=0):
    t=.1; skin=.01; material={'friction':friction,'restitution':restitution}
    walls=[rectangle(t/2,hy+skin,offset=(sign*(hx+skin+t/2),0),**material) for sign in (-1,1)]
    walls += [rectangle(hx+skin+t,t/2,offset=(0,sign*(hy+skin+t/2)),**material) for sign in (-1,1)]
    b={'type':'kinematic','position':[0,0],'velocity':list(velocity),'omega':omega,'polygons':walls}
    if schedule is not None:b['velocity_schedule']=schedule
    return b


def ball(position, radius=.1, velocity=(0,0), friction=0.4, restitution=0, id=None):
    b={'circles':[{'radius':radius,'density':1/(math.pi*radius**2),'friction':friction,'restitution':restitution}],
       'position':list(position),'velocity':list(velocity)}
    if id is not None:b['id']=id
    return b


def row(count=64, speed=1, duration=1/120):
    r=.1; hx=count*r; hy=.3
    return {'id':f'packed_row_{count}','duration':duration,'gravity':[0,0],'collision_skin_m':.01,
        'bodies':[container(hx,hy,velocity=(speed,0),friction=0),
                  *[ball((-hx+r+2*r*i,0),radius=r,friction=0,id=f'b{i}') for i in range(count)]],
        'container':{'half_extents_m':[hx,hy],'radius_m':r},'parameter_provenance':'synthetic_analytic_constraint',
        'analytic':'Zero gaps, no deformation, e=0: all vx must equal the prescribed box speed. Wall work=U*delta P_x; D=W-delta K>=0.'}


def grid(side=10, motion='shake', duration=2, boost=(0,0), gravity=(0,-9.81), friction=.4):
    r=.1; spacing=.205; half=(side-1)*spacing/2+r+.01
    if motion=='stationary':v=(0,0);omega=0;schedule=None
    elif motion=='translate':v=(.6,0);omega=0;schedule=None
    elif motion=='shake':
        v=(.6,0);omega=0
        schedule=[{'time_s':0,'velocity':[.6,0]}, {'time_s':.5,'velocity':[-.6,0]},
                  {'time_s':1,'velocity':[.6,0]}, {'time_s':1.5,'velocity':[-.6,0]}]
        schedule=[c for c in schedule if c['time_s']<duration]
    elif motion=='rotate':v=(0,0);omega=.5;schedule=None
    else:raise ValueError('Unknown container motion')
    v=tuple(v[i]+boost[i] for i in range(2))
    if schedule:
        for c in schedule:c['velocity']=[c['velocity'][i]+boost[i] for i in range(2)]
    balls=[ball(((i-(side-1)/2)*spacing,(j-(side-1)/2)*spacing),radius=r,velocity=boost,
                friction=friction,id=f'b{j*side+i}') for j in range(side) for i in range(side)]
    return {'id':f'grid_{side*side}_{motion}','duration':duration,'gravity':list(gravity),'collision_skin_m':.01,
        'bodies':[container(half,half,velocity=v,omega=omega,schedule=schedule,friction=friction),*balls],
        'container':{'half_extents_m':[half,half],'radius_m':r},'parameter_provenance':'synthetic_declared',
        'material_authenticity':'Idealized coefficients, not material measurements'}


def packed_row_contact_system(count=64, speed=1, sparse=False):
    """Frozen exact contacts for a closed, driven row; no friction or restitution."""
    import numpy as np
    radius=.1;half=count*radius
    centers=[[-half+radius+2*radius*i,0] for i in range(count)]+[[0,0]]
    contacts=[(0,count,[-half,0],[1,0])]
    contacts += [(i+1,i,[-half+2*radius*(i+1),0],[1,0]) for i in range(count-1)]
    contacts += [(count-1,count,[half,0],[-1,0])]
    velocity=np.zeros(3*(count+1));velocity[-3]=speed
    if sparse:
        from research.sparse_contact import assemble_sparse
        return assemble_sparse(centers,[1]*count+[float('inf')],[.005]*count+[float('inf')],contacts),velocity
    from research.contact_solver import assemble_planar
    inverse,G,_=assemble_planar(centers,[1]*count+[float('inf')],[.005]*count+[float('inf')],contacts)
    return inverse,G,velocity
