"""Synthetic SI 3D validation scenes, not experimentally calibrated materials."""
import math
import numpy as np


def sphere(position, radius=.1, mass=1., **kwargs):
    return dict(position=list(position),shapes=[dict(kind='sphere',radius=radius,density=mass/(4*math.pi*radius**3/3))],**kwargs)


def wall_impact(speed=20., restitution=1.):
    return dict(duration=.1,gravity=[0,0,0],bodies=[dict(type='kinematic',position=[-1,0,0],velocity=[speed,0,0],friction=0,restitution=1,shapes=[dict(kind='box',half_extents=[.05,2,2])]),sphere([0,0,0],friction=0,restitution=restitution)])


def driven_row(count=16,speed=20.,axis=0):
    """Initially touching spheres: inelastic driven row must reach wall speed."""
    direction=np.eye(3)[axis];bodies=[dict(type='kinematic',position=(-.15*direction).tolist(),velocity=(speed*direction).tolist(),friction=0,shapes=[dict(kind='box',half_extents=[.05 if k==axis else 1 for k in range(3)])])]
    bodies += [sphere(.2*k*direction,friction=0) for k in range(count)]
    return dict(duration=.04,gravity=[0,0,0],bodies=bodies)


def container(side=3,speed=20.,shake=False,shape='sphere',seed=42):
    """A six-wall 3D container; driven walls, independently integrated contents."""
    radius=.1;spacing=.205;half=side*spacing/2+.045
    walls=[]
    for axis in range(3):
        for sign in [-1,1]:
            center=np.zeros(3);center[axis]=sign*(half+.025)
            ext=np.full(3,half+.05);ext[axis]=.025
            walls.append(dict(kind='box',half_extents=ext.tolist(),center=center.tolist()))
    commands=[]
    if shake: commands=[dict(time_s=.04,velocity=[-speed,0,0]),dict(time_s=.08,velocity=[speed,0,0])]
    box=dict(type='kinematic',position=[0,0,0],velocity=[speed,0,0],velocity_schedule=commands,friction=math.sqrt(.4),shapes=walls)
    rng=np.random.default_rng(seed);bodies=[box]
    for x in range(side):
        for y in range(side):
            for z in range(side):
                position=(spacing*(np.array([x,y,z])-(side-1)/2)).tolist()
                if shape=='sphere':body=sphere(position,friction=math.sqrt(.4))
                elif shape=='box':body=dict(position=position,friction=math.sqrt(.4),shapes=[dict(kind='box',half_extents=[.085,.095,.075],density=1/(8*.085*.095*.075))])
                else:
                    # Convex, asymmetric 3D polyhedra; support radius <=.1m.
                    points=rng.normal(size=(12,3));points/=np.linalg.norm(points,axis=1)[:,None];points*=rng.uniform(.075,.1,size=(12,1))
                    body=dict(position=position,friction=math.sqrt(.4),shapes=[dict(kind='hull',vertices=points.tolist(),density=500.)])
                bodies.append(body)
    return dict(duration=.12,gravity=[0,0,-9.81],bodies=bodies),half
