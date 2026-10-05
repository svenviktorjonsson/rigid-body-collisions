"""Independent physical diagnostics over retained 3D trajectories."""
import itertools
import numpy as np
from scipy.spatial.transform import Rotation
from spatial_engine import prepare, energy


def containment(scene,result,half):
    """Max authored surface excess beyond the six container interior planes.

    Tests every box/hull vertex and exact sphere support in container axes. This
    is stronger than checking centers against an enclosing sphere approximation.
    """
    bodies,_,_,axes,_=prepare(scene);states=np.asarray(result['states']);maximum=-np.inf
    for frame in states:
        container_rotation=Rotation.from_quat(frame[0,3:7]).as_matrix()
        for i,b in enumerate(bodies[1:],start=1):
            R=Rotation.from_quat(frame[i,3:7]).as_matrix()@axes[i]
            for shape in b['shapes']:
                center=frame[i,:3]+R@np.asarray(shape['center'])
                relative=container_rotation.T@(center-frame[0,:3])
                if shape['kind']=='sphere':excess=np.max(abs(relative))+shape['radius']-half
                else:
                    vertices=np.asarray(shape['vertices']) if shape['kind']=='hull' else np.asarray(list(itertools.product([-1,1],repeat=3)))*shape['half_extents']
                    S=container_rotation.T@R@Rotation.from_quat(shape['orientation']).as_matrix()
                    local=vertices@S.T+relative
                    excess=np.max(abs(local))-half
                    if shape['kind']=='hull':excess+=scene.get('margin_m',0)
                maximum=max(maximum,float(excess))
    return maximum


def diagnostics(scene,result,half=None):
    s=np.asarray(result['states']);m=np.asarray(result['mass']);g=np.asarray(scene.get('gravity',[0,0,-9.81]))
    total=energy(result)-np.einsum('tij,j,i->t',s[:,:,:3],g,m)
    out=dict(quaternion_norm_error=float(np.max(abs(np.linalg.norm(s[:,:,3:7],axis=2)-1))),energy_change_minus_boundary_work_J=float(total[-1]-total[0]-result['boundary_work_J']),max_contact_penetration_m=result['max_contact_penetration_m'],max_closing_contact_speed_m_s=result['max_closing_contact_speed_m_s'],coupled_fallbacks=result['coupled_fallbacks'])
    if half is not None:
        out['sampled_container_surface_excess_m']=containment(scene,result,half)
        out['container_surface_excess_m']=max(out['sampled_container_surface_excess_m'],result['max_container_surface_excess_m'])
    return out
