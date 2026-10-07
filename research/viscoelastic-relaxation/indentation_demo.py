"""Local surface geometry control; no per-event experimental crater match.

User n is the actual local contact normal. Experimental rock R_n is measured
against the mean slab normal, so apparent R_n>1 need not create total energy.
This example neither fits crater dimensions to outcomes nor predicts real rocks.
"""
import json,math
from pathlib import Path
import numpy as np


def run():
    craters=[dict(diameter_m=.035,depth_m=.015),dict(diameter_m=.017,depth_m=.016),
             dict(diameter_m=.023,depth_m=.006),dict(diameter_m=.021,depth_m=.005)]
    for crater in craters:
        crater['conical_wall_angle_deg']=math.degrees(math.atan(2*crater['depth_m']/crater['diameter_m']))
    # Synthetic velocities, mass and local restitution, not source measurements.
    initial=np.array([3.,-1.,0.]);mean_n=np.array([0.,1.,0.]);e=.5;mass=1.
    records=[]
    for theta in (0.,15.,30.,45.):
        angle=math.radians(theta);n=np.array([-math.sin(angle),math.cos(angle),0.])
        approach=n@initial;assert approach<0
        impulse=-(1+e)*mass*approach*n
        outgoing=initial+impulse/mass
        before=.5*mass*(initial@initial);after=.5*mass*(outgoing@outgoing)
        local_e=-(outgoing@n)/approach;loss=.5*mass*(1-e*e)*approach**2
        assert abs(local_e-e)<1e-12 and abs(after+loss-before)<1e-12
        apparent=(outgoing@mean_n)/abs(initial@mean_n)
        records.append(dict(local_angle_deg=theta,initial_velocity=initial.tolist(),
            outgoing_velocity=outgoing.tolist(),linear_impulse=impulse.tolist(),
            local_e_n=local_e,mean_slab_apparent_normal_ratio=float(apparent),
            initial_energy_J=before,outgoing_energy_J=after,dissipation_J=loss))
    return dict(crater_source='https://nhess.copernicus.org/articles/18/3045/2018/nhess-18-3045-2018-f12.pdf',
        crater_dimensions_read_from_figure_12=True,crater_records=craters,
        dimensions_matched_to_individual_collision_rows=False,
        conical_geometry_is_simplification=True,synthetic_controls=records,
        apparent_ratio_above_one_with_passivity=any(r['mean_slab_apparent_normal_ratio']>1 for r in records),
        body_is_rigid=True,point_sphere_example_has_zero_normal_force_torque=True,
        independent_angular_closure_validated=False,experimental_prediction=False)


if __name__=='__main__':print(json.dumps(run(),indent=2))
