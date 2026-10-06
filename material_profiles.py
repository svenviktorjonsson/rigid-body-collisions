"""Source-backed pair profiles, deliberately excluding project-fitted values.

A profile identifies a specimen/surface/geometry, not an intrinsic per-body
coefficient. It supplies documented nominal inputs, not a promise of universal
accuracy or a generic mixing rule. Unknown context stays unknown.
"""
from pathlib import Path
from dataclasses import dataclass
import json,math,copy

class MaterialDataUnavailable(ValueError):pass

@dataclass(frozen=True)
class MaterialPairProfile:
    id: str
    normal_restitution: float
    tangential_restitution: float
    sliding_friction: float
    metadata: dict
    @property
    def solver_parameters(self):
        return {'normal_restitution':self.normal_restitution,'tangential_restitution':self.tangential_restitution,'pair_friction':self.sliding_friction}

def documented_profile(profile_id):
    """Select an explicit published specimen/pair profile; no guessed fallback."""
    path=Path(__file__).parent/'research/documented-materials/catalog.json'
    catalog=json.loads(path.read_text())
    row=next((r for r in catalog['profiles'] if r['id']==profile_id),None)
    if row is None:raise MaterialDataUnavailable('No complete documented profile for '+str(profile_id)+'. Specify a catalog profile id; missing coefficients are never fitted or guessed.')
    en=row['normal_restitution']['value'];et=row['tangential_restitution']['value'];mu=row['sliding_friction']['value']
    if not (all(math.isfinite(v) for v in [en,et,mu]) and 0<=en<=1 and -1<=et<=1 and mu>=0):raise MaterialDataUnavailable('Published profile incompatible with supported contact convention/bounds; requires separate modeling, not silent clipping.')
    return MaterialPairProfile(profile_id,en,et,mu,copy.deepcopy(row))

def run_documented_pair(scene,profile_id,**options):
    """Predict one isolated3D pair with a fixed published parameter triple.

    The square root of pair friction is placed on both sides so Bullet
    product mixing realizes that measured pair value. No per-body material law is inferred.
    Heterogeneous groups require per-contact profile resolution, not this wrapper.
    """
    from spatial_engine import run
    profile=documented_profile(profile_id);cfg=copy.deepcopy(scene)
    if len(cfg.get('bodies',[]))!=2:raise ValueError('This wrapper supports one two-body pair; no global coefficient substitution for heterogeneous contact graphs.')
    for key in ['normal_restitution','tangential_restitution','solver']:
        if key in options:raise ValueError('Profile-owned option cannot be overridden: '+key)
    geometry=profile.metadata['geometry']
    sphere_count=sum(len(b.get('shapes',[]))==1 and b['shapes'][0].get('kind')=='sphere' for b in cfg['bodies'])
    if (geometry=='sphere_sphere' and sphere_count!=2) or (geometry=='sphere_plane' and sphere_count!=1):raise ValueError('Scene must use the documented geometry; irregular shapes are not an exact profile validation.')
    for body in cfg['bodies']:
        if len(body.get('shapes',[]))==1 and body['shapes'][0].get('kind')=='sphere':
            radius=float(body['shapes'][0]['radius'])
            if not math.isclose(2*radius,profile.metadata['sphere_diameter_m'],rel_tol=1e-9):
                raise ValueError('Sphere size differs from documented profile; do not silently extrapolate the coefficients.')
            density=profile.metadata.get('sphere_density_kg_m3')
            if density is not None and not math.isclose(float(body['shapes'][0].get('density',1)),density,rel_tol=1e-9):
                raise ValueError('Sphere density differs from documented specimen; do not silently change the material.')
    if geometry=='sphere_plane':
        plane=next(b for b in cfg['bodies'] if not (len(b.get('shapes',[]))==1 and b['shapes'][0].get('kind')=='sphere'))
        if plane.get('type')!='kinematic':raise ValueError('Plane profile requires prescribed support; a freely moving support changes the experimental geometry.')
    for body in cfg['bodies']:body['friction']=math.sqrt(profile.sliding_friction)
    result=run(cfg,solver='coulomb',normal_restitution=profile.normal_restitution,tangential_restitution=profile.tangential_restitution,**options)
    result['documented_material_profile']={'id':profile.id,'source':profile.metadata['source_url'],'parameters':profile.solver_parameters,'context':profile.metadata['conditions'],'full_experimental_accuracy_qualified':False}
    return result
