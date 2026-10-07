"""Evidence-based numerical-effort selection for the unchanged 3D contact law.

Native Coulomb solves adapt iteration work per island to an explicit residual.
The native travel guard adapts update duration to translation and angular speed.
This module qualifies whole-trajectory refinement OFFLINE; it is not an online
proof of truncation error, a material-fitting routine or a cache-preserving retry.
"""
import subprocess
from spatial_engine import run, errors
from research.spatial_metrics import diagnostics

BUDGET=dict(position_m=.005,velocity_m_s=.05,omega_rad_s=.1,orientation_rad=.01)
PHYSICAL=dict(quaternion_norm_error=1e-12,energy_change_minus_boundary_work_J=1.,container_surface_excess_m=.002)


def physical(scene,result):
    half=scene.get('container_interior_half_extents_m')
    if half is not None and len(set(half))!=1:
        raise ValueError('Offline containment audit currently requires a cubic container')
    d=diagnostics(scene,result,half[0] if half else None)
    return d,all(d[k]<=limit for k,limit in PHYSICAL.items() if k in d)


def qualify(scene,runs,reference_names,candidate_names,*,budget=None,costs=None):
    """Two adjacent reference edges must meet quarter budget AND physical gates.

    Rejections remain in runs as {'rejected': reason}; no skipped failed level may
    qualify a reference. No candidate is recommended against an unqualified
    reference. Native tolerance and geometry must be checked before calling this
    function (the study/audit do so); residuals do not certify trajectory error.
    """
    budget=BUDGET if budget is None else budget
    if len(reference_names)<3:raise ValueError('At least three consecutive refinement levels required')
    diagnostic={};eligible={}
    for name,r in runs.items():
        if 'rejected' in r:eligible[name]=False
        else:diagnostic[name],eligible[name]=physical(scene,r)
    edges=[];reference_ok=True
    for left,right in zip(reference_names[:-1],reference_names[1:]):
        if not eligible.get(left) or not eligible.get(right):
            edges.append(dict(left=left,right=right,passed=False,error=None));reference_ok=False
        else:
            error=errors(runs[left],runs[right]);passed=all(error[k]<=budget[k]/4 for k in budget)
            edges.append(dict(left=left,right=right,passed=passed,error=error));reference_ok &= passed
    # Only the three finest declared levels qualify the reference; the entire
    # coarser ladder stays visible in edges. This criterion is frozen in the plan.
    reference_ok=all(e['passed'] for e in edges[-2:])
    reference=reference_names[-1];candidates={}
    for name in candidate_names:
        error=errors(runs[reference],runs[name]) if reference_ok and eligible.get(name) else None
        passed=bool(error and all(error[k]<=budget[k] for k in budget))
        candidates[name]=dict(qualified=passed,error=error,physical_pass=eligible.get(name,False))
    passing=[name for name,r in candidates.items() if r['qualified']]
    choice=min(passing,key=lambda name:costs[name] if costs else runs[name]['step_s']) if passing else None
    return dict(reference_qualified=reference_ok,reference=reference,edges=edges,
                candidates=candidates,choice=choice,diagnostics=diagnostic)


def run_ladder(scene,*,dt=.01,fractions=(.06,.03,.015),candidate_fractions=(.15,.06),iterations=4096,budget=None):
    """Re-run whole scene for each declared effort; retain every rejection reason."""
    runs={};references=[];candidates=[]
    for lane,values,names in [('reference',fractions,references),('candidate',candidate_fractions,candidates)]:
        for i,fraction in enumerate(values):
            name=f'{lane}_{i}';names.append(name)
            try:runs[name]=run(scene,dt=dt,travel_fraction=fraction,iterations=iterations,
                               solver='coulomb',kinematic_contact_phase='start')
            except subprocess.CalledProcessError as e:runs[name]=dict(rejected=e.stderr.strip(),exit_code=e.returncode)
    return dict(runs=runs,receipt=qualify(scene,runs,references,candidates,budget=budget))
