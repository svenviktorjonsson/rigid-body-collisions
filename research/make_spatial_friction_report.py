"""Render the frozen circular-friction study; failed evidence is not reclassified."""
import json
from pathlib import Path
import subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
DIRECTORY=Path(__file__).parent/'spatial-friction'

def main():
    data=json.loads((DIRECTORY/'results/summary.json').read_text());plan=json.loads((DIRECTORY/'plan.json').read_text())
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    colors={'coarse':'#137c9d','medium':'#ef9c32','fine':'#8057a5'};qualified=[]
    for name,scene in data['scenes'].items():
        if not scene['reference_qualified']:continue
        qualified.append(name)
        for mode,result in scene['candidates'].items():
            value=max(result['errors'][0][key]/limit for key,limit in plan['trajectory_budget'].items())
            axes[0].scatter(result['median_step_s'],value,c=colors[mode],marker='o' if result['qualified'] else 'x',s=65)
    axes[0].set(xscale='log',yscale='log',xlabel='Native median seconds (3 retained runs)',ylabel='Worst normalized trajectory error',title='Only qualified references permit comparison')
    axes[0].axhline(1,color='black',ls='--',lw=1)
    for mode,c in colors.items():axes[0].scatter([],[],c=c,label=mode)
    axes[0].legend()
    for i,name in enumerate(qualified):
        scene=data['scenes'][name];choice=scene['choice'];fractions=scene['candidates'][choice]['native_fast_solve_fraction']
        axes[1].barh(i,100*np.mean(fractions),color=colors[choice]);axes[1].text(100*np.mean(fractions)+1,i,f'{100*np.mean(fractions):.1f}%',va='center')
    axes[1].set(yticks=range(len(qualified)),yticklabels=[name.replace('_',' ') for name in qualified],xlim=(0,108),xlabel='Contact solves accepted within 8 sweeps (%)',title='Residual-driven effort at the selected setting')
    fig.savefig(DIRECTORY/'accuracy-effort.png',dpi=180);plt.close(fig)
    rows=[]
    for name,r in data['scenes'].items():
        choice=r['choice'];cost=r['candidates'][choice]['median_step_s'] if choice else None
        rows.append(f"| {name.replace('_',' ')} | {'Pass' if r['reference_qualified'] else 'Fail'} | {choice or 'None'} | {f'{cost:.6f}' if cost is not None else '—'} |")
    text=r'''# Circular 3D friction under slow and rapid container motion

The project now has a circular, coupled Coulomb contact solver with explicit
residual-driven iteration work, prescribed wall reversals and full 3D inertia.
The frozen study qualifies **three of six** trajectory references. It does not
qualify fast shaking with 27 spheres or either random-hull case.

The native off-centre test found and fixed the initial world-inertia frame:
`setCenterOfMassTransform` initializes the rotated inverse inertia before the
first collision. A separate eight-body packed test verifies that a prescribed
100 to −100 m/s reversal reaches contacts on the first update after its command.
Seven 3D friction regression tests check diagonal sliding, sticking, slow and
rapid transition to rolling, off-centre full-tensor contact, 27-body shaking and
rejection on an exhausted contact budget. Eight native analytic friction cases
check circular sliding and retained normal/tangent coupling.

## Contact equations and scope

For each contact, the impulse has one normal and two tangential components.
Let K be the assembled contact mobility, retaining all inter-body, lever-arm and
normal/tangent coupling. The frozen velocity problem is

$$w=Kp-b,\qquad p_n\geq0,\quad w_n\geq0,\quad p_nw_n=0.$$

Sticking and sliding use a circular friction limit:

$$\|p_t\|\leq\mu p_n,\qquad
\begin{cases}
w_t=0,&\|p_t\|<\mu p_n,\\
p_t=-\mu p_n\,w_t/\|w_t\|,&\|w_t\|>0.
\end{cases}$$

Normal complementarity is separate from the friction disk. An associated cone
energy QP would alter the normal law and is not substituted here. One coefficient
covers sticking and sliding. The current lane is inelastic (normal restitution
zero) and has no tangential spring or independent angular contact couple.
Those energy-storing channels are tested separately in
[the elastic-contact study](../elastic-patch/report.pdf).

Block iterations propagate through sparse columns of K; exact 2D disk subproblems
and semismooth Newton acceleration use the same equations. Dense Bullet assembly
is still quadratic; Newton acceleration is limited to at most 512 rows. Every
eight sweeps, a velocity-scaled projection residual is checked against 1e−8 m/s.
The 4096-sweep argument is a maximum budget. Unconverged runs reject instead of
falling back to the upstream friction pyramid. Split position correction solves
normal equations only and applies no tangential correction.

The declared contact geometry tolerance is 1e−9 m. Positive gaps within this
tolerance are treated as touching; tiny penetrations within it are not projected.
This prevents conflicting gap/time targets at nearly redundant face points.
It is a numerical geometry tolerance, not material elasticity. Relative linear
and angular tip speeds adjust the native travel-guard timestep; its rule is a
conservative heuristic, not a proof of swept collision detection for every shape.

## Frozen results

All scenes last 0.12 s, sampled every 0.01 s. Synthetic pair friction is 0.4,
normal restitution zero and gravity −9.81 m/s². Wall speeds are 0.2 or 20 m/s;
shaking reverses at 0.04 and 0.08 s. Random rotating hulls additionally use
10 rad/s wall rotation. The archived plan and execution source were committed
before execution. There are 76 retained attempts: 52 complete histories and
24 explicit native rejections. None is discarded.

| Scene | Reference gate | Selected effort | Native median seconds |
|---|---|---|---:|
'''+ '\n'.join(rows)+r'''

![Trajectory budgets and effort](accuracy-effort.png)

The trajectory budgets are 5 mm position RMS, 0.05 m/s velocity RMS,
0.1 rad/s angular-velocity RMS and 0.01 rad quaternion geodesic RMS. Both
adjacent edges of the three finest declared refinement levels must meet **one
quarter** of each budget. The complete coarser ladder remains visible; it is
not a requirement that every coarse level qualify. No failed refinement level
is skipped. Native physical gates require unit quaternion error at most 1e−12,
container surface excess at most 2 mm at every internal update and total energy
change minus boundary work at most 1 J.

At qualified references, candidates use travel fractions 0.15, 0.06 and 0.015
with the same material/contact equations. Three retained repetitions give each
candidate's median; repeated states are required to be bitwise identical. The
least-cost candidate that passes every gate is selected. No recommendation is
made at an unqualified reference. This selection is offline; the native iteration
and travel guard adapt during integration, but no certified local-error controller
or cache-preserving rollback is claimed.

Fast 27-sphere shaking passes frozen contact residuals yet fails trajectory
refinement: its last angular RMS difference is about 0.499 rad/s and velocity
RMS difference about 0.079 m/s. This is direct evidence that a successful frozen
contact solve and containment are insufficient to certify a trajectory. Both
random-hull cases reject all twelve attempts each.

A subsequent independent review identified an upstream RHS inconsistency:
normal equations and body writeback include a gyroscopic angular-velocity
increment, while upstream tangent equations omit it. The adapter correction and
zero-free-slip two-body regression are a **separate subsequent change**. This
archive retains the original 24 hull rejections; it is not retrospectively fixed.
Its accepted spherical cases have zero anisotropic gyroscopic increment; box
spin is negligible. The matrix energy check in this frozen revision describes
the assembled contact problem; global energy/work is also checked independently.

All timing runs used the same workspace while other research ran concurrently.
These are descriptive costs, not a universal speedup, controlled hardware ranking
or Vektor compiler benchmark. All parameters are synthetic; numerical agreement
and fitting do not authenticate rubber or other materials.

## Reproduce and audit

Execution source: `7c279768f6518e66b1ee620d44bd9b37309b1b5f`.
Pinned unmodified upstream: Bullet 3.25 Float64,
`2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5`.
`results/execution-source.zip` contains the exact code/plan with SHA-256 receipts;
`results/scenes.json` contains full geometry; `results/traces.zip` retains every
completed history and rejection. `results/summary.json` records all diagnostics,
refinement edges, candidate qualifications, timing and source/binary hashes.

Run `python -m research.audit_spatial_friction` to independently recompute
physical gates, refinement, candidate selection and timing receipts. Historical
3D normal gains and the earlier failed 102-history friction study are unchanged.
The Vektor integration target remains `vektor-flow/bootstrap`, paired with spec;
this public Python/C++ evidence is not a completed Vektor port.
'''
    (DIRECTORY/'report.md').write_text(text.replace('−','-'))
    subprocess.run(['pandoc','report.md','-o','report.pdf','--pdf-engine=pdflatex','-V','geometry:margin=22mm','-V','fontsize=10pt'],cwd=DIRECTORY,check=True)
if __name__=='__main__':main()
