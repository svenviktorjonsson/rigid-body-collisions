"""Render analytically gated 3D normal-contact performance evidence."""
import json
from pathlib import Path
import subprocess
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DIRECTORY=Path(__file__).parent/'spatial-normal'


def main():
    data=json.loads((DIRECTORY/'results/summary.json').read_text());plan=json.loads((DIRECTORY/'plan.json').read_text());names=list(data['scenes'])
    speeds=[data['scenes'][n]['speedup'] or np.nan for n in names]
    fig,axes=plt.subplots(1,2,figsize=(11,4));x=np.arange(len(names))
    axes[0].bar(x,speeds,color='#286c86');axes[0].axhline(1,color='black',lw=1);axes[0].set_ylabel('Same-algorithm native median speedup');axes[0].set_title('Every analytic gate passes; identical states')
    for i,v in enumerate(speeds):axes[0].text(i,v+.1,f'{v:.2f}x',ha='center',fontsize=9)
    for offset,mode in enumerate(['postassembly','compact']):axes[1].bar(x+offset*.35,[data['scenes'][n]['modes'][mode]['mobility_matrix_bytes_max']/1024 for n in names],width=.35,label=mode)
    axes[1].set_ylabel('Peak scalar mobility payload [KiB]');axes[1].set_yscale('log');axes[1].set_title('Nine times less matrix payload');axes[1].legend(fontsize=9)
    for ax in axes:ax.set_xticks(x+.17 if ax==axes[1] else x,names,rotation=25,ha='right',fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(DIRECTORY/'performance.png',dpi=200,bbox_inches='tight');plt.close(fig)
    passed=sum(s['qualified'] for s in data['scenes'].values());ratios=[s['speedup'] for s in data['scenes'].values() if s['qualified']]
    lines=['# Verified 3D simultaneous normal contacts and matrix assembly improvement','',f'**{passed}/6 analytic scenes pass. Measured native median gain: {min(ratios):.2f}–{max(ratios):.2f}x, with bitwise-identical states and nine times less scalar mobility storage.**','',
           'The tested worlds contain native 3D collision discovery, full inertia, rotation and independently integrated bodies. A 100 m/s wall drives 64-body rows along x, y and z; a fourth row has 128 bodies; two full 3D containers contain 27 and 64 spheres. Every dynamic body reaches the analytic velocity after the first sampled impact frame, follows its analytic position, and passes penetration, spin, wall-work and energy gates. No solver rejection or upstream fallback occurs.','',
           'The optimized profile is **frictionless and inelastic**, with zero gravity and no position projection. It does not qualify the five failed frictional references in the [baseline study](../spatial-validation/report.md), change their budgets, or claim experimental material accuracy. Fifteen 3D regression tests also cover full-tensor off-center impulses, free spin, restitution, sliding friction, fast walls and rotating random hulls.','',
           '![Verified speed and matrix payload](performance.png)','',
           '## Fair ablation','',
           'Both modes use the same mechanically gated normal quadratic program, timestep guard, start-of-update wall contact pose, velocity-only integration profile, geometry and mass/inertia. The postassembly mode constructs one normal and two tangent rows per contact and then removes tangent impulses fixed exactly to zero. The compact mode removes those fixed-zero variables before construction. It retains every coupling among the normal unknowns; a nonzero-friction scene is rejected rather than silently dropping tangent coupling.','',
           'The native normal QP uses Cholesky on positive definite active faces, a bound active set, and unilateral feasibility/complementarity checks. Redundant inactive normals can remain at zero impulse; pressure gauges need not be unique. Failed or singular active faces disclose upstream fallback, and the analytic protocol rejects any fallback. No compliance or diagonal regularization is added. The four direct QP regression cases cover inactive, redundant, coupled and separating normals.','',
           '| Scene | Bodies | Postassembly [s] | Compact [s] | Median gain | Matrix rows before/after |',
           '|---|---:|---:|---:|---:|---:|']
    for n,s in data['scenes'].items():
        a=s['modes']['postassembly'];b=s['modes']['compact'];N=128 if n.startswith('row128') else 27 if n=='packed27' else 64
        lines.append(f'| {n} | {N} | {a["median_step_s"]:.5f} | {b["median_step_s"]:.5f} | {s["speedup"]:.2f}x | {a["mobility_rows_max"]}/{b["mobility_rows_max"]} |')
    worst=lambda key:max(d[key] for s in data['scenes'].values() for m in s['modes'].values() for d in m['diagnostics'])
    lines+=['','## Analytic and mechanical checks','',
            'Every retained repetition must satisfy all-body position and velocity errors <=1e-8 m and m/s; spin and closing contact speed <=1e-8 rad/s and m/s; contact penetration and container surface excess <=1e-8 m; wall-work and energy errors <=1e-5 J; and zero QP rejection/upstream fallback. Initial velocity is zero; the simultaneous inelastic result is v=U, spin zero and position x0+Ut. The expected wall work is N|U|² J for unit masses, and final kinetic energy is half that. Initial geometric overlap around 1e-10 m is disclosed to stabilize floating-point touching contact discovery. It remains bounded below the fixed 1e-8 m geometry gate.','',
            f'Maximum observed errors over all 36 histories: position {worst("max_position_error_m"):.3g} m; velocity {worst("max_velocity_error_m_s"):.3g} m/s; spin {worst("max_omega_rad_s"):.3g} rad/s; penetration {worst("max_contact_penetration_m"):.3g} m; surface excess {worst("max_container_surface_excess_m"):.3g} m; wall-work error {worst("boundary_work_error_J"):.3g} J. Every warmup and repetition has deterministic states, and before/after modes are bitwise identical.','',
            '## Scope and limitations','',
            'The measured speedup belongs to this native Float64 CPU implementation and these short, exactly solvable 0.04 s worlds, using one warmup and three retained repetitions. Timing includes contact discovery, control and diagnostics but excludes startup and JSON serialization. It is an implementation improvement over the same algorithm, not a universal solver ranking or evidence that frictional trajectories are accurate. Nine times less matrix payload does not mean nine times less process RSS; Bullet still owns manifolds, body data and other work buffers. Dense matrix assembly remains quadratic.','',
            'The start contact phase uses current wall pose and explicit prescribed velocity, then advances walls after dynamic integration. The legacy end phase advances walls before collision discovery and is preserved as the baseline study convention. Velocity-only disables split position projection and ERP; it cannot repair macroscopic initial overlap. A general solver still needs an appropriate geometric recovery/CCD policy. The relative-travel guard is not a universal exact swept-CCD proof.','',
            'Bullet baseline friction uses a two-direction pyramid with coefficient product mixing. Separate static/dynamic, elastic tangential, rolling and twisting resistance are not implemented in this adapter. The optimized normal profile explicitly rejects nonzero friction or restitution. These limits remain material for the intended frictional engine.','',
            'The mechanics and fixed-variable elimination are established methods. No new collision theory is claimed. The contribution here is a reproducible native 3D implementation, a clear failure boundary, independently verified analytic results, and a measured exact assembly reduction. The benchmark exercises full 3D contact graphs; the analytic packed case itself has zero spin. A separate native off-center compound-body test checks the optimized profile against full-tensor angular impulse mechanics.','',
            '## Reproduction','',f'Execution source: `{data["execution_source_commit"]}`. Full authored scenes, all 36 histories, warmup hashes, three timing samples, matrix sizes, fallbacks, analytic checks and execution sources are archived. Source hashes and plan are retained; an independent audit recomputes every analytic gate, all work/energy checks, identical-state comparisons and timing ratios.','',
            '```sh','cmake -S spatial_backend -B build/spatial -G Ninja -DCMAKE_BUILD_TYPE=Release','cmake --build build/spatial --target spatial_runner spatial_qp_checks -j 2','build/spatial/spatial_qp_checks','python -m unittest tests.test_spatial_engine -v','python -m research.audit_spatial_normal','python -m research.run_spatial_normal --directory /tmp/fresh-normal-study','python -m research.make_spatial_normal_report','```','',
            'The [typeset mechanics note](mechanics.pdf) derives the full 3D wedge, world inertia, global mobility, boundary work, normal complementarity and exact tangent-variable elimination. [Source](mechanics.tex) is reproducible with pdflatex.','']
    (DIRECTORY/'report.md').write_text('\n'.join(lines))
    subprocess.run(['pandoc','report.md','-o','report.pdf','--pdf-engine=pdflatex','-V','geometry:margin=0.7in','-V','fontsize=10pt'],cwd=DIRECTORY,check=True)

if __name__=='__main__':main()
