"""Render the completed fix validation, including unresolved packed trajectories."""
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Polygon
import numpy as np

ROOT=Path(__file__).parent/'random-shape-resolution'


def report():
    output=ROOT/'results';data=json.loads((output/'summary.json').read_text());plan=json.loads((ROOT/'plan.json').read_text())
    scenes=json.loads((output/'scenes.json').read_text());records={r['trace']:r for r in data['records']}
    with zipfile.ZipFile(output/'traces.zip') as z:
        traces={r['trace']:json.loads(z.read(r['trace']+'.json'))['result'] for r in data['records']}
    q=data['concave_selections'];qualified={n:s for n,s in q.items() if s['status']=='qualified'}
    ratios=[]
    lines=['# Random-shape fixes and remaining accuracy limits', '',
        'Executed 5 October 2026. Both original native geometry rejections are fixed, and both '
        'original concave drops now have qualified references. **The two packed-box trajectories '
        'remain unresolved.** This is 2D numerical verification with synthetic coefficients; no '
        '3D, material-authentication or universal accuracy claim.', '',
        '## Concrete changes', '',
        '- Preserve shallow convex corners through scaled native hull authoring, then restore '
        'the original core dimensions. No smoothing or corner removal, and no world-unit/solver-slop change.',
        '- Merge adjacent equal-material convex fixtures only when the hull equals their union. '
        'Preserve every original core through containment and equal area, exact mass/COM/inertia, '
        'and heterogeneous or stateful patch boundaries. The original star drops go from twelve '
        'triangles to six convex pieces; the two mixed boxes go from 150/134 fixtures to 87/79.',
        '- Add separately controlled block position iterations and analytic prescribed-wall pose '
        'correction. Prescribed velocity remains unchanged; contents are not repositioned.',
        '- Add an explicitly experimental full-Float64 diagnostic build with checked upstream '
        'archive and original/transformed source inventories. It is a locally transformed Box2D '
        '2.4.1 comparator, not an upstream-supported Float64 release or a general feature certification.',
        '- Add offline `fidelity.select()`: qualify supplied references on adjacent refinement '
        'edges, compare unchanged physical setups, then select the cheapest passing measured '
        'candidate. An unqualified reference produces no verified choice. This is not a universal '
        'online error estimator.', '',
        'The optional internal-point filter is also retained for diagnosis. It rejects reactions '
        'directed into another fixture core. It does not merge duplicate exterior manifolds and '
        'is off by default; it did not resolve packed refinement in exploratory controls.', '',
        '## Original concave cases and fresh seeds', '',
        'The original position/velocity/spin RMS reference budgets remain 0.005 m, 0.0125 m/s '
        'and 0.0125 rad/s; candidate budgets are four times larger. Collision updates refine '
        '128 → 256 → 512 at 64 velocity iterations, while velocity iterations refine '
        '32 → 64 → 128 at 512 collision updates. All four adjacent edges must pass. '
        'The original low-work failures remain archived, rather than overwritten.', '',
        'The new protocol was frozen after exploratory diagnosis and before this repeated study. '
        'Its controller settings were selected using the original cases, then frozen for two '
        'previously untested seeds, 99017 and 13579. Each concave mode has one warm-up and '
        'three repetitions. Three of four references qualify: both original cases and seed13579. '
        'Seed 99017 still fails and gets no verified choice. No thresholds or material values were '
        'changed after observing held-out results.', '',
        '| Seed | Reference qualified | Worst normalized refinement error | Cheapest passing setting | Controller error / candidate budget |',
        '|---:|---|---:|---|---:|']
    for name,selection in q.items():
        worst=max(r['normalized_error'] for r in selection['refinements'])
        adaptive=next((c for c in selection['comparisons'] if c['candidate']=='adaptive'),None)
        lines.append(f"| {name.split('_')[1]} | {selection['status']=='qualified'} | {worst:.4g} | "
                     f"{selection['choice'] or 'none'} | {adaptive['normalized_error'] if adaptive else 'unqualified'} |")
    lines += ['', 'The frozen controller uses 128 collision updates and 32 velocity iterations '
        'during high-motion phases (`travel_threshold=0.02`), then demotes with the existing '
        'contact/travel/dwell policy. All three qualified controller trajectories pass. Its cost '
        'includes controller work. It is slower than the retrospectively cheapest passing fixed '
        'setting in all three cases, so it is not recommended as a universal default.', '',
        '| Seed | Fixed128×32 median (ms) | Controller median (ms) | Fixed/controller ratio | Cheapest passing fixed median (ms) |',
        '|---:|---:|---:|---:|---:|']
    for name,selection in qualified.items():
        fixed=records[name+'__candidate_p128_s32']['median_s'];adaptive=records[name+'__adaptive']['median_s']
        best=records[name+'__'+selection['choice']]['median_s'];ratios.append(fixed/adaptive)
        lines.append(f"| {name.split('_')[1]} | {1000*fixed:.3f} | {1000*adaptive:.3f} | {fixed/adaptive:.3f} | {1000*best:.3f} |")
    lines += ['', f"Compared with keeping its own fine setting throughout, the controller uses "
        f"**{min(ratios):.2f}–{max(ratios):.2f}× less native time**, with the same physical parameters. "
        'This is a bounded result on three qualified scenes, not an arbitrary-shape speedup or '
        'superiority over the cheapest fixed setting. Single-thread BLAS/OMP; times exclude '
        'Python/process startup and serialization.', '',
        '## Packed boxes: still unqualified', '',
        'Both 36-body boxes were tested with exact convex partition merging, full native Float64 '
        'geometry/solver arithmetic, analytic wall motion and independent position refinement. '
        'The physical friction/restitution values and original accuracy gates were retained. '
        'Six edges refine collision updates16/32/64, velocity iterations32/64/128, and position '
        'iterations3/6/12. All six must pass. These are single diagnostic runs, with no timing '
        'speedup claim.', '',
        '| Seed | Worst normalized edge error | Position3/6/12 histories | Qualified |',
        '|---:|---:|---|---|']
    for name,value in data['packed_qualification'].items():
        worst=max(max(e['errors'][k]/v for k,v in plan['reference_budget'].items()) for e in value['refinements'])
        positional=max(max(e['errors'][k]/v for k,v in plan['reference_budget'].items()) for e in value['refinements'][-2:])
        lines.append(f"| {name.split('_')[1]} | {worst:.4g} | {'identical' if positional==0 else 'different'} | {value['qualified']} |")
    sensitivity=data['sensitivity']['errors'];precision={p['precision']:p for p in data['precision_controls']}
    lines += ['', 'Position refinement yields identical histories at these settings, while collision '
        'and velocity refinement still fail badly. Thus the tested position-iteration count alone '
        'does not explain the remaining failure. Fixing wall drift, scalar precision and reducing '
        'partition seams is also insufficient. Exploratory runs at512collision updates and '
        '2048 velocity iterations separately remained unqualified; those logs informed the frozen '
        'fresh protocol and are not substituted for these archived controls.', '',
        'A separate sensitivity control changes one initial body position by1µm at identical '
        'fidelity. It observes an initial printed-state displacement of '
        f"{sensitivity['initial_max_position_difference_m']:.4g}m, then full-trajectory RMS "
        f"differences of {sensitivity['rms_position_m']:.5g}m, {sensitivity['rms_velocity_m_s']:.5g}m/s "
        f"and {sensitivity['rms_spin_rad_s']:.5g}rad/s. This is a **different initial condition**, "
        'not an accuracy comparison. It establishes strong sensitivity in this scene, not a proof '
        'that convergence is impossible or a justification for replacing individual trajectory '
        'budgets with aggregate metrics after the fact.', '',
        '## Precision and boundary controls', '',
        f"At 512 updates/frame, the free-fall final velocity error is "
        f"{precision['float32']['analytic_velocity_error_m_s']:.6g}m/s for Float32 versus "
        f"{precision['float64']['analytic_velocity_error_m_s']:.6g}m/s for the full-Float64 build. "
        'That validates the intended precision change on this analytic primitive. It does not '
        'remove semiimplicit integration truncation error or certify every upstream feature. '
        'The Float64 packed wall histories independently match the exact scheduled boundary '
        'path within1e-12m. The native32-bit analytic-wall regression also removes the dependence '
        'on collision-update count to within Float32 output resolution.', '',
        '## Remaining engineering work', '',
        'The packed case needs evolving body-pair manifold instrumentation, union-boundary '
        'feature identity and duplicate-exterior-contact handling, followed by integration of '
        'the coupled solver with residual/work checks and rollback on rejection. No new coupled '
        'world solver is claimed in this slice. An ensemble benchmark can be scientifically useful '
        'but must be declared separately; it cannot replace these failed pointwise gates. '
        'Held-out seed 99017 also needs further reference qualification.', '',
        '3D remains unimplemented and untested. It requires real 3D shapes/manifolds, full inertia '
        'tensors, two tangential directions, coupled cone friction and separate rolling/twisting '
        'models, then its own benchmarks. Algebraic dimensional extension is not a 3D validation.', '',
        '## Evidence and reproduction', '',
        f"Execution input/source: `{data['source_commit']}`. Controls implementation: "
        '`fba1c28d07d622d5074e57ece8f29bf94a48d8b9`; hull fix: '
        '`ff9e7d263d3888221faf08a935e05d10d0f1c346`. Retain60full histories, all timing '
        'samples, geometry, transformed-source inventory and independent audits. All86local '
        'tests and historical evidence audits pass. These are external physics tests, not Vektor '
        'compiler/native/WASM/GPU acceptance.', '',
        'Build both native comparators and `python -m research.build_precision_backend`; then run '
        '`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.run_shape_resolution`, '
        '`python -m research.audit_shape_resolution` and '
        '`python -m research.make_shape_resolution_report`.', '',
        '![Exact core partition](results/partition.png)', '',
        '![Concave reference qualification](results/concave-refinement.png)', '',
        '![Packed limits](results/packed-refinement.png)']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    with PdfPages(ROOT/'report.pdf') as pdf:
        fig,ax=plt.subplots(figsize=(10,6));ax.axis('off')
        text=(f'Random-shape fix validation — 2D\n\n'
              'Both original hull rejections fixed without changing core geometry.\n'
              'Both original concave-drop references now qualify.\n'
              'One of two additional seeds qualifies; one still fails.\n'
              'Both packed 36-body trajectories remain unqualified.\n\n'
              f'Controller: {min(ratios):.2f}–{max(ratios):.2f}× lower native time than its continuous fine setting.\n'
              'It is slower than the cheapest passing fixed setting in all three cases.\n'
              'Offline selector returns no accuracy-backed choice for failed references.\n\n'
              '60 retained histories; unchanged budgets; independent geometry/source audit.\n'
              'Full Float64, analytic wall motion and exact partition controls tested.\n'
              '86 local tests pass. No 3D, material validation or universal engine claim.\n\n'
              'Complete tables, source pins and remaining work: report.md and results/.')
        ax.text(.03,.96,text,va='top',fontsize=13,linespacing=1.7);pdf.savefig(fig);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4.5))
        old=scenes['random_42_mixed36_shake']['bodies'][1]['polygons']
        new=scenes['random_42_mixed36_shake_exact_partition']['bodies'][1]['polygons']
        for ax,parts,title in zip(axes,[old,new],['Original 12 triangle cores','Same union, 6 convex cores']):
            for i,p in enumerate(parts):ax.add_patch(Polygon(p['vertices'],facecolor=plt.cm.Set3(i%12/12),edgecolor='#444',alpha=.7))
            ax.plot(0,0,'k+');ax.set_aspect('equal');ax.set(xlim=(-.21,.21),ylim=(-.21,.21),title=title,xlabel='m',ylabel='m')
        fig.suptitle('Exact representation change: no boundary smoothing; mass and inertia preserved')
        fig.tight_layout();fig.savefig(output/'partition.png',dpi=170);pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(9,4))
        names=list(q);worst=[max(r['normalized_error'] for r in q[n]['refinements']) for n in names]
        ax.barh([n.split('_')[1] for n in names],worst,color=['#448355' if n in qualified else '#c44d4d' for n in names])
        ax.axvline(1,color='black',ls='--');ax.set_xscale('log');ax.set(xlabel='Worst normalized reference edge error (all edges must be ≤1)',ylabel='Seed',title='Original cases qualify; held-out seed99017 remains unresolved')
        fig.tight_layout();fig.savefig(output/'concave-refinement.png',dpi=170);pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4.5))
        labels=['collision16→32','collision32→64','velocity32→64','velocity64→128','position3→6','position6→12']
        for index,(name,value) in enumerate(data['packed_qualification'].items()):
            values=[max(e['errors'][k]/v for k,v in plan['reference_budget'].items()) for e in value['refinements']]
            ax.bar(np.arange(6)+index*.36,[max(v,.01) for v in values],width=.34,label='seed'+name.split('_')[1])
            for i,v in enumerate(values):
                if v==0:ax.text(i+index*.36,.014,'0',ha='center',fontsize=9)
        ax.axhline(1,color='black',ls='--',label='qualification gate');ax.set_yscale('log');ax.set_xticks(np.arange(6)+.18,labels,rotation=20,ha='right')
        ax.set(ylabel='Normalized adjacent edge error',title='Packed boxes: Float64 + exact partition + analytic walls still fail');ax.legend()
        fig.tight_layout();fig.savefig(output/'packed-refinement.png',dpi=170);pdf.savefig(fig);plt.close(fig)
        fig,ax=plt.subplots(figsize=(10,4.5))
        labels=[n.split('_')[1] for n in qualified]
        for index,(suffix,label) in enumerate([('candidate_p128_s32','Continuous fine128×32'),('adaptive','Controller'),(None,'Cheapest passing fixed')]):
            values=[]
            for name,selection in qualified.items():values.append(1000*records[name+'__'+(suffix or selection['choice'])]['median_s'])
            ax.bar(np.arange(len(labels))+index*.25,values,width=.23,label=label)
        ax.set_xticks(np.arange(len(labels))+.25,labels);ax.set(xlabel='Qualified concave seed',ylabel='Median native engine + controller time [ms]',title='Controller saves idle work; fixed settings remain cheaper');ax.legend()
        fig.tight_layout();fig.savefig(output/'controller-cost.png',dpi=170);pdf.savefig(fig);plt.close(fig)


if __name__=='__main__':report()
