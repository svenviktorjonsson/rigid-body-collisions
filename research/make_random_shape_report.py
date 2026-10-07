"""Render actual random polygon geometry, refinement outcomes and measured timings."""
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from research.run_rigid_study import normalized_error

ROOT = Path(__file__).parent/'random-shapes'


def report():
    output = ROOT/'results'; data = json.loads((output/'summary.json').read_text())
    scenes = json.loads((output/'scenes.json').read_text()); plan = json.loads((ROOT/'plan.json').read_text())
    q = data['qualifications']; accepted = sum(k['accepted'] for k in data['kernels'])
    rejected = [r for r in data['records'] if not r['accepted']]
    passed = [c for c in data['comparisons'] if c['passed']]
    lines = ['# Seeded random-shape verification', '', 'Executed 5 October 2026. Synthetic SI geometry and '
        'coefficients; numerical verification does not authenticate a physical material.', '',
        f"Two seeds generate eight native scenes: irregular convex drops, concave drops, off-centre pair "
        f"collisions and shaking containers with 36 mixed bodies (12 concave). **{sum(v['qualified'] for v in q.values())}/8 "
        f"references qualify** on all four predeclared refinement edges; **{len(passed)}/40 candidate "
        f"settings pass against a qualified reference**. **{len(rejected)} native attempts are rejected**.", '',
        f"The independent sparse kernel accepts **{accepted}/24 solves** on twelve real support-contact "
        'chains of 8, 32 and 128 polygon bodies. All accepted states satisfy independently checked normal '
        'complementarity, impulse application and energy minus prescribed-wall work; friction additionally '
        'satisfies capacity and opposition to final slip. These snapshots have nonzero normal/tangent '
        'coupling. They are constructed from actual polygon vertices, rather than arbitrary contact matrices.', '',
        '![Actual generated shapes](results/shapes.png)', '',
        '## Geometry and physical scope', '',
        'Convex outlines are hulls of seeded random points. Concave star outlines use a fan of triangles '
        'with disjoint interiors and shared edges, joined as one rigid body. Areal density sets each '
        'dynamic body to 1 kg. Exact polygon integrals locate its COM and moment of inertia. Bodies '
        'are centred on their COM and bounded by a declared radius; every initial mixed-box cell is '
        'clear of other bodies and walls including the 0.01 m fixture skin. Fixture skins may overlap '
        'at concave decomposition seams, and internal fixture features are not suppressed. This is '
        'a known limitation of this representation. No deformable rods or FEM continuum is simulated.', '',
        'Native runs use pinned Box2D 2.4.1 and 3.1.1 adapters. The collision skin is fixed across '
        'fidelity settings, friction is 0.4, restitution zero, rolling zero, and gravity 9.81 m/s² '
        'except in the isolated pair. No material coefficient is tuned to make a fast run match. '
        'The frozen kernel uses zero-skin exact core support contacts, friction 0.2 and zero '
        'restitution. It does not integrate these trajectories or perform native collision discovery.', '',
        '## Reference qualification and candidate accuracy', '',
        'Primary collision updates refine 4 → 8 → 16 at 64 velocity iterations; velocity iterations '
        'refine 16 → 32 → 64 at 16 primary updates. All four adjacent edges must have RMS position '
        '≤0.005 m, velocity ≤0.0125 m/s and spin ≤0.0125 rad/s. Candidates use budgets four times '
        'larger. A failed reference prevents an accuracy claim even when a candidate is close to it. '
        'This is a bounded trajectory study, not a proof for all shapes or long time horizons.', '',
        '| Scene | Reference qualified | Worst normalized refinement error | Passing candidate settings |',
        '|---|---|---:|---|']
    for scene in scenes:
        name = scene['id']; worst = max(normalized_error(e['errors'], plan['reference_budget']) for e in q[name]['edges'])
        labels = [c['trace'].split('__')[1] for c in passed if c['scene'] == name]
        lines.append(f"| {name} | {q[name]['qualified']} | {worst:.3g} | {', '.join(labels) or 'none'} |")
    lines += ['', 'For the four qualified scenes, the cheapest passing candidate can be selected '
        'retrospectively from the declared settings. This is scene-specific selection after verification, '
        'not an online controller or a held-out prediction of the cheapest setting.', '',
        '| Qualified scene | Cheapest passing setting | Median native time (ms) | Reference time (ms) | Reference/candidate ratio |',
        '|---|---|---:|---:|---:|']
    records = {r['trace']: r for r in data['records']}
    for name in q:
        valid = [records[c['trace']] for c in passed if c['scene'] == name]
        if not valid: continue
        best = min(valid, key=lambda r: r['median_s'])
        ref = records[name+'__block_p16_s64']
        lines.append(f"| {name} | {best['trace'].split('__')[1]} | {1000*best['median_s']:.3f} | "
                     f"{1000*ref['median_s']:.3f} | {ref['median_s']/best['median_s']:.2f} |")
    lines += ['', 'Timing: one warm-up plus three repetitions, single-thread BLAS/OMP. Native medians '
        'measure compiled engine and controller time, excluding Python/process/serialization. Frozen '
        'kernel medians include inverse-mass/contact-map assembly and solving, excluding geometry '
        'generation, startup and archive writing. These are different scopes and cannot be divided '
        'to claim a full-engine speedup. No random-shape dense control is measured here.', '',
        '## Frozen physical contacts', '',
        'Adjacent bodies occupy disjoint x intervals and touch at their extreme vertices, with '
        'horizontal normals in both support cones. Outer contacts lie on prescribed moving wall '
        'planes. Contact points and lever arms are measured from the actual COM. A random initial '
        'translation and spin makes the solves nontrivial. Each accepted output is archived; failures '
        'retain their inputs and explicit reason. This deliberately tests a chain topology, not every '
        'possible packed polygon contact graph.', '',
        '| Bodies | Seed | Shape | Law | Accepted | Assembly + solve median (ms) |',
        '|---:|---:|---|---|---|---:|']
    for k in data['kernels']:
        _, count, seed, shape = k['snapshot'].split('_')
        timing = f"{1000*k['median_total_s']:.3f}" if k['accepted'] else 'rejected'
        lines.append(f"| {count} | {seed} | {shape} | {k['law']} | {k['accepted']} | {timing} |")
    if rejected:
        lines += ['', '## Native rejections', '']
        for r in rejected: lines.append(f"- `{r['trace']}`: {r['failure']}")
        lines += ['', 'Strict mathematical convexity and the adapter’s edge-length check do not '
            'guarantee acceptance by native hull welding/validation tolerances. A production geometry '
            'pipeline needs an explicit admissibility check and a disclosed repair policy. These '
            'rejections remain failures; their geometry was not regenerated to remove them.']
    failed = [k for k in data['kernels'] if not k['accepted']]
    if failed:
        lines += ['', '## Kernel rejections', '']
        for k in failed: lines.append(f"- `{k['snapshot']}` {k['law']}: {k['failure']}")
    lines += ['', '## Reproduction and next step', '',
        f"Plan/generator initial commit: `dd0f2e8f5f06360494ab11d2e2d7a3864087b79f`. "
        f"Execution source: `{data['source_commit']}`. The runner was changed to preserve native "
        'rejections after the first incomplete pass; geometry, seeds and accuracy gates were unchanged. '
        'The source archive records the execution bytes. `scenes.json`, `traces.zip` and '
        '`summary.json` retain geometry, full trajectories, failed inputs, timings and hashes.', '',
        'Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m research.run_random_shape_study`, '
        'then `python -m research.audit_random_shape_study` and '
        '`python -m research.make_random_shape_report`. Both native backends must be built first.', '',
        'The next integration step is a geometry pipeline that exports actual evolving manifolds into '
        'the sparse solver, with admissibility and decomposition-seam checks. Random-shape kernel '
        'acceptance does not establish native trajectory accuracy or preserve the prior disk-row '
        'speedup on arbitrary dense contact graphs. Failed reference scenes need further refinement '
        'or appropriately declared ensemble/observable metrics before selecting a fast setting.']
    (ROOT/'report.md').write_text('\n'.join(lines)+'\n')
    fig, axes = plt.subplots(2, 8, figsize=(12, 3.3))
    geometries = [g for scene in scenes if 'mixed36' in scene['id'] for g in scene['generated_geometry']]
    selected = geometries[:8]+geometries[36:44]
    for ax, geometry in zip(axes.ravel(), selected):
        ax.add_patch(Polygon(geometry['outline'], facecolor='#d88950' if geometry['concave'] else '#4685b7', alpha=.8))
        ax.plot(0, 0, 'k+', ms=7); ax.set_aspect('equal'); ax.set(xlim=(-.21, .21), ylim=(-.21, .21)); ax.axis('off')
    fig.suptitle('Actual seeded convex and concave outlines; + marks centre of mass')
    fig.tight_layout(); fig.savefig(output/'shapes.png', dpi=180)
    with PdfPages(ROOT/'report.pdf') as pdf:
        pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 5))
        names = list(q); worst = [max(normalized_error(e['errors'], plan['reference_budget']) for e in q[n]['edges']) for n in names]
        ax.barh(names, worst, color=['#408050' if q[n]['qualified'] else '#b54b45' for n in names])
        ax.axvline(1, color='black', linestyle='--'); ax.set_xscale('log'); ax.set_xlabel('Worst normalized adjacent refinement error (gate ≤ 1)')
        ax.set_title('Reference qualification precedes any trajectory accuracy claim')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        fig, ax = plt.subplots(figsize=(10, 6)); ax.axis('off')
        text = (f'Random-shape evidence — 5 October 2026\n\n'
            f'{sum(v["qualified"] for v in q.values())}/8 native references qualified; {len(passed)}/40 candidates passed.\n'
            f'{len(rejected)} native geometry rejections retained.\n'
            f'{accepted}/24 frozen polygon contact solves accepted and independently audited.\n\n'
            'Geometry: irregular convex hulls and concave triangle compounds; exact COM/inertia.\n'
            'Full simulations: drops, off-centre pairs and 36 mixed shapes in shaking boxes.\n'
            'Frozen contacts: real support vertices; normal/tangent/rotation coupling retained.\n\n'
            'Limits: synthetic coefficients; no experimentally characterized materials.\n'
            'Concave fixture skins/seams are a known modelling limitation.\n'
            'Contact-chain verification is not full-world integration or arbitrary graph validation.\n'
            'No random-shape dense-control speedup or universal engine claim.\n\n'
            'Reproduce and audit: research/run_random_shape_study.py,\n'
            'research/audit_random_shape_study.py. Full tables and rejection reasons: report.md.\n'
            'Raw geometry, trajectories, contact states, timings and source hashes: results/.')
        ax.text(.03, .96, text, va='top', fontsize=12, linespacing=1.6); pdf.savefig(fig); plt.close(fig)


if __name__ == '__main__': report()
