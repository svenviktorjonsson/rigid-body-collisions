"""Build the measured sparse-kernel report and plots from retained records."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def report():
    root = Path(__file__).parent/'sparse-islands'
    data = json.loads((root/'results/summary.json').read_text())
    normal = {(r['count'],r['method']):r for r in data['records'] if r['kind']=='normal'}
    friction = [r for r in data['records'] if r['kind']=='friction']
    stress = [r for r in data['records'] if r['kind']=='stress']
    a,b=normal[256,'dense_optimizer'],normal[256,'sparse_active']
    c,d=normal[1024,'dense_active'],normal[1024,'sparse_active']
    lines=['# Sparse coupled-contact performance', '',
        'Executed 4 October 2026. This is a measured improvement to a frozen-contact kernel. '
        'It does not establish full-engine speed, collision-discovery performance, physical material '
        'accuracy, Vektor compiler execution, WASM or GPU performance.', '',
        f"At 256 balls the new sparse pipeline is **{a['median_total_s']/b['median_total_s']:.1f} times faster** "
        f"than the existing dense verification optimizer: {1000*a['median_total_s']:.1f} ms versus "
        f"{1000*b['median_total_s']:.2f} ms for contact construction, matrix preparation and solve. "
        f"At 1,024 balls, sparse solving is **{c['median_solve_s']/d['median_solve_s']:.1f} times faster** "
        f"than the dense version of the same active-set algorithm; the cold pipeline gain is "
        f"{c['median_total_s']/d['median_total_s']:.1f} times. This separates a better algorithm from sparse-factorization gains.", '',
        'All normal cases meet a maximum individual velocity error of 1e-8 m/s and the declared '
        'reaction impulse tolerance. The packed rigid row must propagate box velocity to every '
        'ball. Actuator work and kinetic/dissipated energy are independently audited. '
        'The sparse implementation reaches 100,000 balls; the dense controls are only executed '
        'within their declared size limits. No extrapolated dense runtime is presented.', '',
        '## Protocol and numerical model', '',
        'The plan was pushed before execution. Seven interleaved repeats follow one warm-up per '
        'setting. BLAS/OMP threads are one. Each solve performs a fresh factorization; there is no '
        'reuse of the answer. Cold totals include Python contact construction, mass/contact-map '
        'assembly and mobility preparation. Process/library startup, archive writing and audit are '
        'excluded. Timings are one-machine evidence, not a general hardware guarantee.', '',
        'The old baseline is the existing dense L-BFGS-B normal optimizer with stationarity '
        'correction. The same-algorithm dense control uses the new active set with dense factorization. '
        'The sparse version stores inverse mass as a vector, the contact Jacobian as CSR and '
        'factors only the free contact block. Bound steps preserve feasibility. Singular free '
        'blocks use least squares without adding contact softness. Failed KKT/velocity checks reject '
        'the result. Block active sets can still fail on some graphs; no universal convergence claim.', '',
        'Automatic normal dispatch uses dense factorization for at most 128 contacts and sparse '
        'otherwise, with the same physical law and tolerances. Friction dispatch uses the smaller '
        'bounded tangential solve only if the normal/tangent cross-block is exactly zero. '
        'Other contacts use a sparse semismooth solve with merit line search and an explicitly '
        'reported least-squares fallback. Small nonzero coupling is never discarded.', '',
        '## Normal results', '',
        '| Balls | Method | Cold median (ms) | Solve median (ms) | Operator arrays (KiB) |',
        '|---:|---|---:|---:|---:|']
    for (n,method),r in normal.items():
        lines.append(f"| {n:,} | {method} | {1000*r['median_total_s']:.4g} | {1000*r['median_solve_s']:.4g} | {r['explicit_operator_bytes']/1024:.4g} |")
    lines+=['', 'Operator arrays count explicit inverse-mass, contact-map and mobility storage. '
        'They exclude input objects, preparation scratch, LU factors, Python/library memory and '
        'process overhead; this is not a peak-RAM measurement. Sparse factor fill can be large '
        'on more connected graphs. Linear storage/work trends shown here belong to the row topology.', '',
        '![Normal solver scaling](results/normal-scaling.png)', '',
        '## Coulomb friction results', '',
        'Rows start with lateral velocity 2 sin(0.37 times body index), wall velocity (1,0.2) m/s '
        'and synthetic friction 0.02 or 0.4. The first includes sliding contacts, the second sticks '
        'in these snapshots. The implicit inelastic law enforces normal complementarity, static '
        'capacity and saturated friction opposing final slip. It has one friction coefficient, '
        'zero restitution, no stored tangential elasticity and no rolling couple. It is not the '
        'separate elastic static/dynamic/history material model.', '',
        '| Balls | Friction | Strategy | Cold median (ms) | Solve median (ms) | Sliding contacts |',
        '|---:|---:|---|---:|---:|---:|']
    for r in friction:
        lines.append(f"| {r['count']:,} | {r['friction']} | {r['strategy']} | {1000*r['median_total_s']:.4g} | {1000*r['median_solve_s']:.4g} | {r['stats']['sliding_contacts']} |")
    lines+=['', 'The fast path is not uniformly faster. At 1,024 balls with friction 0.02 it is slower '
        'than the general solve; at 10,000 balls with friction 0.4 it reduces solve time but the measured '
        'cold pipeline is slightly slower. Contact construction dominates some large cases. '
        'All sixteen outputs pass normal/cone/slip/energy gates and match their counterpart strategy '
        'within 1e-8. No physical parameter is changed to obtain the speed gain.', '',
        'Rigid redundant pressure impulses can be nonunique. The normal active-set initialization '
        'chooses a pressure distribution; friction can depend on that choice. Verified residuals '
        'do not prove a unique or experimentally authenticated force history. Finite compliance '
        'or another declared pressure-selection policy is needed where that distinction matters.', '',
        '## Irregular held-out stress set', '',
        f"**{sum(r['passed'] for r in stress)}/100 accepted; four rejected.** The seed 20261005 set "
        'was declared before the run and differs from the exploratory development seed 81. '
        'It contains random off-centre algebraic contact graphs, not measured or guaranteed valid '
        'shape geometries. All accepted outputs pass an independent dense point-map audit of '
        'normal complementarity, friction and body-energy change. One accepted case uses the '
        'disclosed fallback. Failed input snapshots are retained with the numerical errors.', '',
        'Rejected case IDs: '+', '.join(str(r['index']) for r in stress if not r['passed'])+'. '
        'These failures establish an implementation/model limitation. The experiment does not '
        'prove whether each case has no Coulomb solution. Do not deploy this prototype as the sole '
        'general collision solver or silently count fallback/rejection as success.', '',
        '## What this changes', '',
        'The many-body normal correctness oracle no longer needs dense all-body/contact matrices '
        'or the iterative optimizer used by the earlier reference for the tested rows. A frozen island can '
        'select a faster algebraic solver without changing friction or restitution. The gain is '
        'an implementation result using established mechanics and numerical methods, not new '
        'collision theory. The earlier dense moving-container trajectories remain unqualified; '
        'this kernel benchmark does not repair or replace that evidence.', '',
        'Next: port sparse assembly and factorization into the actual engine, profile contact '
        'discovery/input preparation, preserve contact/history state, and test complete moving '
        'containers against independently qualified observables. General rejected cases need '
        'a robust recovery policy with the same mechanical contract. Automatic numerical placement '
        'and native/WASM/GPU acceptance in Vektor remain separate compiler work.', '',
        '## Reproduction and prior art', '',
        'Run the commands in README.md. The independent audit checks 139 snapshots: 23 normal '
        'outputs, 16 friction outputs and 100 stress inputs/outcomes.69 local tests pass. '
        'The versioned source archive, state archive, all timing samples and hashes are retained.', '',
        f"Execution source: `{data['source_commit']}`. State archive SHA-256: `{data['states_zip_sha256']}`.", '',
        'Relevant prior art: Alart and Curnier (1991), '
        '[A mixed formulation for frictional contact problems prone to Newton like solution methods]'
        '(https://doi.org/10.1016/0045-7825(91)90022-x); Anitescu and Potra (1997), '
        '[Formulating Dynamic Multi-Rigid-Body Contact Problems with Friction as Solvable Linear Complementarity Problems]'
        '(https://doi.org/10.1023/a:1008292328909). Titles, authors, years and DOIs were checked '
        'through Crossref metadata; that check is not a full-paper equation comparison. '
        'See also [SciPy sparse linear algebra](https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html), '
        '[Catto, Solver2D](https://box2d.org/posts/2024/02/solver2d/) and the existing joint publication review.', '']
    fig,ax=plt.subplots(figsize=(8,4))
    for method in ('dense_optimizer','dense_active','sparse_active','auto_active'):
        rows=[r for (_,m),r in normal.items() if m==method]
        ax.loglog([r['count'] for r in rows],[r['median_solve_s']*1000 for r in rows],'o-',label=method)
    ax.set(xlabel='Balls in frozen driven row',ylabel='Median solve time [ms]',title='Equal normal velocity budget: 1e-8 m/s; seven interleaved repeats')
    ax.grid(alpha=.25,which='both');ax.legend(fontsize=8);fig.tight_layout();fig.savefig(root/'results/normal-scaling.png',dpi=160)
    (root/'report.md').write_text('\n'.join(lines))


if __name__=='__main__': report()
