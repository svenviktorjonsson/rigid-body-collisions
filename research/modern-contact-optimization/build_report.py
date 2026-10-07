"""Build a separate numerical report; do not change the symbolic article."""
import hashlib
import json
from pathlib import Path
import subprocess

HERE=Path(__file__).resolve().parent


def escape(value):
    return str(value).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def main():
    runs=[json.loads((HERE/'run-v5/results.json').read_text())]
    rows=[row for run in runs for row in run['benchmarks']]
    assert len(rows)==51
    million=[row for row in json.loads((HERE/'schedule-v1/results.json').read_text())['cases'] if row['contacts']==1000000]
    verified=runs[0]['verification']
    assert hashlib.sha256((HERE.parents[1]/'contact_history.py').read_bytes()).hexdigest()==runs[0]['hashes']['candidate']
    selected=[row for row in rows if row['count']==10000]
    names={'stick':'Static/sticking','coherent_slide':'Coherent lower','reversing_slide':'Reversing',
           'coherent_upper':'Coherent upper','coherent_mixed':'Mixed directions','coupled_face':'Coupled face'}
    lines=[]
    for row in selected:
        ms=row['median_ms']
        lines.append(f"{row['modes']} & {names[row['scenario']]} & {ms['frozen']:.1f} & {ms['candidate_cold']:.1f} & {ms['candidate_hint']:.1f} & {row['speedup_cold']:.2f}/{row['speedup_hint']:.2f} \\\\")
    native=[]
    for row in million:
        native.append(f"{escape(row['kind'])} & {row['islands']:,} & {row['colors']} & {row['serial_contacts']:,} & {row['build_median_ms']:.2f} & {row['cache_hit_median_ms']:.2f} \\\\")
    prep=[r['preparation_ms'] for r in rows]
    gains=[r['speedup_cold'] for r in rows];hints=[r['speedup_hint'] for r in rows]
    folder=HERE/'report';folder.mkdir(exist_ok=True)
    source=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=22mm]{geometry}
\usepackage{hyperref}
\newcommand{\toprule}{\hline}\newcommand{\midrule}{\hline}\newcommand{\bottomrule}{\hline}
\renewcommand{\arraystretch}{1.1}
\hypersetup{colorlinks=true,urlcolor=blue}
\setlength{\parindent}{0pt}\setlength{\parskip}{6pt}
\begin{document}
\begin{center}\Large Efficient numerical methods for the contact model\\
\normalsize Local memory optimization and reusable contact topology\\7 October 2026\end{center}
\textbf{Scope.} This work preserves the declared fixed-mode spring/slider law
and adds dimension-independent scheduling for complete force-and-angular-impulse
contact blocks. It does not finish the general collision model, validate new
material parameters, or establish superiority over another physics engine.

\section*{What was implemented}
The contact-history solver prepares its small Cholesky factors and face metadata
once. It uses the original full unconstrained arithmetic for every static-capacity
decision. On yielding it can try a caller-owned
previous face, then a face predicted by the current unconstrained solution.
Every trial must satisfy the current box constraints and KKT conditions. If the
guesses fail, the complete original face search remains available. Zero capacity
is a singleton interval. Opening retains its separate released-mode-energy
ledger; finite/passivity checks remain mandatory. No inverse is formed and no
friction, restitution, stiffness or mass value is fitted.

Previous-face hints are not persistent physical state. Contact identity and the
existing mode history remain the caller's responsibility. The prepared mobility,
stiffness and timestep must remain fixed; changing them requires a new prepared
object. The component still covers only one to three declared modes.

The C++17 indexed scheduler connects contacts through mutable bodies, splits
islands, and builds conflict-free colors. A complete contact includes both its
force impulse and independent angular impulse; the body angular update also
includes the force lever moment. A shared read-only support does not merge
independent bodies. Color search is bounded; excess conflicts run serially.
Cache reuse checks every ordered endpoint, mutable-body flag and color budget.
An invalid request cannot corrupt the previous plan.

The optional OpenMP visitor has barriers between colors. It currently commits
prescribed or already gated blocks; it is not a qualified colored nonlinear
world solver. Other joints and shared mutable constitutive states must be
included before islands can be treated as independent solves.

\section*{Research basis and next step}
Warm starts, substeps and constraint variants are described in
\href{https://box2d.org/posts/2024/02/solver2d/}{Catto's Solver2D}; persistent
connectivity in \href{https://box2d.org/posts/2023/10/simulation-islands/}{Simulation
Islands}; and graph coloring/data layout in
\href{https://box2d.org/posts/2024/08/releasing-box2d-3.0/}{Box2D 3.0}.
\href{https://jrouwe.github.io/JoltPhysicsDocs/5.5.0/index.html}{Jolt's architecture}
is a reference for multicore island processing.
\href{https://mujoco.readthedocs.io/en/latest/computation.html}{MuJoCo's computation
reference} includes coupled angular friction, but its material law is not a
replacement for the requested motion-directed law.

Next priority is a body-local sparse/matrix-free application of the article's
full shared-point wrench mobility. Accumulate signed force and couple momentum,
apply inverse mass/full world inertia, gather relative motions and project into
the allowed component directions. This preserves inter-contact coupling without
a dense contact matrix. Qualify it against dense 2D/3D oracles before adoption.
\newpage
\section*{Paired local-history timings}
Seven alternating repetitions after warmup; same inputs and consumed outputs.
The table reports medians for 10,000 repeated Python local responses, not an
evolving world trajectory. Motion/history evolution is tested separately. Validation, triangular
solves, current friction checks and energy gates are included. World discovery,
integration, group solving and preparation are excluded. These timings do not
replace the earlier native supported-response benchmark. Host: AMD Ryzen 7
3700X, Linux x86-64, Python 3.14.4, NumPy 2.5.3.

``Current'' is the default no-hint solver; ``Warm'' also supplies the previous
accepted face. The frozen implementation is retained byte-for-byte. The coherent
lower case deliberately favors the original enumeration order; upper and mixed
cases expose its ordering cost. Reversing cases invalidate old hints. Coupled-face
cases test a wrong unconstrained guess and a valid previous face.

\begin{center}\small
\begin{tabular}{rlrrrr}\toprule
Modes & Case & Frozen ms & Current ms & Warm ms & Gain current/warm\\\midrule
'''+'\n'.join(lines)+r'''
\bottomrule\end{tabular}\end{center}
All 51 size/scenario combinations are retained in the machine-readable results
(100, 1,000 and 10,000 updates). Across them the default gain is '''+f'{min(gains):.2f}--{max(gains):.2f}'+r''' times;
with hints it is '''+f'{min(hints):.2f}--{max(hints):.2f}'+r''' times. These are measured local numerical gains,
not an all-scene 2x gate or a competitor ranking. A hint can cost time when it
fails or when the current prediction already succeeds.

Preparation is deliberately reported separately: frozen objects took
'''+f"{min(p['frozen'] for p in prep):.3f}--{max(p['frozen'] for p in prep):.3f}"+r''' ms;
prepared candidates took '''+f"{min(p['candidate'] for p in prep):.3f}--{max(p['candidate'] for p in prep):.3f}"+r''' ms.
Candidate preparation includes all zero-capacity patterns and their small factor
metadata. Frequent mobility changes can reduce or erase its amortized benefit.

The preliminary v2/v3/v4 runs and candidates are preserved but not accepted
as final performance. An exact-static-limit audit found 9 branch changes from
roundoff in the fast unbounded solve; differing static/dynamic friction could
then cause large motion changes. The final solver retains the original unbounded
trial for all static decisions, without changing any material threshold. The initial SLSQP
oracle stopped short on one scaled quadratic; its failure log is retained. The
final independent oracle uses an equivalent bounded least-squares transform and
SciPy BVLS, eliminating zero-capacity coordinates explicitly.
\newpage
\section*{Topology costs and correctness}
One million contacts per graph; seven alternating fresh builds/cache hits.
Single-worker C++17 -O3, without fast-math. Fresh-build timing includes planner
allocation/construction, stops before plan destruction, and excludes input graph
creation. Reuse includes full equality checking and lookup; no allocation on a
hit. These costs exclude collision discovery and all physical solving.

\begin{center}\small
\begin{tabular}{lrrrrr}\toprule
Graph & Islands & Colors & Serial tail & Build ms & Reuse ms\\\midrule
'''+'\n'.join(native)+r'''
\bottomrule\end{tabular}\end{center}
The million-contact hub illustrates the ownership limit: nearly every contact
shares one dynamic body, so coloring cannot manufacture parallel independence.
The heterogeneous-degree case is graph topology, not simulated irregular-shape
geometry. Shapes and materials do not enter this planner.

\textbf{Verification.} All 63 focused repository tests pass, including the
existing supported, restitution, measured-inertia and conservation checks.
'''+str(verified['cases'])+r''' seeded local-history cases ('''+str(verified['sliding_cases'])+r''' yielded) agree
with the frozen physical outputs to '''+f"{verified['max_scaled_frozen_difference']:.3g}"+r''' maximum scaled difference.
'''+str(verified['independent_optimizer_cases'])+r''' cases were independently checked by BVLS;
the maximum impulse difference was '''+f"{verified['independent_optimizer_max_impulse_error']:.3g}"+r'''.
Sequential motion/history, opening/recontact, zero capacities, singular body
mobility, wrong face hints and coupled fallback have explicit controls.
An additional 300 exact/adjacent-static-limit controls preserve every original
static/dynamic branch; static impulses are bit-identical to the original.

The native planner passes 300 randomized graph comparisons with an independent
BFS connectivity reference and write-conflict checks. Prescribed 2D/3D force
plus independent-couple applications verify linear and angular momentum,
including prescribed-support reactions. Eight-worker and serial colored outputs
are bit-identical. Both OpenMP and portable serial builds pass. These are
mechanics identities, not restitution/friction predictions of measured impacts.

\textbf{Remaining limits.} General moving full t/s history, zero-motion directional
closure, joint normal impact, a physical coupled finite-patch budget, and dense
irregular/group trajectory accuracy remain open. Coloring can change nonlinear
iteration convergence and must be qualified on complete scenes. Existing public
material parameters and experimental errors are unchanged; no empirical
accuracy gain is claimed. The main compiler and native world defaults are
unchanged, as is the supported-response C ABI v1. This planner is a C++ header
interface, not a completed compiler/GPU port.

\textbf{Reproducibility.} Final code/harness hashes are in run-v5/results.json;
earlier checkpoints e66c4d5 and f7e5b8f retain preliminary versions. Frozen source, earlier outcomes, final
sample timings, source hashes, independent audits and tests are retained in
\href{https://github.com/svenviktorjonsson/rigid-body-collisions/tree/research/adaptive-benchmark-validation/research/modern-contact-optimization}{research/modern-contact-optimization}.
The separate symbolic article is preserved.
\end{document}
'''
    (folder/'report.tex').write_text(source)
    for k in (1,2):
        command=['pdflatex','-interaction=nonstopmode','-halt-on-error','report.tex']
        result=subprocess.run(command,cwd=folder,capture_output=True,text=True)
        (folder/f'build-{k}.txt').write_text(result.stdout+result.stderr);result.check_returncode()
    for suffix in ('aux','log','out'):(folder/f'report.{suffix}').unlink(missing_ok=True)


if __name__=='__main__':main()
