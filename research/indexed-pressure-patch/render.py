"""Standalone scientific figures and a concise research checkpoint report."""
import json, math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit import footprint
from model import evaluate

here=Path(__file__).resolve().parent
audit=json.loads((here/'audit-v2/audit.json').read_text())
refinement=json.loads((here/'refinement-v2/refinement.json').read_text())
b1=json.loads((here/'benchmark-v1/native.json').read_text())['batches']
b2=json.loads((here/'benchmark-v2/native.json').read_text())['batches']
origin=json.loads((here/'reference-point-v1/audit.json').read_text())
colors=dict(hertz='#2360a8',ellipse='#d47b13',irregular='#12856e',interval='#7943a2')
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})

fig,ax=plt.subplots(figsize=(7.2,3.9))
for kind in ['hertz','ellipse','irregular']:
    rows=[x for x in refinement['rows'] if x['kind']==kind]
    x=np.array([r['velocity_spin_ratio'] for r in rows])
    y=np.array([r['fixed_256_site_error'] for r in rows])
    refined=np.array([r['measured_error_against_147456_site_reference'] for r in rows])
    ax.scatter(x,y,color=colors[kind],marker='o',facecolors='none',label=kind+' / 256 sites')
    accepted=np.array([r['accepted_estimate'] for r in rows])
    ax.scatter(x[accepted],refined[accepted],color=colors[kind],marker='o')
    ax.scatter(x[~accepted],refined[~accepted],color=colors[kind],marker='x')
    for a,b,c in zip(x,y,refined):ax.plot([a,a],[b,c],color=colors[kind],alpha=.22)
ax.axhline(1e-4,color='black',linestyle='--',linewidth=1,label='probe target 1e-4')
ax.set(xscale='log',yscale='log',xlabel='Center sliding speed / (patch radius × axial spin)',ylabel='Scaled force / couple error vs fine integration',ylim=(1e-15,.01))
ax.legend(fontsize=8,ncol=2);ax.grid(alpha=.15);fig.tight_layout()
fig.savefig(here/'refinement-error.pdf');fig.savefig(here/'refinement-error.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(7.2,3.9))
for kind in ['interval','hertz','ellipse','irregular']:
    for mode,marker in [(0,'o'),(1,'s'),(2,'^')]:
        rows=[x for x in b2 if x['shape']==kind and x['mode']==mode]
        if not rows:continue
        ax.scatter([x['count']*(1+list(colors).index(kind)*.035) for x in rows],[x['speedup'] for x in rows],color=colors[kind],marker=marker,label=kind+[' loaded',' twist',' opening'][mode])
ax.axhline(1,color='black',linewidth=1);ax.set(xscale='log',yscale='log',xlabel='Number of indexed contact responses',ylabel='Direct / compact kernel time',ylim=(1,40))
ax.legend(fontsize=7,ncol=3);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(here/'kernel-gains.pdf');plt.close(fig)

# Signed size error and angular error on one diagram, using the same input states.
fig,ax=plt.subplots(figsize=(7.2,3.9));deviation=[]
for kind in ['hertz','ellipse','irregular']:
    for ratio in [.1,.3,.75,1.,2.,8.]:
        args=([.002,.015,-.02],[ratio*.012*10,0,-.01],[.1,.2,10],10000,10,.4)
        ref=evaluate(footprint(kind,192,768),*args)
        for n,marker in [(8,'o'),(32,'x')]:
            result=evaluate(footprint(kind,n,4*n),*args)
            size=100*(np.linalg.norm(result['force'])/np.linalg.norm(ref['force'])-1)
            # atan2 avoids precision loss for nearly parallel moment directions.
            angle=math.degrees(math.atan2(np.linalg.norm(np.cross(result['moment'],ref['moment'])),result['moment']@ref['moment']))
            ax.scatter(size,angle,color=colors[kind],marker=marker,s=30)
            deviation.append(dict(kind=kind,ratio=ratio,sites=4*n*n,force_size_error_percent=size,moment_angle_error_degrees=angle))
    ax.scatter([],[],color=colors[kind],label=kind)
ax.axvline(0,color='black',linewidth=.8);ax.axhline(0,color='black',linewidth=.8)
ax.set(xlabel='Signed resultant-force magnitude error (%)',ylabel='Resultant-couple direction error (degrees)')
ax.legend(fontsize=8);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(here/'deviation-map.pdf');fig.savefig(here/'deviation-map.png',dpi=160);plt.close(fig)
(here/'deviation-map.json').write_text(json.dumps(deviation,indent=2)+'\n')

bench_rows=[]
for row in b1:
    if row['count']!=1000000:continue
    confirmation=[r['speedup'] for r in b2 if r['shape']==row['shape'] and r['mode']==row['mode']]
    bench_rows.append(f"{row['shape']} / {['loaded','mixed twist','opening'][row['mode']]} & {row['sites']} & {row['reference_seconds']*1000:.2f} & {row['compact_seconds']*1000:.2f} & {row['speedup']:.2f} & {min(confirmation):.2f}--{max(confirmation):.2f}" + r" \\")
directions=[]
for row in audit['directional_examples']:
    directions.append(f"{row['kind']} & {row['linear_residual']:.6f} & {row['angular_residual']:.6f}" + r" \\")
accepted=sum(r['accepted_estimate'] for r in refinement['rows'])
worst=max(r['measured_error_against_147456_site_reference'] for r in refinement['rows'])
text=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,hyperref,array}
\newcommand{\toprule}{\hline}\newcommand{\midrule}{\hline}\newcommand{\bottomrule}{\hline}
\hypersetup{colorlinks=true,urlcolor=blue}
\setlength{\parindent}{0pt}\setlength{\parskip}{5pt}\setlength{\emergencystretch}{2em}
\newcommand{\rw}[1]{\mathord{#1\mkern2mu\wedge}}
\title{Indexed pressure patches for efficient rigid-body contact}
\author{Viktor Jonsson's collision-model research\quad Research checkpoint}
\date{7 October 2026}
\begin{document}\maketitle
\textbf{Outcome.} A small pressure-patch candidate generates both force and
independent torque from the same local deformation/friction calculation. An exact
moment reduction speeds every tested synthetic indexed CPU batch. These are
mechanical and numerical results: no new improvement against experimental
measurements is claimed, and production collision examples remain unqualified.

\textbf{What changed.} Previously the new finite-patch branch handled pure axial
spin only. The candidate now admits simultaneous sliding, axial spin, transverse
rotation, asymmetric pressure, irregular footprints and local normal unloading.
Bodies stay rigid; contact compression has three affine coordinates and the
footprint has a small site/template representation. This is a frozen planar
foundation reference, not full hydroelastic geometry or a deformation mesh.

\section*{Mechanics with the established notation}
Keep angular momentum $\mathbf L$, numerical reference length $\ell$,
inertia $\mathbb I$, $\mathbf V=[\mathbf v;\ell\boldsymbol\omega]$ and
$\mathbf P=[\mathbf p;\mathbf L/\ell]$. Contact impulses remain
$\delta\mathbf p,\delta\mathbf L$; combined changes use uppercase $\Delta\mathbf P$.
Each body retains only index $k$. In the body update, the contact-origin couple is
additional to the force's center-to-contact moment:
\[
 \Delta\mathbf L_k=\rw{\mathbf r_k}\,\delta\mathbf p+\delta\mathbf L.
\]
Here the displayed contact acts with positive incidence sign; signed incidence
supplies the opposite body's contribution. The article's transpose-only
wedge reversal remains unchanged.

For one pressure site, $\mathbf r=[x,y,0]^T$,
$\mathbf b=[1,y,-x]^T$, $\mathbf d$ contains virtual compression/tilts and
$\mathbf q=[v_z,\omega_x,\omega_y]^T$. Let $d=\mathbf b^T\mathbf d$,
$u=\mathbf b^T\mathbf q$. Then $\dot d=-u$. With total stiffness $K$, damping $C$
and positive site weight $w$, the unilateral normal contribution is
\[
 f=w\max(0,K\max(d,0)-Cu)\quad(d>0),\qquad f=0\quad(d\leq0).
\]
Local tangential slip is $[v_x-\omega_z y,\ v_y+\omega_z x]^T$;
its kinetic traction opposes this slip with magnitude $\mu_d f$.
At zero local slip this kinetic branch is zero; static friction remains an
unresolved additional branch. Summing $\mathbf f_j$ and
$\rw{\mathbf r_j}\,\mathbf f_j$ over site $j$ produces resultant force and
the independent contact couple. The same local friction budget supplies both.

\newpage
\section*{Stored energy, release and an exact small matrix}
Stored normal energy is $U=\tfrac12K\sum_j w_j\max(d_j,0)^2$.
Normal dissipative power is $D_n=\sum_j(w_j K\max(d_j,0)-f_j)u_j\geq0$;
kinetic sliding loss is $D_t=\mu_d\sum_j f_j |\mathbf u_j^{\rm slip}|\geq0$.
The instantaneous work identity is
\[
 \mathbf F\cdot\mathbf v+\mathbf T\cdot\boldsymbol\omega+\dot U+D_n+D_t=0.
\]
Unloading/clipping keeps stored energy and its dissipation in the ledger instead
of deleting a spring state. This isolated frozen transient does not implement
state transport into a later moving footprint/contact.

Cache the small Gram matrix
\[
 \mathbb G=\sum_j w_j\mathbf b_j\mathbf b_j^T.
\]
If all sites are compressed and trial pressures positive, the normal
force/rolling-moment triple is exactly
$\mathbb G(K\mathbf d-C\mathbf q)$.
If axial twist is zero, local tangential velocity is uniform and its force/moment
also reduce using the same pressure moments. A conservative footprint-box test
admits this branch. Other states retain site-level evaluation, including opening.
This is an algebraic reduction, with no altered material constants or tolerances.

For circular Hertz weights, uniform compression and zero center slip,
$T_x=-Ca^2\omega_x/5$ produces normal-deformation rolling resistance.
Pure axial sliding gives $T_z=-3\pi\mu_dNa/16$ times spin sign, using the existing
dynamic coefficient and physical radius $a$. This is not a new fitted coefficient,
a universal material law or a new mechanics identity. Mixed states cannot simply
apply both their separate maximum sliding force and maximum axial moment.

\begin{center}\begin{tabular}{p{.2\linewidth}p{.69\linewidth}}\toprule
Input & Treatment \\\midrule
$e_n,e_t$ & Existing restitution inputs kept separate; not double-imposed on top of compliant dynamics. Mapping to measured compliant response still required.\\
$\mu_s,\mu_d$ & Fixed documented values remain fixed; this candidate uses only the dynamic branch.\\
$\mu_r$ & Retain documented rolling resistance with its physical length scale; do not add it again to pressure-induced losses.\\
$K,C,a$ & Independent effective contact characterization/geometry. Current controls use explicit synthetic values, not rubber or rock measurements.\\
$\ell$ & Coordinate length, never a material/footprint radius.\\\bottomrule
\end{tabular}\end{center}

\newpage
\section*{Indexed computation and numerical verification}
Use separate body-component arrays, flat contact fields and shared immutable
footprint templates. A contact owns a range of signed incidences; each incidence
has one owner body $k$, sign and lever arm. Site $j$ owns one pressure sample.
Gather contact motion, compute a local wrench, then scatter force and lever moment
plus independent couple. No dense body-by-contact tensor or global inverse is
needed for this local operation. Full rapid groups still require a coupled solve.

Vektor Flow's Section 0 requires explicit reductions: for a contact-local view,
\texttt{f\_x: sum\_j(f\_x\_j)}. Repeated dimension names are not implicit sums.
Keep one index per entity and ordered reductions where floating-point order is
part of the contract. Current Python/C++ does not certify a VKF compiler port,
ragged GPU gather/scatter, collision integrator or evolving-history semantics.

240 controls cover planar/spatial mechanics, 50 loaded and 190 partly open or
clipped states, arbitrary world frames and body incidences. Maximum scaled
instantaneous energy-rate residual is \texttt{@@POWER@@}; indexed work/global
momentum residual is \texttt{@@INCIDENCE@@}. Analytic Hertz spin moment and cached
normal Gram calculations agree near roundoff. The isolated normal transient
tracks kinetic, stored and lost energy with maximum absolute residual
\texttt{@@ODE@@} J against initial 0.5 J.

\textbf{Directional compatibility remains a real issue.}
The user's directions are full relative velocity $\hat{\mathbf t}$ and full
relative angular velocity $\hat{\mathbf s}$, with
\[
 \delta\mathbf p=\delta p_n\hat{\mathbf n}+\delta p_t\hat{\mathbf t},\qquad
 \delta\mathbf L=\delta L_s\hat{\mathbf s}+\delta L_n\hat{\mathbf n}.
\]
We do not redefine these directions. In one synthetic mixed state, orthogonal
residuals outside those allowed spans are:
\begin{center}\begin{tabular}{lrr}\toprule
Footprint & Force residual (N) & Couple residual (N m)\\\midrule
@@DIRECTIONS@@
\bottomrule\end{tabular}\end{center}
These residuals depend on the contact origin. Moving to the center of normal
pressure removes the transverse couple; force residuals become 0.070009, 0.617335
and 0.192376 N respectively. The corresponding full relative velocity is evaluated
at that physical point; t is not relabeled. 100 origin-shift and 300 length-scale
controls preserve body torque, work and stored energy near roundoff.
This does not prove every possible directional closure is incompatible. It shows
that origin/pressure geometry and the remaining mixed-force discrepancy need
reconciliation before equivalence/adoption. Zero velocity/spin needs a static
direction convention too.

\newpage
\section*{Efficiency with identical prescribed sites}
Warning-free C++17 \texttt{-O3 -Wall -Wextra -Werror}, no fast-math.
The timed operation includes body gather, patch force/couple, ordered body
scatter and output resets. All 44 first-run batches (100, 10,000, 100,000 or one
million responses) improve. A second 22-batch run alternates timing order and improves
every batch too. 48 native/Python controls pass each run, maximum scaled
error $2.28\times10^{-15}$. These are frozen responses, not interacting scenes.

The first-run million-response totals are below. The last column gives the
alternating-order confirmation range for 10,000/100,000 responses, not a statistical
confidence interval or a million-body result.
\begin{center}\small\begin{tabular}{lrrrrr}\toprule
Case & Sites & Direct ms & Compact ms & Gain & Confirmation\\\midrule
@@BENCHMARK@@
\bottomrule\end{tabular}\end{center}
\includegraphics[width=\linewidth]{kernel-gains.pdf}
The large no-twist gains come from eliminating site loops. Mixed-spin/opening
gains come from compact component arithmetic with the same sites. Footprint
generation, frame transformations, shear/static branches, time integration,
detection and group solves are outside timing. Allocations are excluded. First
run's inaccurate array-size estimate is retained and corrected in v2; timings
and wrenches are unchanged. Host load affects timing, and no end-to-end claim or
renewed 2x gate is made.

\newpage
\section*{Patch accuracy: difficult states stay visible}
The fast exact reduction has the same prescribed-site output; that alone cannot
certify a continuous pressure-patch approximation. Near a local slip-zero,
fixed coarse quadrature is inaccurate and convergence can be nonmonotone.
We probe 18 mixed states for circular/elliptic/irregular shapes against a
147,456-site reference. Two consecutive small changes are required to admit a
bounded refinement estimate. @@ACCEPTED@@/18 estimates admit; all 18 finest
returned values happen to satisfy the $10^{-4}$ scaled-error target, maximum
@@WORST@@. The six declined estimates remain declines. This finite check is
not a rigorous bound for arbitrary states, and 65,536-site worst branches are
too expensive for a cheap general contact law.

\includegraphics[width=\linewidth]{refinement-error.pdf}
Open dots: fixed 256 sites. Filled dots: admitted refinement estimates. Crosses:
budget-ended declines. Vertical segments connect the same inputs. Error is the
maximum of force error/N and couple error/(N a). It compares numerical integration
of the candidate, not reality.

\includegraphics[width=\linewidth]{deviation-map.pdf}
Signed force-size error and couple-angle error against the same fine-patch input,
with shape colors. Dots use 256 sites; crosses use 4,096. The reference point is
the origin. These are numerical error points, not experimental accuracy points.

\newpage
\section*{Where the real-data comparison stands}
This checkpoint adds no measured collision records and fits no coefficients.
The earlier comparison tables remain unchanged:
\begin{center}\begin{tabular}{p{.20\linewidth}p{.69\linewidth}}\toprule
Dataset & Status retained from the experimental report\\\midrule
Glass, 24 rows & Fixed documented comparator: normal/tangent COM RMSE 0.036674/0.017805 m/s; spin reconstructed and nonindependent. Not validation of this patch.\\
Ball/surface, 8 summaries & Target-supplied tangential restitution includes measured spin; conditional comparison. Earlier shared moment fit worsens error and remains rejected.\\
Rocks, 75 rows & Historical fitted sphere proxy: 25-height-test RMSE 1.004366/1.304693 m/s and 12.876456 rad/s. Missing actual attitude/inertia/footprints; damping-only extension worsens response and remains rejected.\\
Rolling, 17 points & Earlier relaxation reduction improves 8 reused evaluation points by 51.53/67.46/33.71 percent across three specimens. Exploratory reuse, not new blind validation or identified bulk relaxation.\\\bottomrule
\end{tabular}\end{center}

\textbf{Next authenticity work.} Recover matched footprint/pressure or force-time
measurements and independently characterized stiffness/damping/shear response;
keep documented friction and restitution fixed. A finite shear-history/mode branch
is needed for grip/slip reversal and restitution, and local yield/indentation
geometry for damaged concrete. Reconcile the admissible directional map before
adoption. Replace expensive singular mixed-state quadrature with a reduced,
energy-consistent approximation that passes the same force/couple error controls.
No per-case endpoint correction or newly invented material coefficient is added.

\textbf{Precedent.} Elandt, Drumwright, Sherman and Ruina (IROS 2019),
\href{https://arxiv.org/abs/1904.11433}{pressure-field contact}, motivates resultant
force/moment from locally compliant traction while retaining rigid bodies. Our
flat foundation is simpler and does not implement their intersection geometry.
\href{https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html}{Drake's guide}
describes the pressure-patch approach and warns that automatic/default material
values need application-specific characterization. No novelty or measured-material
authenticity follows from adopting a patch representation.

\textbf{Reproduce.} See README and INDEXED-DESIGN. Authoritative new evidence:
audit-v2, refinement-v2, benchmark-v1 full ladder and benchmark-v2 alternating-order
confirmation. v1 metadata-label mistakes remain retained with explicit correction
receipt and unchanged numerical values. No production source is changed.
\end{document}
'''
for key,value in dict(POWER=f"{audit['maximum_scaled_power_balance_error']:.3g}",INCIDENCE=f"{audit['maximum_indexed_work_momentum_error']:.3g}",ODE=f"{audit['normal_transient_energy_error']:.3g}",DIRECTIONS='\n'.join(directions),BENCHMARK='\n'.join(bench_rows),ACCEPTED=str(accepted),WORST=f"{worst:.3g}").items():text=text.replace('@@'+key+'@@',value)
(here/'report.tex').write_text(text)
print('Figures and report.tex written')
