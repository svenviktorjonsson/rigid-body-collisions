"""Export the signed error scatter and a concise downloadable model update."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

H=Path(__file__).resolve().parent
r=json.loads((H/'evidence-v1/results.json').read_text())
b=json.loads((H/'benchmark-v2/results.json').read_text())
a=json.loads((H/'audit-v2/audit.json').read_text())

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(11,4.6))
colors=['#0072B2','#D55E00','#009E73']
labels=['Moderately old tennis','Old tennis','New tennis']
for data,color,label in zip(r['tennis']['datasets'],colors,labels):
    rows=[p for p in data['records'] if p['split']=='held_out']
    for p in rows:
        x=p['omega_rad_s'];obs=p['observed_mu_r']
        base=100*(p['constant_prediction']/obs-1);candidate=100*(p['relaxation_prediction']/obs-1)
        ax[0].plot([x,x],[base,candidate],color=color,alpha=.35,lw=1)
        ax[0].scatter([x],[base],marker='x',color=color,s=55,alpha=.5)
        ax[0].scatter([x],[candidate],marker='o',color=color,s=42)
    ax[0].scatter([],[],color=color,label=label)
ax[0].axhline(0,color='#222',lw=1)
ax[0].set(xlabel='Apparent angular speed (rad/s)',ylabel='Rolling-coefficient relative error (%)',
          title='Rolling: all 8 evaluation errors decrease')
ax[0].legend(fontsize=8)
models=r['rocks']['height_split']
for kind,marker,alpha in [('constant','x',.5),('viscoelastic','o',.8)]:
    for diameter,color,label in [(.1,'#0072B2','10 cm proxy'),(.2,'#D55E00','20 cm proxy')]:
        rows=[p for p in models[kind]['records'] if p['diameter_m']==diameter]
        ax[1].scatter([p['incoming_normal_m_s'] for p in rows],
            [p['prediction'][0]-p['observed'][0] for p in rows],color=color,marker=marker,s=35,alpha=alpha,
            label=label if kind=='viscoelastic' else None)
ax[1].axhline(0,color='#222',lw=1)
ax[1].set(xlabel='Incoming normal speed (m/s)',ylabel='Outgoing normal speed error (m/s)',
          title='Rocks: viscoelastic sphere law is worse')
ax[1].legend(fontsize=8)
fig.legend(handles=[Line2D([],[],marker='x',linestyle='',color='#555',label='Constant comparator'),
                    Line2D([],[],marker='o',linestyle='',color='#555',label='Relaxation model')],
           loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,0))
fig.tight_layout(rect=(0,.065,1,1))
fig.savefig(H/'error-map.pdf');fig.savefig(H/'error-map.png',dpi=180);plt.close(fig)

def tex_escape(text):return text.replace('_',r'\_')
rows=[]
for d,label in zip(r['tennis']['datasets'],labels):
    rows.append(f"{label} & {d['constant']['rmse']:.6f} & {d['relaxation']['rmse']:.6f} & {d['rmse_reduction_percent']:.1f}\\% \\\\")
rolling_rows='\n'.join(rows)
param_rows='\n'.join(f"{label} & {d['estimated_effective_relaxation_s']*1000:.3f} & {d['training_count']} & {d['evaluation_count']} \\\\" for d,label in zip(r['tennis']['datasets'],labels))
bench_rows=[]
for batch in b['batches']:
    k={p['kernel']:p['nanoseconds_per_response'] for p in batch['kernels']}
    bench_rows.append(f"{batch['count']:,} & {k['constant_rolling']:.2f} & {k['relaxation_exp']:.2f} & {k['relaxation_cached_factor']:.2f} & {k['normal_size_speed_and_table']:.2f} \\\\")
bench_rows='\n'.join(bench_rows)
template=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=21mm]{geometry}
\usepackage{amsmath,amssymb,array,graphicx,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue,pdftitle={Efficient contact deformation model update}}
\newcommand{\rw}[1]{\mathord{#1\mkern2mu\wedge}}
\newcommand{\lw}[1]{\mathord{\wedge\mkern2mu#1}}
\setlength{\parindent}{0pt}\setlength{\parskip}{7pt}
\setlength{\emergencystretch}{2em}
\begin{document}
\begin{center}\Large Efficient contact deformation model update\\[4pt]
\normalsize Rigid bodies, local material memory and independent torque\\7 October 2026\end{center}

\textbf{There is a promising cheap improvement for sustained rolling. There is no
new validated solution for all experimental collisions.} A one-parameter
speed-dependent rolling law improves all eight evaluation points in the existing
tennis data partition. A related normal-impact law makes the rock sphere-proxy
predictions worse; that rock extension is rejected. Both results are retained.

\section*{What changed, and what remains fixed}
The bodies remain rigid. Local compression and material relaxation represent
unresolved deformation. Normal and tangential restitution, static friction
$\mu_s$ and dynamic friction $\mu_d$ already documented for existing profiles
remain unchanged. There is no substitution of a sliding coefficient for a measured
static coefficient. The new estimates concern missing effective rolling behavior
for three tennis specimens. They are not universal rubber material properties.

One nonnegative relaxation parameter per specimen is compared with one nonnegative
constant rolling coefficient per specimen. Both have the same parameter count.
No per-outcome adjustment or negative friction is used. The physical suggestion
comes from viscoelastic theory, but these quasirolling tennis data alone cannot
identify a true continuum relaxation time.
\begin{center}\small
\begin{tabular}{lrrr}\hline
Specimen & Constant RMSE & Relaxation RMSE & Reduction\\\hline
@@ROLLING_ROWS@@
\hline\end{tabular}\end{center}
RMSE is in the dimensionless effective rolling coefficient. Every one of the eight
evaluation points has a smaller absolute residual; the maximum error decreases
for each specimen. These points were not used to estimate the parameter, but
the data and a first error preview were already inspected during model selection.
This is \textbf{exploratory reuse of a split}, not new blind validation.

\includegraphics[width=\linewidth]{error-map.pdf}
The horizontal black line is zero error. Crosses denote the constant comparator;
dots denote the relaxation candidate. The left observable is rolling resistance,
not a measured outgoing angular-velocity vector. The right contains all 25
release-height evaluation records for rocks; no unfavorable record is removed.

\newpage
\section*{Contact-local mechanics with the user's impulse notation}
Let $\tau_v$ denote relaxation time; the source denotes it by $A$.
For a supported sphere of radius $R$, mass $m$, inertia $\mathbb I=\alpha mR^2$,
normal load $N$, and pure rolling angular speed $\omega\ge0$, the restricted
viscoelastic law is
\[
 \mu_r=\tau_v\omega,\qquad
 \delta L_s=-R\int\mu_r N\,dt,\qquad
 \delta\mathbf L=\delta L_s\hat{\mathbf s}.
\]
The independent angular impulse is essential. The body change includes the
force lever arm as well as this independent impulse:
\[
 \Delta\mathbf L=\rw{\mathbf r}\,\delta\mathbf p+\delta\mathbf L,\qquad
 \mathbf r=-R\hat{\mathbf n}.
\]
The supported no-slip condition is
\[
 \mathbf v+\lw{\mathbf r}\,\boldsymbol\omega=\mathbf0.
\]
The right-hand wedge denotes the transpose; factor order is preserved. Resolving
the static reaction together with the independent torque gives
\[
 \mathbb I\dot\omega+mR^2\dot\omega=-\tau_v NR\omega,
 \qquad
 \gamma=\frac{\tau_v NR}{\mathbb I+mR^2}.
\]
For constant support load and geometry the exact finite update is
\[
 \omega^+=\omega^-\exp(-\gamma h),\qquad
 \delta\mathbf L=\bigl(\mathbb I+mR^2\bigr)(\omega^+-\omega^-)\hat{\mathbf s}.
\]
The parentheses here group a scalar sum. Matrix associativity does not allow
removing grouping around sums. The linear impulse follows from $m\Delta\mathbf v$.
It balances the force-induced angular change; it must not be omitted.

With $N=mg$, the instantaneous static demand satisfies
\[
 \frac{|F_t|}{N}=\frac{\tau_v\omega}{1+\alpha}\le\mu_s.
\]
It is largest initially and decreases during this branch. The implementation
rejects insufficient static capacity rather than asserting no slip. A sliding
branch using $\mu_d$ is not implemented here. The state has no axial spin; it
does not establish rolling/twisting closure for arbitrary 3D contacts.

The rigid kinetic energy is $E=\tfrac12(\mathbb I+mR^2)\omega^2$, and the
loss over a step is $E^-\bigl[1-\exp(-2\gamma h)\bigr]\ge0$. There is no
resistive spin reversal. Small increments use \texttt{expm1} for numerical accuracy.

The scaled variables remain
\[
 \mathbf V=\begin{bmatrix}\mathbf v\\\ell\boldsymbol\omega\end{bmatrix},\quad
 \mathbf P=\begin{bmatrix}\mathbf p\\\mathbf L/\ell\end{bmatrix},\quad
 \Delta\mathbf P_c=\begin{bmatrix}\delta\mathbf p\\\delta\mathbf L/\ell\end{bmatrix}.
\]
The reference length $\ell$ changes representation, not resistance or motion.
At zero relative contact velocity the user's direction $\hat{\mathbf t}$ is
undefined. This prototype obtains the static reaction from the rolling constraint;
it does not silently redefine $\hat{\mathbf t}$ or claim to have completed that
directional branch. This issue must be resolved when integrating contact history.

\newpage
\section*{How indentation can determine normal restitution cheaply}
Under a small-deformation spherical contact, a viscoelastic reduction is
\[
 F_n=\kappa x^{3/2}+\frac32\tau_v\kappa\sqrt{x}\,\dot x,\qquad
 m_{\rm eff}\ddot x+F_n=0.
\]
Here $x$ is compression, not an impulse symbol. The elastic energy and loss rate are
\[
 U=\frac25\kappa x^{5/2},\qquad
 D_n=\frac32\tau_v\kappa\sqrt{x}\,\dot x^2\ge0.
\]
For incident normal speed $u$, the isolated collision reduces to one dimensionless
parameter
\[
 \beta=\frac32\tau_v\left(\frac{\kappa}{m_{\rm eff}}\right)^{2/5}u^{1/5}.
\]
The prototype integrates this reference problem once to build a monotone table.
Runtime restitution can then be obtained by a cubic lookup. For profiles using
a measured restitution at a reference condition, matching that characterization
must remain an explicit calibration with known stiffness and mass. A single
restitution value is not enough to identify all deformation properties.

Release occurs when the normal force first reaches zero during unloading.
For $\tau_v>0$ this happens before compression reaches zero. Extending the law
until geometric recovery creates an attractive interval. The remaining spring
energy is recorded separately as unresolved internal recovery. It is not erased
or counted as proven heat. The isolated endpoint table does not track that internal
state into a later impact. Endpoint restitution must not be imposed a second time
on top of the resolved damping.

\textbf{Efficient isolated pair maps are not simultaneous-contact solvers.}
Using one whole-impact map per pair in rapid groups can violate coupled momentum,
work and contact history. A group implementation needs the coupled mobility solve
already derived in the article, with local history and branch transitions.

\section*{The rock experiment rejects this simple viscoelastic extension}
Both candidates fit one normal-response parameter using the same normalized
normal-speed residual on 50 records. They evaluate on the existing 25 records
released from 4.5 m. A further four-fold check leaves out complete slab-angle groups.
The viscoelastic size law assumes homogeneous spherical mass and curvature:
$\beta=b(0.05\,\mathrm m/R)(u/(1\,\mathrm{m/s}))^{1/5}$.
The fitted $b$ is an effective proxy, not independently identified viscosity.

\begin{center}\small
\begin{tabular}{lrr}\hline
Evaluation & Constant normal law & Viscoelastic sphere law\\\hline
25 height records, normal RMSE (m/s) & 1.00437 & 1.13969\\
75 pooled angle-fold records, normal RMSE (m/s) & 1.14028 & 1.19421\\
Height-set maximum error (m/s) & 2.29727 & 2.34718\\
Angle-fold maximum error (m/s) & 2.33496 & 2.84656\\\hline
\end{tabular}\end{center}
The extension worsens both splits and their maximum errors. It is rejected for
these rock records. Conditional tangential/spin calculations keep the historical
fitted coefficients; these are not documented material values or actual-shape tests.
Actual facet attitude and inertia remain unavailable. Failure does not show that
all viscoelasticity is absent; it shows this particular reduction is insufficient.

\newpage
\section*{A better cheap direction for rocks: local indentation geometry}
The source experiment used faceted limestone samples and softer concrete slabs.
It documents repeated surface damage, indentation edges and possible rapid second
contacts that alter rebound direction. Its figure 12 labels these indentation
dimensions:
\begin{center}\small
\begin{tabular}{rrr}\hline
Diameter (cm) & Depth (cm) & Simplified conical wall angle\\\hline
3.5 & 1.5 & 40.60 degrees\\
1.7 & 1.6 & 62.02 degrees\\
2.3 & 0.6 & 27.55 degrees\\
2.1 & 0.5 & 25.46 degrees\\\hline
\end{tabular}\end{center}
These photographs are not matched to individual collision rows. Conical geometry
is an illustrative reduction, not an inferred experimental contact frame. The
depths are too large to assume the small Hertz half-space limit automatically.

A synthetic frictionless control makes the distinction clear. A rigid sphere
arrives with $\mathbf v^-=[3,-1,0]^T$ m/s, mass 1 kg and local $e_n=0.5$.
Against a horizontal normal, it leaves with $[3,0.5,0]^T$ m/s. Against a local
normal $[-0.5,0.86603,0]^T$ tilted 30 degrees from the mean normal, it leaves with
$[1.22548,2.07356,0]^T$ m/s. The ratio measured against the mean slab normal is
2.07356, while total kinetic energy decreases from 5 to 2.90072 J.
The actual local restitution is still 0.5. Tangential incident energy is redirected
upward; no negative friction or active elasticity is needed.

This control does not predict a real rock. A central sphere normal force has zero
lever-arm torque. For facets, a pressure centroid displaced from the nominal contact
point can also contribute an independent angular impulse. Distributed force and
moment must be accumulated before checking the permitted directional components.

The next rock candidate should therefore retain a local indentation depth and
rim/pressure geometry, use a documented or separately estimated yield/indentation
law, and resolve a few contact sites. Local normals and pressure lever arms then
follow geometry instead of a fitted function of the measured rebound. Bodies
remain rigid. A few contact states add cost in proportion to contacts rather than
body mesh nodes. Nominal C25 concrete strength is not an indentation-hardness
measurement and cannot simply be substituted for one.

\section*{A better rubber bounce candidate: shear history plus pressure moment}
Primary ball experiments show grip, tangential vibration and friction-force reversal.
The rolling law above describes sustained support; it is not automatically a law
for the short grip phase of a bounce. A small shear state and one or two independently
characterized vibration modes can represent that phase. Static traction can return
stored elastic energy while sliding friction opposes actual slip. Asymmetric
normal pressure can supply the separate angular impulse.

For a practical benchmark, compare a force-history Mindlin reduction and a tiny
contact-patch reference. Force history handles unloading more carefully than some
displacement-rescaling variants, but energy accounting and frame transport must
still be checked. Neither changing stiffness without accounting for stored energy
nor clearing elastic history at separation is acceptable by itself. This continuation
implements no new general shear/plastic/opening branch or empirical rubber-bounce fit.

\newpage
\section*{Independent spin torque without a new fitted friction coefficient}
An additional restricted patch calculation directly addresses the zero-center-slip
spin case. For pure spin about the normal of a circular Hertz contact, the center
has zero tangential velocity but the rest of the patch slides. With radial pressure
$p(\rho)=p_0\sqrt{1-\rho^2/a^2}$, contact radius $a$, and the existing dynamic
sliding coefficient $\mu_d$, the integrated load and moment magnitude are
\[
 N=2\pi\int_0^a p(\rho)\rho\,d\rho=\frac23\pi p_0a^2,
 \qquad
 T=2\pi\mu_d\int_0^a p(\rho)\rho^2\,d\rho
   =\frac{\pi^2}{8}\mu_dp_0a^3.
\]
Consequently,
\[
 T=\frac{3\pi}{16}\mu_dNa,\qquad
 \delta p_t=0,\qquad
 \delta\mathbf L=\delta L_n\hat{\mathbf n}.
\]
The linear shear forces cancel by symmetry; their moment does not. The small
impulse $\delta L_n$ changes spin even though a center-point sliding rule would
produce no tangential impulse. This is a standard traction integral, not a claim
of new mechanics. For constant load and patch size with isotropic inertia, the
ideal pure-sliding update followed by zero-external-torque arrest is
\[
 \delta L_n=-\operatorname{sgn}(\omega_n)
 \min\!\left(\frac{3\pi}{16}\mu_dNah,\;\mathbb I|\omega_n|\right),
 \qquad \omega_n^+=\omega_n^-+\frac{\delta L_n}{\mathbb I}.
\]
The mechanical loss is
$-\omega_n^-\delta L_n-\delta L_n^2/(2\mathbb I)\ge0$.
No extra spinning-friction coefficient is fitted. The physical length is contact
radius $a$, not the coordinate reference length $\ell$.

Twenty-four rotated spatial controls independently integrate distributed tractions
and verify net force, torque, power and body kinetic energy. Zero friction and
zero spin preserve the state; resistance arrests spin without reversal.
For axial spin $\hat{\mathbf s}$ and $\hat{\mathbf n}$ are dependent; the prototype
allocates this resultant to $\delta L_n$, not to two independently identifiable
scalar couples. Mixed spin/translation, torsional elastic microslip and changing
normal pressure are not resolved here. A transient's pressure distribution can
differ from the Hertz profile, so this coefficient must not be transferred blindly
to a rubber bounce. No measured spin improvement is claimed from these controls.

\section*{Literature comparators for the next coupled model}
The pressure-field contact model of Elandt, Drumwright, Sherman and Ruina computes
resultant force and moment from a contact patch while bodies stay nominally rigid.
It is a useful reference for irregular geometry and independent moments. The
pressure fields are prepared in advance; no body deformation field is solved at
runtime. Its reported scope does not capture internal wave dynamics of shells,
so a pressure field alone is insufficient for every rubber-ball vibration.

Drake provides an implementation reference. Its pressure modulus, resolution and
damping need characterization; default values are not automatically documented
material properties. This continuation did not run Drake or validate a pressure
model against the collision data.

For local plastic indentation, Zunker and Kamrin's dimensionality-reduced
elastic-perfectly-plastic contact models offer a mechanically derived comparison
with contact history. They do not by themselves validate brittle rock/concrete
crushing or recover missing facet orientation. Both references help constrain
the next small patch/history model; neither is silently adopted as a material fit.

\newpage
\section*{Physical verification and cost}
The exact rolling update passes 100 planar and 100 rotated spatial controls at
three representation lengths, including direct linear/angular momentum, kinetic
energy, zero slip, time subdivision, zero resistance and static-capacity rejection.
Twenty independent continuous-ODE comparisons pass. Maximum relative errors are
about $2.04\times10^{-15}$ for momentum and $1.15\times10^{-15}$ for energy.
This is a 3D sphere law restricted to planar motion when appropriate; it is not a
derived plane-strain constitutive law for arbitrary 2D shapes.

Normal controls recover the analytic elastic collision duration and the known
weak-damping coefficient. Twenty-five tighter reference controls keep energy
balance error below $4.54\times10^{-11}$ in dimensionless energy. The 513-node
normal table covers $0\le\beta\le8$; 64 independent random tighter solves show
maximum restitution error $1.35\times10^{-7}$. Out-of-range use is rejected.
Grid arrays plus interpolation coefficients occupy 24,592 bytes, excluding
language/object overhead. Local Python setup took about 7.75 seconds.

The C++ benchmark uses double precision, \texttt{-O3} and no fast-math.
Sixteen native/Python controls pass at rounding error; cached/uncached rolling
checksums match. Median local kernel costs are:
\begin{center}\small\setlength{\tabcolsep}{4pt}
\begin{tabular}{rrrrr}\hline
Responses & Constant roll & Relaxation, exp & Cached factor & Normal map\\\hline
@@BENCH_ROWS@@
\hline\end{tabular}\end{center}
All timings are nanoseconds per scalar response, including resultant impulse and
rigid-energy changes. The normal kernel includes size/speed conversion and table
lookup. The cached rolling factor requires unchanged support load, geometry and
timestep; otherwise the exponential must be recomputed. One million response
records occupy 88 MB of input/output arrays. These are synthetic array kernels,
not one million interacting bodies or the native engine's full scenes. Timings
exclude detection, evolving frames, contact networks, branch transitions and
simulation output. No all-case performance or 2x claim is made.

\section*{Parameters and evidence boundaries}
\begin{center}\small
\begin{tabular}{lrrr}\hline
Specimen & Estimated effective $\tau_v$ (ms) & Fit points & Evaluation points\\\hline
@@PARAM_ROWS@@
\hline\end{tabular}\end{center}
Reported one-PDF-point sensitivities are about 1.826--1.895, 2.122--2.197 and
24.521--25.562 ms respectively. They are digitization sensitivity ranges, not
confidence intervals. Leave-one-point-out diagnostics also favor the relaxation
law for each specimen. The source uses belt-derived apparent speed and allows
slight skid. It supplies no matched specimen mass, inertia, normal stiffness,
impact restitution and static/dynamic friction characterization. Hence no claim
that these fitted effective times can already predict the same balls' bounces.

The production engine is unchanged. General full-wrench contact closure, native
rolling/twisting, rapid irregular-group qualification and independently characterized
full-state comparisons remain open. The numerical report's prior failures remain
available, and the user-removed 2x gate is not reinstated.

\newpage
\section*{Sources and reproducibility}
Brilliantov and P\"oschel, \emph{Rolling as a continuing collision},
EPJ B 12 (1999), 299--301;
\url{https://arxiv.org/pdf/cond-mat/0203343}.
Source rolling coefficient has dimensions of length: $M=\mu_{\rm roll}N$ and
$\mu_{\rm roll}=\tau_v v$. Here $M=\mu_r RN$, so for pure rolling $\mu_r=\tau_v\omega$.
Its scope requires small deformation and sufficiently slow quasi-static rolling
relative to material relaxation; the tennis apparatus does not independently
establish those assumptions.

Schwager and P\"oschel, \emph{Coefficient of restitution for viscoelastic spheres:
the effect of delayed recovery}, Phys. Rev. E 78 (2008), 051304;
\url{https://arxiv.org/pdf/0708.1434}. Repulsive force-zero release and delayed
recovery motivate the normal reference stopping rule.

Cross, \emph{Grip-slip behavior of a bouncing ball}, Am. J. Phys. 70 (2002),
1093--1102; \url{https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/21.\%20GripSlip.PDF}.
Cross, \emph{Impact of a ball on a surface with tangential compliance},
Am. J. Phys. 78 (2010), 716--720;
\url{https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/47.\%20MoreSpin.pdf}.
These sources support retaining contact shear/mode history rather than assuming
all momentary zero slip is ordinary sustained rolling.

Wang et al., \emph{Effects of the impact angle on the coefficient of restitution
in rockfall analysis based on a medium-scale laboratory test}, NHESS 18 (2018),
3045--3061; \url{https://doi.org/10.5194/nhess-18-3045-2018}.
Sections 2.2 and 5.1 and figure 12 document slab damage and proposed rebound
constraints. Figure values are not per-impact calibration data.

Singh et al., 2008 tennis quasirolling source,
\url{https://arxiv.org/pdf/0809.4823v2}. Experimental blue points from page 6 are
reused from the separately archived full report; author regression curves are
not the new validation targets.

LAMMPS granular model documentation,
\url{https://docs.lammps.org/pair_granular.html}, accessed 7 October 2026.
Its comparison of force/displacement unloading histories supplies an implementation
reference, not independent proof of this project's contact law.

Elandt et al., \emph{A pressure field model for fast, robust approximation of net
contact force and moment between nominally rigid objects}, IROS 2019;
\url{https://arxiv.org/abs/1904.11433}.
Drake Hydroelastic Contact User Guide,
\url{https://drake.mit.edu/doxygen_cxx/group__hydroelastic__user__guide.html}.
Zunker and Kamrin, \emph{A mechanically-derived contact model for adhesive
elastic-perfectly plastic particles}, parts I and II;
\url{https://arxiv.org/abs/2309.07300} and
\url{https://arxiv.org/abs/2309.07317}.

Reproduce from \path{research/viscoelastic-relaxation/} with NumPy, SciPy and
Matplotlib installed. Run:
\begin{quote}\ttfamily\small
experiment.py --output NEW-DIRECTORY\\
audit.py --evidence NEW-DIRECTORY --output NEW-AUDIT\\
benchmark.py --evidence NEW-DIRECTORY --output NEW-BENCHMARK
\end{quote}
Outputs refuse to overwrite existing evidence. \texttt{render.py} builds the
authoritative scatter and report source; \texttt{build.sh} compiles this PDF.
Machine-readable evidence contains all points, folds, retained failures and
source fingerprints. Third-party PDFs are cached outside the repository.
\end{document}
'''
template=template.replace('@@ROLLING_ROWS@@',rolling_rows).replace('@@PARAM_ROWS@@',param_rows).replace('@@BENCH_ROWS@@',bench_rows)
(H/'report.tex').write_text(template)
print(json.dumps({'figure':'error-map.pdf','report_source':'report.tex'}))
