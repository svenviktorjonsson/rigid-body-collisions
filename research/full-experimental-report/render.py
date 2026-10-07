"""Standalone PDF/HTML report with full row tables and a deviation scatter."""
import argparse
import html
import json
import math
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def escape(x):
    x=str(x)
    return ''.join({'\\':r'\textbackslash{}','_':r'\_','%':r'\%','&':r'\&','#':r'\#','{':r'\{','}':r'\}'}.get(c,c) for c in x)


def table(headers,rows):
    return '\n'.join([r'\begin{center}\footnotesize\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{'+'l'*len(headers)+'}', r'\hline',
        ' & '.join(escape(x) for x in headers)+r'\\\hline',
        *[' & '.join(escape(x) for x in row)+r'\\' for row in rows],
        r'\hline\end{tabular}\end{center}'])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('evidence',type=Path);args=parser.parse_args()
    r=json.loads((args.evidence/'results.json').read_text())
    audit=json.loads((args.evidence/'independent-audit.json').read_text());assert audit['pass_']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,ax=plt.subplots(figsize=(8,4.5))
    points=[]
    for row in r['glass']['records']:
        points.append(dict(label=f"Glass row {row['source_excel_row']}",group='Glass: fixed inputs, no free couple',color='#2467b3',**row['diagram']))
    for row in r['balls']['records']:
        R=row['radius_m']; p=row['held_out_signed_moment']['predicted_spin_factor_rad_m']; et=row['fixed_e_t'];en=row['fixed_e_n'];a=math.radians(25)
        predicted=abs(np.array([en*math.cos(a),R*p-et*math.sin(a),R*p]))
        observed=abs(np.array(row['conditional_measured_signature']))/4
        angle=math.degrees(math.atan2(np.linalg.norm(np.cross(predicted,observed)),predicted@observed))
        size=100*(np.linalg.norm(predicted)/np.linalg.norm(observed)-1)
        group='Superball: held-out moment hypothesis' if row['ball']=='superball' else 'Golf: held-out moment hypothesis'
        points.append(dict(label=row['ball']+' / '+row['surface'],group=group,color='#d47912' if row['ball']=='superball' else '#478447',
                           signature_angle_error_deg=angle,signature_size_error_percent=size))
    for group in dict.fromkeys(p['group'] for p in points):
        pts=[p for p in points if p['group']==group]
        ax.scatter([p['signature_angle_error_deg'] for p in pts],[p['signature_size_error_percent'] for p in pts],color=pts[0]['color'],label=group,s=33,alpha=.85)
    ax.axhline(0,color='#888',lw=.8);ax.axvline(0,color='#888',lw=.8)
    ax.set(xlabel='Angle between observable magnitude signatures (degrees)',ylabel='Relative signature size error (%)')
    ax.legend(fontsize=8);ax.grid(alpha=.15);fig.tight_layout()
    fig.savefig(HERE/'deviation-map.png',dpi=180);fig.savefig(HERE/'deviation-map.pdf');plt.close(fig)
    (HERE/'deviation-map.json').write_text(json.dumps(dict(points=points,definition='Glass signature: abs(normal velocity), abs(COM tangent velocity). Ball signature: abs(normal velocity), abs(COM tangent velocity), R*abs(spin); normal/tangent values use source restitution. Angle is not a spatial direction; no pooled accuracy score across signatures. Rocks excluded because this map is not validation of actual rock geometry.'),indent=2)+'\n')
    fig,ax=plt.subplots(figsize=(8,4.3))
    for d,c in zip(r['tennis']['datasets'],('#2467b3','#478447','#d47912')):
        for split,marker in [('train','o'),('held_out','s')]:
            ps=[x for x in d['points'] if x['split']==split]
            ax.scatter([x['apparent_omega_rpm'] for x in ps],[x['measured_effective_mu_r'] for x in ps],color=c,marker=marker,
                       facecolors=c if split=='train' else 'none',label=d['specimen'].replace('_',' ')+' / '+split)
        a,b=d['valid_apparent_rpm_range'];ax.plot([a,b],[d['estimated_constant_mu_r']]*2,color=c,lw=1)
    ax.set(xlabel='Apparent ball angular speed (rpm)',ylabel='Measured effective rolling coefficient')
    ax.legend(fontsize=7,ncol=2);ax.grid(alpha=.15);fig.tight_layout();fig.savefig(HERE/'rolling-data.png',dpi=180);fig.savefig(HERE/'rolling-data.pdf');plt.close(fig)
    pages=[]
    def page(title,content):pages.append((title,content))
    page('Experimental comparison and physical scope',r'''
\textbf{We do not yet have a model validated across the real-life datasets.}
This report covers every presently recovered comparison record: 24 glass collisions,
eight oblique ball/surface summaries, 75 rock impacts, and 17 rolling measurements.
These are public experimental records, not invented target states. They are not
124 independently characterized full-state events: some coefficients summarize
overlapping characterization data. Numerical verification and experimental agreement
are separate questions.

Documented parameters remain fixed. Missing parameters are estimated only in small,
explicit calibration models. Same-outcome fitting is not prediction. Every unfavorable
result is retained. The full engine still lacks validated rolling/torsional material
closure and has unresolved rapid irregular-contact cases. No 2x performance gate is
applied; the user removed it.
''' + table(['Comparison','Records','Outcome','Qualification'],[
        ['Glass fixed-input comparator',24,'0.0367 / 0.0178 m/s RMSE','No independent spin'],
        ['Ball endpoint/moment test',8,'4 passive rolling nominal fits','Singleton calibration'],
        ['Shared moment, held-out surface',8,'RMSE 1.048 to 1.243 rad/m','Worse; rejected'],
        ['Rock sphere-proxy comparator','50 + 25','Spin RMSE 12.88 rad/s','Actual shape absent'],
        ['Tennis rolling, held-out speed','9 + 8','Constant worse than source curves','Quasirolling only'],
        ['Isolated rolling mechanics',200,'All pass at three length scales','Synthetic controls']
    ]) + r'''
The strongest result is a consistent impulse/momentum framework including an
independent angular impulse. That framework alone does not specify authentic contact
forces. Energy-passive estimates can still predict poorly; this is demonstrated here.
No comparison in this report establishes full 3D actual-shape validation with measured
signed angular-velocity vectors and independently measured inertia tensors.
''')
    page('Mechanics used and physical checks',r'''
The notation retains the user's combined variables and inertia symbol:
\[
 \mathbf V=\begin{bmatrix}\mathbf v\\\ell\boldsymbol\omega\end{bmatrix},\quad
 \mathbf P=\begin{bmatrix}\mathbf p\\\mathbf L/\ell\end{bmatrix},\quad
 \Delta\mathbf P_c=\begin{bmatrix}\delta\mathbf p\\\delta\mathbf L/\ell\end{bmatrix}.
\]
The full angular update is
\[
 \Delta\mathbf L_k=\rw{\mathbf r_k}\,\delta\mathbf p+\delta\mathbf L.
\]
The independent couple is not the lever-arm moment counted twice. In the ball
calculations the input has no spin and the plane fixes the onset axis of generated
spin. This is a declared planar onset branch, not a recovered general mixed-spin law.
All scalar directional components remain in the recorded output.

Keep $\hat{\mathbf t}$ along full relative contact velocity. When it contains normal
approach, the source's normal/tangential friction must be transformed through the
direction map. With $c=\hat{\mathbf n}^{\mathsf T}\hat{\mathbf t}$,
\[
 \hat{\mathbf n}^{\mathsf T}\delta\mathbf p=\delta p_n+c\delta p_t,\qquad
 \|Q\delta\mathbf p\|=\sqrt{1-c^2}\,|\delta p_t|,
 \quad Q=\mathbf 1-\hat{\mathbf n}\hat{\mathbf n}^{\mathsf T}.
\]
The report code checks the full impulse reconstructed from this nonorthogonal
direction map; it does not silently redefine $\hat{\mathbf t}$ as another direction.

For pure rolling the physical moment length is $a_r=R$:
\[
 |\delta L_s|\le\mu_r R\hat{\mathbf n}^{\mathsf T}\delta\mathbf p,\qquad
 |\delta L_s/\ell|\le\mu_r R/\ell\;\hat{\mathbf n}^{\mathsf T}\delta\mathbf p.
\]
With $\inertia=\alpha mR^2$, no-slip motion slows at $\mu_r g/(1+\alpha)$ and needs
static capacity $\mu_s\ge\mu_r/(1+\alpha)$. Sliding dissipation is zero on that
branch. The torque supplies dissipation while static friction supplies the reaction.
The coordinate length $\ell$ is not a physical deformation/contact length.

The isolated rolling checks cover 100 planar and 100 arbitrarily rotated spatial
cases, three representation lengths each, angular arrest, a zero-torque control,
independent Newton/Euler balances, and rejection when static capacity is insufficient.
Maximum relative motion, energy and scale errors are approximately $1.1\times10^{-15}$.
These do not validate the production collision solver.
''')
    readiness=r['material_readiness']
    page('Separate material parameter table',r'''
The catalog has 11 published normal/tangential/sliding triples. It does not provide
11 complete profiles for the expanded static/dynamic/rolling model. A missing static
coefficient is not set equal to a sliding coefficient. A missing rolling coefficient
is not called zero measured friction. The legacy catalog and native results remain
historical comparator evidence.
''' + table(['Profile','e_n','e_t','Sliding mu','mu_s','mu_r'],[
        [p['profile_id'],f"{p['e_n']:.3f}",f"{p['e_t']:.3f}",f"{p['documented_sliding_mu']:.3f}",'missing','missing'] for p in readiness
    ]) + r'''
The glass test uses its unchanged documented values. None of the listed missing
angular/static inputs is identified by those 24 records. Public coefficients apply
to their specified material pair, specimen size, condition and measurement convention.
Rubber compounds, felt age, surface roughness and support compliance cannot be erased
by labeling a parameter ``rubber friction.''

Dimensionless rolling coefficients and dimensional moment lengths must be distinguished.
Fuchs et al.'s microsphere measurements support this normalization, but their measured
rolling resistance combines two contacts and adhesion. It is not a macroscopic rubber
or single glass/silicon collision profile. No such transfer is used here.
''')
    page('Glass collisions: fixed parameters versus measured output',r'''
All 24 glass worksheet records are recalculated using the nominal sphere mass/radius,
homogeneous inertia and the documented coefficients. The baseline has zero free couple.
Fresh analytical endpoints agree with the preserved native runs to $10^{-7}$ m/s.
All predictions pass total-energy accounting. The measured outgoing normal velocity
and author-reconstructed COM tangent velocity are compared row by row in the appendix.
''' + table(['Output','RMSE','Interpretation'],[
        ['Normal velocity',f"{r['glass']['normal_rmse_m_s']:.6f} m/s",'Experimental residual'],
        ['COM tangent velocity',f"{r['glass']['center_tangent_rmse_m_s']:.6f} m/s",'Experimental residual'],
        ['Contact tangent velocity','0.062317 m/s','Spin reconstructed by source']
    ]) + r'''
The worksheet reconstructs contact spin using angular momentum. Consequently its
contact tangent output is not an independently measured spin vector. The coefficient
chart may share its characterization trials with the worksheet. This is reproduction
using published inputs, not independent cross-condition material validation.

The nominal diameter and homogeneous tensor are modeling approximations. We do not
fit an independent couple to author-reconstructed spin and then claim its agreement
as proof that the real angular impulse was measured.
''')
    ballrows=r['balls']['records'];cv=r['balls']['leave_one_surface_out']
    page('Ball spin: fixed restitution and a held-out moment test',r'''
For each row the measured normal and tangential restitution are held fixed. The
homogeneous-sphere inertia factor is assumed, not measured. The spin factor
$S=\omega^+/v^-$ is the tested output. At zero incoming spin and incidence angle
$\theta$ to the normal, an independent impulse with signed moment length $d$ gives
\[
 S=\frac{(1+e_t)\sin\theta-d/R\,(1+e_n)\cos\theta}{(1+\alpha)R}.
\]
First test $d=0$. Then fit one shared $d/R$ per ball using three surfaces and predict
the fourth; repeat for all surfaces. A signed pressure moment is distinct from
negative rolling friction. The constant cross-surface hypothesis is not accepted
merely because each prediction passes a total-energy check.
''' + table(['Ball / surface','e_n','e_t','Measured S','d=0','Held out S'],[
        [x['ball']+' / '+x['surface'],f"{x['fixed_e_n']:.2f}",f"{x['fixed_e_t']:.2f}",f"{x['observed_spin_factor_rad_m']:.1f}",f"{x['force_only_spin_factor_rad_m']:.3f}",f"{x['held_out_signed_moment']['predicted_spin_factor_rad_m']:.3f}"] for x in ballrows
    ]) + f"Spin RMSE worsens from {cv['force_only_spin_rmse_rad_m']:.3f} to {cv['signed_moment_spin_rmse_rad_m']:.3f} rad/m; worst error worsens from {cv['worst_force_only_error_rad_m']:.3f} to {cv['worst_signed_moment_error_rad_m']:.3f} rad/m. This shared correction is rejected.\n\n" + r'''
This is conditional endpoint validation: source restitution comes from the same
experiment, while each held-out spin is unused in that fold's new fit. It is not
independent validation of the supplied restitution/friction material values.
In particular the source's $e_t$ itself contains the measured outgoing spin.
Withholding that spin from the new fit is therefore not a blind outgoing-state test.
''')
    page('What the missing angular parameters would have to be',r'''
Estimating a separate moment from each measured spin explains that row algebraically,
but supplies no held-out test. A purely dissipative rolling model requires $\mu_r\ge0$.
At nominal inputs four rows require an assisting couple; fitting a negative friction
coefficient is therefore rejected. This is a necessary endpoint sign test, not
a verified rolling-force history. A deforming elastic patch can return energy via
an assisting moment without violating total passivity. A positive total-energy
check alone does not establish its traction distribution or true contact size.
''' + table(['Ball / surface','Needed mu_r','Offset d (mm)','Passive rolling at nominal'],[
        [x['ball']+' / '+x['surface'],f"{x['required_unconstrained_mu_r']:.5f}",f"{1000*x['singleton_signed_offset']['physical_offset_m']:.3f}",'yes' if x['passive_rolling_explanation_possible_at_nominal'] else 'no'] for x in ballrows
    ]) + r'''
The offset column is a signed inverse-mechanics diagnostic, not measured geometry.
Its magnitude lies within the sphere radius and the reconstructed total outgoing
energy is below incident energy for these rows. The tighter actual patch-radius
bound is unknown. Inertia and force-history errors can mimic the same moment, so
these endpoints do not uniquely identify rolling resistance.

Published incidence, restitution and spin error limits are propagated into the JSON
envelope. They are maximum-error limits, not statistical confidence intervals.
An incompatibility at a nominal point must not be interpreted as a confident material
rejection when that envelope includes zero. No independent $\mu_s$ or $\mu_d$ is
invented from the net impulse ratio; that ratio is only a necessary integrated
capacity diagnostic for a given force history.
''')
    page('Tennis rolling: estimate missing coefficients and test speed dependence',r'''
Experimental circle centers are extracted from the primary PDF's three plots, not
from its regression curves. Duplicate PDF paths are removed. Nine alternating
speed points calibrate one positive constant per specimen; eight other points test
it. The author's published curves remain fixed and are a separate comparator.
''' + table(['Specimen','Estimated mu_r','Held-out RMSE','Fixed curve RMSE'],[
        [d['specimen'],f"{d['estimated_constant_mu_r']:.5f}",f"{d['held_out_constant_rmse']:.6f}",f"{d['held_out_fixed_published_curve_rmse']:.6f}"] for d in r['tennis']['datasets']
    ]) + r'''
\begin{center}\includegraphics[width=.9\linewidth]{rolling-data.pdf}\end{center}
Circles are calibration points; open squares are held-out points. Horizontal lines
are fitted constants. Each constant improves over an absent rolling moment, but
each loses to the fixed speed-dependent curve. A universal constant is especially
inadequate for the new specimen. This motivates rate-dependent deformation loss,
not an arbitrary per-impact correction.

The source curves were fitted by their authors to these published measurements,
including the points withheld from our constant fit. Their smaller residual is
source reproduction, not independent evidence of predictive superiority. Only the
constant-fit residual has the declared local calibration/test separation.

The apparatus measures quasirolling with slight skid. The estimates are restricted
to the reported apparent-speed ranges and specimens. Ball radii/inertia and complete
matched impact parameters are not supplied. No outgoing bounce state is validated
by these rolling points. One-PDF-point digitization sensitivities and leave-one-out
estimate ranges are retained in the results. Negative source intercepts are not
extrapolated into negative material resistance.
''')
    rocks=r['rocks']
    page('Rocks and non-spherical collision scope',r'''
The 75 limestone/concrete records are genuine experimental magnitudes. The earlier
three-parameter sphere surrogate is recomputed unchanged, with its original
50-record calibration and 25-record release-height holdout. These project estimates
are not independently documented material values. The user now permits small fits,
but this does not make a fitted shape proxy an authentic rock model.
''' + table(['Held-out quantity','RMSE'],[
        ['Normal velocity',f"{rocks['held_out_rmse']['normal_m_s']:.4f} m/s"],
        ['Tangent velocity',f"{rocks['held_out_rmse']['tangent_m_s']:.4f} m/s"],
        ['Angular speed',f"{rocks['held_out_rmse']['angular_rad_s']:.4f} rad/s"]
    ]) + r'''
Exact collision mesh, central tensor, attitude and signed angular vectors are absent.
Zero incoming spin and coplanar signed reconstruction are assumptions. A conditional
inverse moment is recorded under those same assumptions, but it cannot identify a
true rolling coefficient. All 75 row values remain in the appendix/CSV; failures
are not removed.

The synthetic rotated cuboid, irregular 2D/3D controls and large rapid worlds are
numerical tests. They cannot replace non-spherical experimental validation. The
13-case rapid/irregular baseline remains unqualified. Production still rejects
nonzero rolling/twisting inputs rather than silently accepting an absent mechanism.
The 200 new rolling checks concern a separate sustained-contact branch.
''')
    page('One deviation diagram, with the observations stated',r'''
Each dot compares a predicted outgoing magnitude signature with a measured one.
Horizontal position is the angle between signatures; vertical position is the
relative difference in signature norm. A perfect match is at $(0,0)$. This angle
is not a physical heading error.
\begin{center}\includegraphics[width=\linewidth]{deviation-map.pdf}\end{center}
Glass uses the two observed translational components. Balls use normal velocity,
tangent velocity and $R|\omega|$; tangent velocity is reconstructed from source
tangential restitution and measured spin. Ball dots use the held-out signed-moment
hypothesis, not same-point exact fits. The signature channels differ, so no pooled
accuracy percentage is computed. Rock dots are excluded from this fresh diagram
because its actual geometry/directional model is not validated.

The standalone HTML plot permits hovering to identify a case. Machine-readable
coordinates, definitions and complete records accompany the PDF. Historical bar
figures are not used as the new model's validation.
''')
    page('Deformation and vibration: the supported next model',r'''
\textbf{Use a finite-duration deformable patch with a small set of internal modes.}
An instantaneous rigid impulse cannot describe the measured stretch, force reversal,
pressure migration and residual vibration. Begin with normal indentation, tangential
shear and a rocking/asymmetric-pressure mode, with positive mass/stiffness and
nonnegative damping. A distributed patch supplies the independent angular impulse:
\[
 \delta\mathbf p=\int\sum_k\mathbf f_k\,dt,\qquad
 \delta\mathbf L=\int\sum_k\rw{\mathbf x_k-\mathbf x_c}\,\mathbf f_k\,dt.
\]
The body's angular update still contains both $\rw{\mathbf r}\,\delta\mathbf p$
and this independent $\delta\mathbf L$. Internal vibration velocities contribute
to each patch point's actual slip. Friction must use that slip, not only an undeformed
rigid contact velocity. The general component map must be checked against those
resultants; silently projecting away an unrepresented moment is unacceptable.

For modal coordinates $\mathbf q$, a compact material model is
\[
 M_q\ddot{\mathbf q}+C_q\dot{\mathbf q}+K_q\mathbf q=H^{\mathsf T}\mathbf f,
 \qquad \mathbf u_c=B\mathbf V+H\dot{\mathbf q}.
\]
Use the gradient of one elastic potential for contact force and moment. Its normal
derivative must be retained if shear stiffness depends on compression. Track rigid
kinetic energy, modal kinetic energy, stored strain and dissipation; do not delete
stored energy when the contact opens. Integrating then yields effective restitution;
do not impose a second endpoint impulse on top of a resolved compliant collision.

Hertz/Mindlin contact and local stick/slip provide an established starting point for
small elastic spheres. Large rubber deformation needs a hyperelastic/viscoelastic
shell or solid description, then modal reduction. Pressure history can produce
rolling resistance; adding a fitted rolling torque to an already resolved loss
mechanism can double-count dissipation. Retain $\mu_r a_r$ only for unresolved loss.
''')
    page('How to select and validate the deformation model',r'''
The Cross 2014 author manuscript contains measured force/spin histories and separate
vibration tests for a hollow ball and a superball. It supports this mechanism, but
its force plate and granite measurements use different counterfaces. Their friction
values must not be mixed. Attached-ball vibration modes also use different boundary
conditions from a free bounce; their frequencies constrain, rather than uniquely
determine, the dynamic contact stiffness. [5]
''' + table(['2014 specimen / pair','Mass','Diameter','Source sliding mu'],[
    ['Hollow ball / granite','44.2 g','59.4 mm','not reliably measured'],
    ['Solid superball / granite','46.2 g','46.0 mm','0.45 +/- 0.08'],
    ['Hollow ball / G10 force plate','same specimen','same specimen','approximately 1.8']
]) + r'''
The reported hollow-ball attached-mode periods are approximately 11 ms normal and
5 ms tangential; the solid ball has a 4.5 ms normal period and 3.5 ms high-frequency
tangential period, plus a slower mode. These are separate from the 2010 58 mm
Superball and are not automatically a controlled size-only comparison. Static
friction and modal damping still require independent estimates.

Keep geometry, independently measured modal periods and pair friction fixed where
available. Identify at most a few missing stiffness/damping or patch parameters from
normal force histories and independent vibration decay, then reserve oblique spin
and velocity outputs. Fit a mode's damping from its decay, not from every desired
outgoing spin. Use one parameter set over speed, incidence and incoming spin.

First qualify normal force/time and deformation, then tangential force sign changes,
then the independent moment/angular response. Compare full histories and endpoints,
including worst-case errors, static/sliding transitions and energy. An apparent
endpoint improvement that breaks force histories, restitution inputs or total
passivity is rejected. Test a small patch-resolution ladder and a time-step ladder.
If two or three modes cannot reproduce those independent checks, obtain a converged
finite-element shell/solid reference and reduce its verified modes.

This is a literature-supported implementation direction, not a demonstrated new
data fit. The earlier constant shear/rocking candidate worsened held-out error by
23.7\%; the new shared moment estimate also worsens it. Neither is adopted. Merely
adding another oscillator and tuning it to every endpoint is not sufficient.

Reproduction: run calculate.py into a fresh output directory, run audit.py against
it, render.py, then build.sh. The authoritative evidence is evidence-v3; v1 and v2
are preserved preliminary checkpoints. Source snapshots, SHA-256 values, all records
and the independent audit accompany the report. No generic material lookup can yet
promise authentic full-motion predictions across the requested cases.
''')
    core=json.loads((HERE/'contact-memory-v2/audit.json').read_text())
    page('Keep rigid bodies: a small contact-memory matrix',r'''
The latest user request favors a contact model that encapsulates local deformation
without deformable meshes for all bodies. Keep the existing rigid-body inertia and
contact mobility. Add elastic history $\boldsymbol\eta$, a small positive stiffness
matrix $K$, and nonnegative damping matrix $C$ at an active contact. Angular history
uses $\ell$ times angular displacement, dual to the angular impulse divided by $\ell$.

For a frozen elastic sticking branch and step $h$, the new research core uses
\begin{align*}
 Z&=\tfrac12h^2K+hC,\\
 [W_d+2Z^{-1}]\boldsymbol\lambda&=-2\mathbf z^- -2Z^{-1}hK\boldsymbol\eta^-,\\
 \mathbf z^+&=\mathbf z^-+W_d\boldsymbol\lambda,\\
 \boldsymbol\eta^+&=\boldsymbol\eta^-+h\mathbf z_m,\qquad
 \mathbf z_m=\tfrac12\mathbf z^-+\tfrac12\mathbf z^+.
\end{align*}
Only a small contact matrix is factored. For constant geometry, step and material
it is cached. Body scatter/gather remains the same impulse update. Linear and
independent angular channels may couple through $K$, while the resulting impulse
components remain $\delta p_n,\delta p_t,\delta L_s/\ell,\delta L_n/\ell$.
The energy balance of this branch is
\[
 \Delta E+\Delta U+h\mathbf z_m^{\mathsf T}C\mathbf z_m=0,\qquad
 U=\tfrac12\boldsymbol\eta^{\mathsf T}K\boldsymbol\eta.
\]
This core is implemented in contact\_memory.py and independently checked with
100 planar and 100 spatial body/contact systems, including 20 dependent-direction
cases. It preserves energy/work and representation-length invariance and shows
second-order convergence to an exact oscillator solution. The physical energy error
is below $1.5\times10^{-15}$ relative in those controls.

Opening, Coulomb yielding, changing directions and nonlinear indentation are not yet
implemented in this new core. It is a verified elastic branch, not a production
collision law or a demonstrated experimental improvement. Contact history cannot
simply be discarded at separation; retained internal vibration or explicit loss
must account for its energy. This is the cheap mechanism to extend, rather than a
per-body deformable mesh.
''' + f"The current Python four-channel cached step measures {core['median_python_cached_4channel_step_microseconds']:.1f} microseconds median locally. This includes validation and Python overhead; it is not a native end-to-end engine benchmark or an all-case performance guarantee.\n")
    page('Rock friction: pressure and plasticity rather than raw impulse powers',r'''
Indentation, asperity crushing and interlocking can add apparent resistance beyond
a sliding-test coefficient. That possibility does not establish which mechanism
dominates these rock records. Keep documented $\mu_s,\mu_d$ as the underlying pair
coefficients. Model an additional passive deformation/ploughing contribution only
when it is independently constrained. Do not silently call net impulse divided by
normal impulse a measured dynamic coefficient.

A physically scaled candidate variable is maximum indentation relative to local
effective curvature, $\chi=\delta_{\max}/R$. For an elastic Hertz reference with
effective modulus $E^*$, coefficient $k_H=4E^*\sqrt R/3$ and incident normal energy
$E_n$, the estimated compression is
\[
 U_n=\tfrac25 k_H\delta^{5/2},\qquad
 \delta_{\max}=\left[\frac{5E_n}{2k_H}\right]^{2/5}.
\]
This is a local proxy valid for small elastic contact, not a solution for crushing
angular rock facets. A yield/history variable is needed once that approximation
breaks down. Thornton--Ning provides an established elastic/plastic normal-contact
route; LAMMPS granular contact demonstrates practical normal, shear-history,
rolling and torsion laws with rigid particles. [10,11]

An $a\chi+b\chi^2$ correction can be tested as a bounded approximation to extra
resistance or a moment-length response over a declared range. It is a hypothesis,
not a proven material law. Fit at most one or two missing coefficients to a training
condition, hold material friction fixed, and reject it if held-out velocity/spin,
worst-case error or passivity worsens. Check identifiability against contact size,
yield and initial-spin uncertainty.

Unnormalized impulse-polynomial coefficients have units and can change meaning
with mass, size and the interval over which impulse is accumulated. For a resting
contact that interval changes with timestep. If an impulse variable is used, it must
refer to the complete impact and an independently specified physical impulse scale,
not the arbitrary coordinate length $\ell$. Pressure, contact work or $\chi$ offer
a clearer route to transferring coefficients across sizes. No pressure-dependent
rock law has been adopted or fitted in the reported comparisons.
''')
    sources=[
        ('Cornell impact coefficients','https://grainflowresearch.mae.cornell.edu/impact/data/Impact%20Results.html'),
        ('Original glass worksheet','https://grainflowresearch.mae.cornell.edu/impact/data/Results-3mmglass-binary'),
        ('Cross 2010: ball/surface experiments','https://www.physics.usyd.edu.au/~cross/PUBLICATIONS/48.%20EnhanceBounce.pdf'),
        ('Wang et al. 2018: rock impacts','https://doi.org/10.5194/nhess-18-3045-2018'),
        ('Cross 2014: oblique rubber bounce, author manuscript','https://www.researchgate.net/publication/278104773_Oblique_Bounce_of_a_Rubber_Ball'),
        ('Singh et al. 2008: rolling experiment','https://arxiv.org/abs/0809.4823'),
        ('Fuchs et al. 2014: rolling/sliding/torsion','https://www2.msm.ctw.utwente.nl/sluding/PAPERS/2014_WeinhartFuchs_GMr.pdf'),
        ('Maw, Barber and Fawcett 1976: elastic oblique contact','https://websites.umich.edu/~jbarber/Wear1976.pdf'),
        ('NASA 2005: measured rolling and failed bounce extension','https://ntrs.nasa.gov/api/citations/20050217091/downloads/20050217091.pdf'),
        ('LAMMPS granular contact: history, rolling and torsion','https://docs.lammps.org/pair_granular.html'),
        ('Thornton--Ning 1998: elastic/plastic normal contact','https://doi.org/10.1016/S0032-5910(98)00099-0')
    ]
    page('Primary sources and current limitations','\n\n'.join(f'[{i}] \\href{{{url}}}{{{escape(name)}}}' for i,(name,url) in enumerate(sources,1)) + r'''

Full measured signed 3D states and central tensors are not available for the
recovered rock fixtures. Dynamic/static/rolling profiles remain incomplete for
most pairs. Published coefficient charts, worksheet reconstructions and singleton
summary rows cannot independently identify every angular/material channel. Those
limits constrain what the present numbers establish; they are not hidden defaults.
''')
    glassrows=r['glass']['records']
    page('Appendix: all 24 glass collision comparisons',table(['Excel row','Normal obs.','Normal pred.','Tangent obs.','Tangent pred.'],[
        [x['source_excel_row'],f"{x['observed'][0]:.5f}",f"{x['predicted'][0]:.5f}",f"{x['observed'][1]:.5f}",f"{x['predicted'][1]:.5f}"] for x in glassrows
    ])+r'All values are m/s. Tangent output is the source-reconstructed COM quantity. No independently measured spin is added.')
    for start in range(0,75,25):
        page(f'Appendix: rock records {start+1}--{start+25}',table(['Row','Split','v_n obs/pred','v_t obs/pred','omega obs/pred'],[
            [x['source_row'],'test' if x['split']=='held_out' else 'train',
             f"{x['measured_magnitudes'][0]:.3f} / {x['predicted_historical_sphere_proxy'][0]:.3f}",
             f"{x['measured_magnitudes'][1]:.3f} / {x['predicted_historical_sphere_proxy'][1]:.3f}",
             f"{x['measured_magnitudes'][2]:.2f} / {x['predicted_historical_sphere_proxy'][2]:.2f}"] for x in rocks['records'][start:start+25]
        ])+r'Velocities are m/s; angular speed is rad/s. Prediction is the historical sphere proxy, not actual rock geometry. ``test'' denotes the original release-height holdout.')
    pointrows=[]
    for d in r['tennis']['datasets']:
        for p in d['points']:pointrows.append([d['specimen'],p['split'],f"{p['apparent_omega_rpm']:.3f}",f"{p['measured_effective_mu_r']:.6f}",f"{p['constant_mu_r_prediction']:.6f}",f"{p['published_curve_prediction']:.6f}"])
    page('Appendix: all 17 digitized rolling measurements',table(['Specimen','Split','rpm','Measured mu_r','Constant','Fixed curve'],pointrows))
    preamble=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{amsmath,amssymb,array,graphicx,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue,pdftitle={Experimental comparison and deformation assessment}}
\newcommand{\rw}[1]{\mathord{#1\mkern2mu\wedge}}
\newcommand{\inertia}{\mathbb I}
\setlength{\parindent}{0pt}\setlength{\parskip}{7pt}
\begin{document}
\begin{center}\Large Experimental comparison and deformation assessment\\[4pt]
\normalsize Rigid-body linear and independent angular impulses\\7 October 2026\end{center}
'''
    # Matrix digits follow the user's double-stroke style even in this shorter report.
    preamble=preamble.replace('\\setlength{\\parindent}',r'''\DeclareFontFamily{U}{reportbb}{}
\DeclareFontShape{U}{reportbb}{m}{n}{<-6> bbold5 <6-9> bbold7 <9-> bbold10}{}
\DeclareMathAlphabet{\matrixdigit}{U}{reportbb}{m}{n}
\pdfmapfile{+../scaled-contact-article/fonts/blackboard/bbold.map}
\newcommand{\id}{\matrixdigit{1}}
\setlength{\parindent}''')
    content='\n\\newpage\n'.join(r'\section*{'+escape(title)+'}\n'+body.replace(r'\mathbf 1',r'\id') for title,body in pages)
    (HERE/'report.tex').write_text(preamble+content+'\n\\end{document}\n')
    htmlpages=[]
    for title,body in pages:
        # Keep HTML concise; full typed content and tables are in the PDF/CSVs.
        htmlpages.append('<li>'+html.escape(title)+'</li>')
    encoded=json.dumps(points)
    document='''<!doctype html><meta charset="utf-8"><title>Experimental comparison</title>
<style>body{font:16px system-ui;max-width:1000px;margin:32px auto;padding:0 20px;color:#19232f}a{color:#245f9f}svg{width:100%;border:1px solid #ccd5df}table{border-collapse:collapse}td,th{padding:8px;border-bottom:1px solid #dde2e8;text-align:left}#tip{min-height:35px}</style>
<h1>Experimental comparison and deformation assessment</h1><p>7 October 2026</p>
<p><strong>The model is not yet validated across real-life datasets.</strong> Documented coefficients are fixed; estimation and held-out prediction are separately labeled. All 107 collision records and 17 rolling measurements are included in the PDF and data files.</p>
<p><a href="report.pdf" download>Download full PDF report</a> · <a href="evidence-v3/results.json" download>Complete evidence JSON</a> · <a href="evidence-v3/glass.csv" download>Glass CSV</a> · <a href="evidence-v3/balls.csv" download>Balls CSV</a> · <a href="evidence-v3/rocks.csv" download>Rocks CSV</a></p>
<table><tr><th>Test</th><th>Result</th></tr><tr><td>Glass velocities</td><td>Normal RMSE 0.0367; tangent RMSE 0.0178 m/s</td></tr><tr><td>Ball held-out signed moment</td><td>Spin RMSE worsens 1.048 → 1.243 rad/m; rejected</td></tr><tr><td>Rock held-out proxy</td><td>Angular-speed RMSE 12.88 rad/s; actual shape absent</td></tr><tr><td>Rolling constants</td><td>Positive estimates; fixed speed-dependent source curves predict better</td></tr></table>
<h2>Outgoing-state deviation map</h2><p>Perfect agreement is at (0,0). Glass dots compare translation; ball dots also include radius-scaled spin and use a held-out moment hypothesis. This is a signature angle, not a heading. No pooled score is used across different signatures.</p><svg id="plot" viewBox="0 0 900 450"></svg><p id="tip">Hover over a dot to identify its case.</p>
<h2>Deformation model to investigate</h2><p>Resolve normal indentation, tangential shear and asymmetric pressure/rocking with passive internal modes and a distributed contact patch. Derive the independent angular impulse from the traction moment. Validate force reversals and vibration decay before outgoing spin. This is a supported next model, not an already demonstrated accuracy improvement.</p>
<h2>PDF contents</h2><ol>'''+''.join(htmlpages)+'''</ol>
<script>const pts='''+encoded+''';const svg=document.querySelector('#plot');const ns='http://www.w3.org/2000/svg';function el(tag,attrs,text){let e=document.createElementNS(ns,tag);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));if(text)e.textContent=text;svg.append(e);return e}let xmax=Math.max(...pts.map(p=>p.signature_angle_error_deg))*1.1;let ymin=Math.min(-1,...pts.map(p=>p.signature_size_error_percent))*1.2,ymax=Math.max(1,...pts.map(p=>p.signature_size_error_percent))*1.2;const X=x=>70+760*x/xmax,Y=y=>360-290*(y-ymin)/(ymax-ymin);el('line',{x1:70,y1:Y(0),x2:830,y2:Y(0),stroke:'#aaa'});el('line',{x1:70,y1:70,x2:70,y2:360,stroke:'#aaa'});for(let i=0;i<=5;i++){let x=xmax*i/5,y=ymin+(ymax-ymin)*i/5;el('text',{x:X(x),y:380,'text-anchor':'middle','font-size':12},x.toFixed(1));el('text',{x:60,y:Y(y)+4,'text-anchor':'end','font-size':12},y.toFixed(1));}el('text',{x:450,y:418,'text-anchor':'middle'},'Observable signature angle error (degrees)');el('text',{x:72,y:35},'Relative signature size error (%)');pts.forEach(p=>{let c=el('circle',{cx:X(p.signature_angle_error_deg),cy:Y(p.signature_size_error_percent),r:5,fill:p.color});c.addEventListener('mouseenter',()=>document.querySelector('#tip').textContent=p.label+' | '+p.group+' | angle '+p.signature_angle_error_deg.toFixed(3)+'° | size '+p.signature_size_error_percent.toFixed(3)+'%')});let groups=[...new Map(pts.map(p=>[p.group,p.color]))];groups.forEach(([g,c],i)=>{el('circle',{cx:380,cy:25+i*20,r:4,fill:c});el('text',{x:393,y:29+i*20,'font-size':12},g)});</script>'''
    (HERE/'report.html').write_text(document)
    (HERE/'report-build.json').write_text(json.dumps(dict(authoritative_evidence=str(args.evidence.resolve().relative_to(HERE)),sections=len(pages),independent_audit_pass=True,collision_records=107,rolling_points=17,model_reproduces_all_experiments=False),indent=2)+'\n')
    print(f'Rendered {len(pages)} report sections with all row tables')


if __name__=='__main__':main()
