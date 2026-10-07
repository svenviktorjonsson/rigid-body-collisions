"""Typeset verified elastic-contact evidence and visible bounce previews."""
import io
import json
from pathlib import Path
import subprocess
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
ROOT=Path(__file__).resolve().parents[1];DIRECTORY=ROOT/'research/elastic-completion'
plt.rcParams.update({'axes.spines.top':False,'axes.spines.right':False,'font.size':10})

def main():
 summary=json.loads((DIRECTORY/'summary.json').read_text());audit=json.loads((DIRECTORY/'independent-audit.json').read_text());plan=json.loads((DIRECTORY/'plan.json').read_text());assert audit['qualified_cases']==18 and audit['qualified_original_cases']==10
 source=summary['execution_source_commit']
 with zipfile.ZipFile(DIRECTORY/'traces.zip') as archive:
  def trace(name):
   with np.load(io.BytesIO(archive.read(name+'/fine.npz'))) as saved:return {k:saved[k] for k in saved.files}
  normal=trace('elastic-normal-axis-spin');weighted=trace('compression-weighted-reversal');limited=trace('weighted-high-spin-low-friction-budget');chain=trace('vertical-floor-ceiling');floor=trace('same-floor-oblique-mixed-positive');negative=trace('elastic-normal-axis-spin-negative-twist')
 fig,axes=plt.subplots(2,2,figsize=(11,7.4));fig.suptitle('Elastic spin reversal and energy return: verified sphere–plane material',fontsize=14)
 axes[0,0].plot(normal['times']*1000,normal['states'][:,8],label='Initial spin +10');axes[0,0].plot(negative['times']*1000,negative['states'][:,8],label='Initial spin −10');axes[0,0].axhline(0,color='gray',lw=.6);axes[0,0].set(xlabel='Time (ms)',ylabel='Normal-axis spin (rad/s)',title='An independent twisting couple reverses spin');axes[0,0].legend()
 for key,label in [('kinetic_J','Kinetic'),('normal_stored_J','Normal spring'),('twist_stored_J','Twisting spring')]:
  value=normal[key].sum(axis=1) if normal[key].ndim==2 else normal[key];axes[0,1].plot(normal['times']*1000,value,label=label)
 axes[0,1].plot(normal['times']*1000,normal['kinetic_J']+normal['stored_J']+normal['dissipated_J'],'k--',label='Accounted total');axes[0,1].set(xlabel='Time (ms)',ylabel='Energy (J)',title='Stored energy returns at separation');axes[0,1].legend()
 axes[1,0].plot(limited['times']*1000,limited['states'][:,8],color='#be6242',label='Spin');axes[1,0].axhline(0,color='gray',lw=.6);axes[1,0].set(xlabel='Time (ms)',ylabel='Spin (rad/s)',title='Low capacity: +10 → +5.20433; no reversal')
 axes[1,1].plot(limited['times']*1000,limited['kinetic_J'],label='Kinetic');axes[1,1].plot(limited['times']*1000,limited['stored_J'],label='Stored');axes[1,1].plot(limited['times']*1000,limited['dissipated_J'],label='Dissipated');axes[1,1].plot(limited['times']*1000,limited['kinetic_J']+limited['stored_J']+limited['dissipated_J'],'k--',label='Accounted total');axes[1,1].set(xlabel='Time (ms)',ylabel='Energy (J)',title='Original high-spin case: 4,233 of 20,000 evaluations');axes[1,1].legend()
 fig.tight_layout(rect=(0,0,1,.95));fig.savefig(DIRECTORY/'spin-energy.png',dpi=170);plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(11,7.4));fig.suptitle('Continuous chained trajectories: no resetting spin or velocity between impacts',fontsize=14)
 axes[0,0].plot(chain['times'],chain['states'][:,2]);axes[0,0].axhline(.1,color='gray',ls='--');axes[0,0].axhline(.13,color='gray',ls='--');axes[0,0].set(xlabel='Time (s)',ylabel='Centre height (m)',title='Five alternating floor / ceiling lift-offs')
 axes[0,1].plot(chain['times'],chain['states'][:,8]);axes[0,1].axhline(0,color='gray',lw=.6);axes[0,1].set(xlabel='Time (s)',ylabel='Spin (rad/s)',title='Normal-axis spin changes sign on each bounce')
 axes[1,0].plot(floor['states'][:,0],floor['states'][:,2]);axes[1,0].set(xlabel='Horizontal position (m)',ylabel='Centre height (m)',title='Three same-floor oblique bounces under gravity')
 axes[1,1].plot(floor['times'],floor['states'][:,3],label='Horizontal velocity (m/s)');axes[1,1].plot(floor['times'],floor['states'][:,7]/25,label='Tangent-axis spin / 25');axes[1,1].plot(floor['times'],floor['states'][:,8]/50,label='Normal-axis spin / 50');axes[1,1].axhline(0,color='gray',lw=.6);axes[1,1].set(xlabel='Time (s)',ylabel='Displayed scaled components',title='COM motion and both spin components alternate');axes[1,1].legend()
 fig.tight_layout(rect=(0,0,1,.95));fig.savefig(DIRECTORY/'bounces.png',dpi=170);plt.close(fig)
 # Keep the real sphere radius and geometry. Top view represents normal-axis
 # spin; the height panel separately represents vertical translation.
 for name,data,ceiling in [('floor-ceiling-preview',chain,True),('same-floor-preview',floor,False)]:
  fig,axes=plt.subplots(1,2,figsize=(8,4));ax,top=axes;ax.set_aspect('equal');ax.set(xlabel='Horizontal position (m)',ylabel='Height (m)');ax.axhline(0,color='#546270',lw=2)
  xmin=float(np.min(data['states'][:,0]))-.12;xmax=float(np.max(data['states'][:,0]))+.12;ax.set_xlim(xmin,xmax);ax.set_ylim(-.02,.27)
  if ceiling:ax.axhline(.23,color='#546270',lw=2)
  ball=plt.Circle((0,.11),.1,facecolor='#4c91b6',alpha=.7,edgecolor='#24536d');ax.add_patch(ball);path,=ax.plot([],[],color='#669fb8',lw=1);label=ax.text(.02,.98,'',transform=ax.transAxes,va='top');top.set_aspect('equal');top.set_xlim(-.13,.13);top.set_ylim(-.13,.13);top.axis('off');top.set_title('Normal-axis spin indicator');top.add_patch(plt.Circle((0,0),.1,facecolor='#d8eaf2',edgecolor='#24536d'));marker,=top.plot([],[],color='#c06445',lw=3)
  times=data['times'];phase=np.r_[0,np.cumsum(.5*(data['states'][1:,8]+data['states'][:-1,8])*np.diff(times))];indices=np.linspace(0,len(times)-1,80).round().astype(int)
  def update(frame):
   i=indices[frame];ball.center=data['states'][i,:2][0],data['states'][i,2];path.set_data(data['states'][:i+1,0],data['states'][:i+1,2]);marker.set_data([0,.095*np.cos(phase[i])],[0,.095*np.sin(phase[i])]);label.set_text(f"t = {times[i]:.3f} s\nspin = {data['states'][i,8]:+.2f} rad/s");return ball,path,marker,label
  animation=FuncAnimation(fig,update,frames=len(indices),interval=50,blit=True);animation.save(DIRECTORY/(name+'.gif'),writer=PillowWriter(fps=20));plt.close(fig)
 rows=[]
 for name in ['elastic-normal-axis-spin','elastic-tangent-axis-spin','low-friction-no-spin-reversal','compression-weighted-reversal','weighted-high-spin-low-friction-budget','same-floor-oblique-mixed-positive']:
  row=next(c for c in summary['cases'] if c['name']==name);m=row['runs'][-1]['metrics'];omega=m['final_omega_rad_s'];rows.append(f"{name.replace('-',' ')} & {omega[1]:.5g}, {omega[2]:.6g} & {m['max_energy_residual_J']:.2g} & {m['rhs_evaluations']} \\\\")
 tex=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=18mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref}
\hypersetup{hidelinks}
\title{Elastic contact forces and independent angular impulses\\Completed sphere--plane verification}
\author{Rigid Body Collisions research prototype}\date{5 October 2026}
\begin{document}\emergencystretch=2em\maketitle
All ten original cases now satisfy their unchanged gates. The complete predeclared study verifies 18 of 18 cases over 54 histories, with no rejected attempts. Earlier failed studies remain archived. These are synthetic constitutive hypotheses and mathematical/numerical verification cases: no measured rubber calibration, new friction law, or arbitrary-body native coupling is claimed. Frozen source: \texttt{SOURCEPIN}.
\section*{A generalized impulse includes an independent couple}
For a solid sphere, $\mathbb I=2mR^2/5$. With a fixed-plane normal $\hat n$ directed into the admissible half-space, use the reference lever $r=-R\hat n$. The cross-product matrix and contact velocity are
\[
r\wedge=\begin{bmatrix}0&-r_z&r_y\\r_z&0&-r_x\\-r_y&r_x&0\end{bmatrix},\qquad
v_c=v+(r\wedge)^\mathsf T\omega.
\]
The independent angular impulse $\Delta L$ is separate from the angular momentum generated by a point force:
\[
m\Delta v=\Delta p+mg\,\Delta t,\qquad
\mathbb I\Delta\omega=(r\wedge)\Delta p+\Delta L.
\]
For an instantaneous fixed-lever impulse without gravity, define
\[
V=\begin{bmatrix}v_c\\\omega\end{bmatrix},\quad
\Delta P=\begin{bmatrix}\Delta p\\\Delta L\end{bmatrix},\quad
\Delta V=M\Delta P,\quad
M=\begin{bmatrix}1/m-(r\wedge)\mathbb I^{-1}(r\wedge)&-(r\wedge)\mathbb I^{-1}\\
\mathbb I^{-1}(r\wedge)&\mathbb I^{-1}\end{bmatrix}.
\]
Scalar matrix additions implicitly multiply the identity. The mixed-unit vector is bookkeeping; $V^\mathsf T\Delta P$ still has energy units. The kinetic change is
\[
\Delta E_k=(V^-)^\mathsf T\Delta P+\tfrac12\Delta P^\mathsf T M\Delta P.
\]
A normal-axis spin change requires $\Delta L$: $(r\wedge)\Delta p$ has zero component along $\hat n$.
\section*{One potential, a shared yield budget, and a complete energy ledger}
Let $h=(q_1,q_2,a\theta)^\mathsf T$, $K=\operatorname{diag}(k_t,k_t,k_\theta/a^2)$, $f=(\delta/R)^p$, and $\delta=\max(0,R-\hat n\cdot x+d)$. Here $a$ is an effective contact length, with no required meshed patch. The two supported exponents are $p=2$ for the weighted material and $p=0$ for the linear benchmark:
\[
U=\tfrac12k_n\delta^2+U_h,\quad U_h=\tfrac12f h^\mathsf TKh,\quad
F_n=k_n\delta+c_n\max(0,\dot\delta)+pU_h/\delta.
\]
\[
F_t=-fk_tq,\quad\tau_n=-fk_\theta\theta,\quad
\sqrt{\|F_t\|^2+(\tau_n/a)^2}\leq\mu k_n\delta\leq\mu F_n.
\]
The shared ellipsoid is a phenomenological capacity approximation, not an exact traction surface for every pressure distribution. One coefficient covers stick and slide. Let $e=fKh$, $s=e/\|e\|$, $u=(T^\mathsf Tv_c,a\hat n\cdot\omega)^\mathsf T$. Associated plastic flow removes outward yield loading:
\[
\dot h=u-\lambda s,\quad
\lambda=\max\left(0,\frac{s\cdot[p(\dot\delta/\delta)e+fKu]-\mu k_n\dot\delta}{f\,s^\mathsf TKs}\right),\quad
\dot D=c_n\max(0,\dot\delta)\dot\delta+\lambda\|e\|\geq0.
\]
Consequently, including gravity,
\[
\frac{d}{dt}\left(\tfrac12m\|v\|^2+\tfrac12\mathbb I\|\omega\|^2-mg\cdot x+U+D\right)=0.
\]
\newpage
\section*{Numerical repair and material-preserving effort selection}
The earlier rejected high-spin, low-friction case exhausted 20,000 RHS evaluations when explicit integration repeatedly switched yield branches at individual stages. The revised solver follows elastic or yielded modes between exact yield, release and lift-off events. Material parameters, the force law and the budget are unchanged. Exact ballistic free motion skips contact evaluations between impacts. A grazing contact with no inward crossing creates no spurious zero-time transitions.

The high-spin weighted case now needs 4,233 evaluations at the finest level: $\Omega_z:10\to5.20433335$ rad/s, $v_z:-1\to1.00371591$ m/s, $D=0.14210701$ J, complete energy residual $1.59\times10^{-14}$ J. Low friction correctly does not reverse this spin. Its negative-spin counterpart has the reflected result.

For $p=0$, zero damping, no prior history, zero gravity during contact and sufficiently large shared capacity, a conditional exact branch uses matched frequencies:
\[
\omega_c=\sqrt{k_n/m},\quad m_t=(1/m+R^2/\mathbb I)^{-1},\quad k_t=m_t\omega_c^2,\quad k_\theta=\mathbb I\omega_c^2.
\]
With incoming normal speed $V_n>0$, tangent contact velocity $u_t$ and normal spin $\Omega_n$, sticking is proved throughout the half-period when
\[
\sqrt{(m_t\|u_t\|)^2+(\mathbb I\Omega_n/a)^2}\leq\mu mV_n.
\]
The exact impulse and duration are
\[
\Delta p=2mV_n\hat n-2m_tu_t,\quad\Delta L=-2\mathbb I\Omega_n\hat n,\quad t_c=\pi/\omega_c.
\]
Otherwise the dispatcher resolves the \emph{same configured material} with history. It never replaces elastic return by an inelastic Coulomb law for speed. Sphere/plane scope, deformation budget and total energy gates apply to both branches.
\par\noindent\includegraphics[width=\textwidth]{spin-energy.png}
\newpage
\section*{Continuous bounce sequences and independent evidence}
For the matched elastic contact, a vertical floor--ceiling path undergoes five lift-offs in 0.18 s. Both twist signs reverse at every impact, without resetting the body state. A separate oblique same-floor trajectory under gravity has three bounces: horizontal COM motion, tangent-axis spin and normal-axis spin all alternate. It retains the original $k_n=10^8$ N/m material. Gravity perturbs the ideal contact period, producing a brief resolved yield tail before lift-off. Over three bounces its plastic loss is $2.095\times10^{-6}$ J; its separation-store removal is only $4.89\times10^{-18}$ J. The complete energy residual remains $1.17\times10^{-12}$ J.

An arbitrary oblique floor--ceiling path need not retrace: reversing the plane normal changes tangential force--spin coupling. The verified chained ceiling trajectory is vertical with normal-axis twist; the verified oblique back-and-forth trajectory repeatedly meets the same floor. Additional rolling-couple mechanics would require separate justification and validation.
\par\noindent\includegraphics[width=.9\textwidth]{bounces.png}
\begin{center}\footnotesize\begin{tabular}{p{62mm}rrr}\toprule
Case & $\Omega_y,\Omega_z$ (rad/s) & Energy error (J) & RHS\\\midrule
TABLEROWS
\bottomrule\end{tabular}\end{center}
The independent audit recomputes sample and accepted-step energy, linear momentum, offset-force versus independent-couple angular momentum, shared capacity, both adjacent refinement edges and every bounce criterion. Dispatchers are replayed from archived source in a clean process. Same-material slow/rapid examples cover 0.01 and 100 m/s in both twist directions. These are checks of an ideal material, not evidence that actual rubber is rate independent or accurate at those speeds. Force peaks are numerical accepted-step/sample diagnostics, not certified continuous suprema. The native arbitrary-shape engine does not yet integrate these independent elastic couples.
\section*{Related mechanics and calibration limits}
\begingroup\small\sloppy
Rod Cross, ``Grip-slip behavior of a bouncing ball'', \emph{American Journal of Physics} 73 (2005), DOI: \href{https://doi.org/10.1119/1.2008299}{10.1119/1.2008299}, demonstrates that friction magnitude alone does not determine elastic spin return. Xydas and Kao, \emph{IJRR} 18 (1999), DOI: \href{https://doi.org/10.1177/02783649922066673}{10.1177/02783649922066673}, discuss soft-contact force/moment capacity. The accompanying retained mechanics review provides qualifications and measured counterexamples. No novelty or publication claim follows solely from the verified examples.
\endgroup
\end{document}
'''
 tex=tex.replace('SOURCEPIN',source[:12]).replace('TABLEROWS','\n'.join(rows));(DIRECTORY/'report.tex').write_text(tex)
 subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','report.tex'],cwd=DIRECTORY,check=True,stdout=subprocess.DEVNULL)
 high=next(c for c in summary['cases'] if c['name']=='weighted-high-spin-low-friction-budget')['runs'][-1]['metrics'];markdown=f'''# Completed elastic force-and-couple verification

**10/10 original cases and 18/18 total cases qualify**, with 54 completed histories and no rejected attempts. The frozen source is `{source}`. Original material parameters, gates, high-spin 20,000-evaluation budget and earlier archives remain unchanged.

Hybrid yield/release events eliminate branch chattering; exact ballistic free flight preserves the same mechanics between contacts. The earlier high-spin low-friction weighted case now completes in {high['rhs_evaluations']} evaluations, with spin +10 → +5.20433335 rad/s, 0.14210701 J dissipated and energy residual 1.59e-14 J. Low friction correctly prevents reversal. Both twist signs are verified.

The continuous vertical floor/ceiling sequence has five alternating impacts. The oblique same-floor gravity sequence has three back-and-forth bounces with horizontal motion and both spin components alternating. Its final residual contact store is released through the resolved yield tail; explicit separation loss is below 5e-18 J. Arbitrary oblique floor/ceiling retracing is not asserted.

![Elastic spin and energy](spin-energy.png)

![Continuous bounce sequences](bounces.png)

![Floor/ceiling preview](floor-ceiling-preview.gif)

![Same-floor preview](same-floor-preview.gif)

[Typeset mechanics, evidence and limits](report.pdf). [Independent audit](independent-audit.json).

This is a fixed-sphere/fixed-plane material prototype. Conditional exact and resolved branches use the same material and agree with the refined histories. It does not yet supply independent elastic couples to the native arbitrary-body engine. Parameters are synthetic rubber-like hypotheses, not measured rubber calibration; neither constitutive novelty nor material authenticity follows from numerical convergence. Costs were collected during collaborative work and do not establish a speed ranking.
''';(DIRECTORY/'report.md').write_text(markdown)
 print('Rendered completed paper, spin/energy and bounce figures, and two previews')
if __name__=='__main__':main()
