"""Render archived elastic-contact evidence; no simulations or parameter fits."""
import io
import json
from pathlib import Path
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DIR=Path('research/elastic-patch')


def main():
    global DIR
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--directory',default=str(DIR))
    DIR=Path(parser.parse_args().directory)
    summary=json.loads((DIR/'summary.json').read_text()); cases={c['name']:c for c in summary['cases']}
    with zipfile.ZipFile(DIR/'traces.zip') as archive:
        def load(name):
            path=name+'/fine.npz'
            if path in archive.namelist():
                contents=archive.read(path)
            else:
                with zipfile.ZipFile('research/elastic-patch/traces.zip') as original:
                    contents=original.read(path)
            with np.load(io.BytesIO(contents)) as data: return {k:data[k] for k in data.files}
        spin=load('elastic-normal-axis-spin'); tangent=load('elastic-tangent-axis-spin')
        weighted=load('compression-weighted-reversal'); small=load('low-friction-no-spin-reversal')
        floor=load('oblique-floor-retrace'); ceiling=load('oblique-ceiling-retrace'); both=load('vertical-floor-ceiling')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(11,7),constrained_layout=True)
    ax=axes[0,0]
    for data,label in [(spin,'elastic torsion, μ=3'),(small,'yielding torsion, μ=0.1')]:
        ax.plot(data['times']*1000,data['states'][:,8],label=label)
    ax.axhline(0,color='.5',lw=.7); ax.set(xlabel='time (ms)',ylabel='normal-axis spin (rad/s)',title='Independent torque impulse reverses spin'); ax.legend()
    ax=axes[0,1]
    for key,label in [('kinetic_J','kinetic'),('normal_stored_J','normal spring'),('twist_stored_J','twisting spring')]:
        values=spin[key]; ax.plot(spin['times']*1000,np.sum(values,axis=1) if values.ndim==2 else values,label=label)
    ax.plot(spin['times']*1000,spin['kinetic_J']+spin['stored_J']+spin['dissipated_J'],ls='--',color='black',label='accounted total')
    ax.set(xlabel='time (ms)',ylabel='energy (J)',title='Energy is stored and released'); ax.legend()
    ax=axes[1,0]
    ax.plot(tangent['times']*1000,tangent['states'][:,7],label='tangent-axis spin (rad/s)')
    ax.plot(tangent['times']*1000,10*tangent['states'][:,3],label='10 × horizontal speed (m/s)')
    ax.axhline(0,color='.5',lw=.7); ax.set(xlabel='time (ms)',title='Offset point force couples spin and translation'); ax.legend()
    ax=axes[1,1]
    ax.plot(weighted['times']*1000,weighted['states'][:,8],label='normal-axis spin (rad/s)')
    ax2=ax.twinx();ax2.plot(weighted['times']*1000,weighted['twist_stored_J'][:,0],color='#c05621',label='twisting store')
    ax2.set_ylabel('twisting store (J)',color='#c05621')
    ax.axhline(0,color='.5',lw=.7); ax.set(xlabel='time (ms)',ylabel='normal-axis spin (rad/s)',title='Compression-weighted contact, μ=1'); ax.legend(loc='lower left')
    fig.savefig(DIR/'spin-energy.png',dpi=180); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    ax=axes[0]
    ax.plot(floor['states'][:,0]*1000,floor['states'][:,2]*1000,label='floor: chosen incoming spin')
    ax.plot(ceiling['states'][:,0]*1000,ceiling['states'][:,2]*1000,label='ceiling: opposite incoming spin')
    ax.axhline(100,color='.5',ls='--',lw=.8);ax.axhline(200,color='.5',ls='--',lw=.8)
    ax.set(xlabel='horizontal COM (mm)',ylabel='vertical COM (mm)',title='Conditional oblique retracing; centre contact levels');ax.legend()
    ax=axes[1]
    ax.plot(both['times']*1000,both['states'][:,2]*1000,label='COM height (mm)')
    ax.set(xlabel='time (ms)',ylabel='COM height (mm)',title='Repeated vertical floor–ceiling path')
    ax2=ax.twinx();ax2.plot(both['times']*1000,both['states'][:,8],color='#c05621',label='spin');ax2.set_ylabel('normal-axis spin (rad/s)',color='#c05621')
    fig.savefig(DIR/'retrace.png',dpi=180);plt.close(fig)
    lines=['# Elastic force-and-couple contact verification','',
           f'Frozen source: `{summary["execution_source_commit"]}`. **{summary["qualified_cases"]}/{len(summary["cases"])} cases meet the frozen gates**; {summary["completed_histories"]} completed histories and {summary["rejected_attempts"]} retained rejected attempts.','',
           'The sphere has an independent twisting couple, in addition to the angular momentum transferred by an offset contact force. Elastic contact history stores energy and can reverse spin; Coulomb dissipation alone does not supply that elastic return. These are synthetic material hypotheses and mathematical verification cases, not a fitted or experimentally authenticated rubber model.','',
           '| Case | Qualified | Fine outgoing velocity (m/s) | Fine outgoing spin (rad/s) | Energy residual (J) |','|---|---:|---|---|---:|']
    for row in summary['cases']:
        fine=row['runs'][-1]
        if fine['status']=='completed':
            m=fine['metrics'];v=', '.join(f'{x:.6g}' for x in m['final_velocity_m_s']);w=', '.join(f'{x:.6g}' for x in m['final_omega_rad_s']);err=f'{m["max_energy_residual_J"]:.3g}'
        else:v=w='rejected';err='—'
        lines.append(f'| {row["name"]} | {"yes" if row["qualified"] else "no"} | {v} | {w} | {err} |')
    lines+=['','![Spin and elastic energy](spin-energy.png)','',
            'The matched linear normal/torsional oscillators use one contact duration, return both spring stores to zero, reverse 10 rad/s to −10 rad/s, and preserve total energy. The coupled tangential oscillator reverses horizontal-axis spin from 10 to −30/7 rad/s while generating COM speed 4/7 m/s. The low-friction case ends at +9 rad/s with 0.038 J dissipated, showing that sufficiently strong elastic contact and the yield budget both matter.','',
            'The default compression-weighted potential includes its normal derivative, so energy can transfer between normal and rotational channels. An individual channel can have an effective restitution larger than one while the complete system remains passive. The shared ellipsoidal shear/couple yield is a phenomenological soft-finger approximation; it is not an exact traction limit surface for every pressure distribution. One friction coefficient covers sticking and sliding. Normal damping acts during compression only.','',
            '![Conditional retracing and repeated bounces](retrace.png)','',
            'Oblique retracing requires the chosen spin and elastic stiffness; it is not universal. Floor and ceiling oblique examples use opposite incoming tangent-axis spins. The repeated floor–ceiling example follows a vertical path with normal-axis spin reversing on each impact. All finite contact compression uses a reference lever arm equal to the undeformed sphere radius.','',
            'Slow and rapid tests use the **same** stiffness, mass, radius and friction parameters, with speeds 0.01 and 100 m/s. This verifies the integration strategy over a broad rate range for this ideal elastic law; it does not demonstrate rate-independent real rubber. The high-spin low-friction weighted case has a frozen RHS evaluation budget; exhaustion is retained as rejection, with no accepted state fabricated.','',
            'Timings were collected while native friction research ran concurrently. They describe these executions and are not an engine speed ranking. Reference refinement uses three DOP853 settings with frozen state, energy and yield budgets. Stored ZIP traces and source files permit independent energy, momentum and impulse bookkeeping. The reported internal-step yield peak is numerical, rather than a continuous mathematical supremum.','',
            '[Typeset mechanics and evidence](report.pdf). [Independent mechanics and literature review](../elastic-patch/review.md).']
    if DIR.name=='elastic-patch-refined':
        lines[4:4]=['This follow-up changes numerical tolerances only and targets the six original failures. The original [4/10 study and two rejected attempts](../elastic-patch/report.md) remain unchanged. Five additional cases now qualify, making **9/10 distinct examples verified across both studies**; weighted high-spin/low-friction remains rejected. Tangential and oblique figures reuse the previously qualified original traces; other figures use refined traces.','']
    (DIR/'report.md').write_text('\n'.join(lines)+'\n')
    rows=[]
    for row in summary['cases']:
        fine=row['runs'][-1]
        if fine['status']=='completed':
            m=fine['metrics']; spin_text=', '.join(f'{v:.5g}' for v in m['final_omega_rad_s']); error=f'{m["max_energy_residual_J"]:.2g}'
        else: spin_text='rejected';error='--'
        name=row['name'].replace('-',' ')
        rows.append(name+' & '+('yes' if row['qualified'] else 'no')+' & '+spin_text+' & '+error+r' \\')
    tex=r'''\documentclass[10pt]{article}
\usepackage[a4paper,margin=19mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,booktabs,hyperref}
\title{Elastic contact forces and independent angular couples\\Verified sphere--plane examples}
\author{Rigid Body Collisions research prototype}
\date{5 October 2026}
\begin{document}\maketitle
The model is a synthetic constitutive hypothesis, not a calibrated rubber law or a new friction law. It verifies an independent collision couple, explicit elastic storage, yielding and conditional reversal. Frozen source: \texttt{SOURCE}. QUALIFIED of TOTAL cases satisfy the registered gates; HIST histories and REJECT rejected attempts are retained. Numerical timings were collected concurrently with native friction research and cannot establish a general speed ranking.
\section*{A point-like wrench includes a couple}
For a sphere of mass $m$, radius $R$ and inertia $\mathbb I=\frac25mR^2$, a fixed plane has normal $\hat n$ directed into the admissible half-space. The reference lever is $r=-R\hat n$; compression is $\delta=\max(0,R-\hat n\cdot x+d)$.
\[
r\wedge=\begin{bmatrix}0&-r_z&r_y\\r_z&0&-r_x\\-r_y&r_x&0\end{bmatrix},\qquad
v_c=v+(r\wedge)^\mathsf{T}\omega.
\]
\[
\Delta p=\int F\,dt,\quad \Delta L=\int \tau_n\hat n\,dt,\qquad
m\Delta v=\Delta p+\int mg\,dt,\quad
\mathbb I\Delta\omega=(r\wedge)\Delta p+\Delta L.
\]
The normal-axis spin change requires the independent couple $\Delta L$: the offset point force has zero torque component along $\hat n$. A point-like wrench is a representation of effective contact mechanics, not a claim that a mathematical point traction carries a moment.
\section*{Elastic history, shared yield and energy}
Let $q$ be the two-component tangential displacement, $\theta$ the relative twist angle and $a$ an effective material contact length. Define $h=(q_1,q_2,a\theta)^\mathsf{T}$, $K=\operatorname{diag}(k_t,k_t,k_\theta/a^2)$ and $f=(\delta/R)^p$, with $p=2$ for the default compression-weighted contact.
\[
U=\tfrac12k_n\delta^2+\tfrac12 f h^\mathsf TK h,\quad U_h=\tfrac12 f h^\mathsf TK h,\qquad
F_n=k_n\delta+c_n\max(0,\dot\delta)+pU_h/\delta.
\]
\[
F_t=-f k_t q,\qquad \tau_n=-f k_\theta\theta,\qquad
\sqrt{\lVert F_t\rVert^2+(\tau_n/a)^2}\leq\mu k_n\delta\leq\mu F_n.
\]
The last bound shares the normal-load budget between sliding and twisting. It is a phenomenological ellipsoidal capacity approximation, not an exact patch traction result. Using the elastic base normal load makes the budget conservative even when history or damping adds normal force. One coefficient $\mu$ covers sticking and sliding; this prototype does not yet fit distinct static and dynamic coefficients. Damping acts only during compression.

Let $T$ contain orthonormal plane tangents and $u=(T^\mathsf Tv_c,a\hat n\cdot\omega)^\mathsf T$. During elastic stick, $\dot h=u$. At the yield boundary, associated plastic flow has $\dot h=u-\lambda s$, with $s=fKh/\lVert fKh\rVert$ and $\lambda\geq0$ chosen to prevent outward motion of the yield function. Consequently
\[
\dot D=c_n\max(0,\dot\delta)\dot\delta+\lambda\lVert fKh\rVert\geq0,\qquad
\frac{d}{dt}\left(\tfrac12m\lVert v\rVert^2+\tfrac12\mathbb I\lVert\omega\rVert^2-mg\cdot x+U+D\right)=0.
\]
The default store vanishes smoothly at separation. The optional constant-stiffness benchmark uses $p=0$ and explicitly records any residual history store as separation loss; matched oscillator tests return it to zero. No elastic state is silently deleted. Adaptive integration rejects an exhausted RHS evaluation budget.
\section*{Independent exact oscillators}
For the linear benchmark $p=0$, no damping and no yielding, choose
\[
\omega_c=\sqrt{k_n/m},\quad t_c=\pi/\omega_c,\qquad
k_\theta=\mathbb I k_n/m,\quad k_t=\frac{\mathbb I k_n}{\mathbb I+mR^2}.
\]
The normal and twist modes then share a half period: $v_n^-=-V\to v_n^+=V$, $\Omega_n^+=-\Omega_n^-$. Throughout contact, $|\tau_n|/F_n=\mathbb I|\Omega_n^-|/(mV)$; the no-yield requirement is therefore $\mu a\geq\mathbb I|\Omega_n^-|/(mV)$, not merely an endpoint impulse inequality.

For a solid sphere dropped with zero horizontal COM speed and tangent-axis spin $\Omega$, the same elastic tangential oscillator gives
\[
\Omega_t^+=-\tfrac37\Omega_t^-,\qquad v_t^+=\tfrac47R\Omega_t^-.
\]
With $v_t^-=v_0$ and appropriately oriented $\Omega_t^-=-v_0/(\alpha R)$ at the floor, $\alpha=\mathbb I/(mR^2)=2/5$, both horizontal velocity and spin reverse. A ceiling needs the opposite incoming spin. Arbitrary rubber-ball throws do not necessarily retrace. A vertical floor--ceiling path with pure normal-axis spin can repeat these exact elastic reversals.
\newpage
\section*{Evidence and limitations}
\begin{center}\small\begin{tabular}{p{65mm}clc}\toprule
Case & Gates & Fine outgoing spin (rad/s) & Energy error (J)\\\midrule
ROWS
\bottomrule\end{tabular}\end{center}
\includegraphics[width=\textwidth]{spin-energy.png}
\includegraphics[width=\textwidth]{retrace.png}
\noindent
The weighted contact can transfer energy between normal and rotational channels, so a single-channel restitution may exceed one while total mechanics remain passive. Low friction can dissipate enough elastic input to prevent reversal. The same-material slow and rapid examples use speeds $0.01$ and $100$ m/s, demonstrating numerical behavior of this ideal law, not real rubber at those rates. All traces, failures, frozen parameters, source and three-level refinements are archived. This is a sphere/plane prototype; arbitrary-shape simultaneous contact needs separate engine validation. The accompanying literature review distinguishes known soft-finger mechanics from any unproven publication novelty.
\end{document}
'''
    tex=tex.replace('SOURCE',summary['execution_source_commit'][:12]).replace('QUALIFIED',str(summary['qualified_cases'])).replace('TOTAL',str(len(summary['cases']))).replace('HIST',str(summary['completed_histories'])).replace('REJECT',str(summary['rejected_attempts'])).replace('ROWS','\n'.join(rows))
    if DIR.name=='elastic-patch-refined':
        text=r'This follow-up tightens integration tolerances only, with unchanged physics and gate budgets. The original four qualified cases, all six failures and two rejected attempts remain archived. Five additional cases now qualify, making nine of ten distinct examples verified across the studies. Weighted high-spin/low-friction remains rejected. Figures retain previously qualified original tangential/oblique traces and use refined traces otherwise.'
        tex=tex.replace(r'\section*{A point-like wrench includes a couple}',text+'\n'+r'\section*{A point-like wrench includes a couple}')
    (DIR/'report.tex').write_text(tex)


if __name__=='__main__': main()
