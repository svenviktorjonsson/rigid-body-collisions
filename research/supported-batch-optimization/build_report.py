"""Build the integration summary from archived benchmark and test receipts."""
import json,subprocess
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P=Path(__file__).resolve().parent
if __name__=='__main__':
    out=P/'report';out.mkdir(exist_ok=True)
    results=json.loads((P/'run-v3/results.json').read_text())
    selected=[x for x in results['batches'] if x['responses']==1000000]
    fig,ax=plt.subplots(figsize=(8,3.6))
    x=[r['threads'] for r in selected];y=[r['candidate_median_ms'] for r in selected]
    ax.errorbar(x,y,yerr=[[r['candidate_median_ms']-r['candidate_min_ms'] for r in selected],[r['candidate_max_ms']-r['candidate_median_ms'] for r in selected]],fmt='o-',capsize=4,label='Candidate: median and observed range')
    ax.axhline(20,color='#b14a27',ls='--',label='20 ms target')
    ax.scatter([1],[selected[0]['baseline_median_ms']],marker='x',s=70,color='black',label='Matched frozen single-worker reference')
    ax.set_xticks(x);ax.set_xlabel('Requested CPU workers');ax.set_ylabel('One million independent updates (ms)')
    ax.set_ylim(bottom=0)
    ax.set_title('Field loads, all outputs and mechanics gates included')
    ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'cost.pdf');plt.close(fig)
    rows='\n'.join(f"{r['threads']} & {r['candidate_median_ms']:.3f} & {r['candidate_min_ms']:.3f}--{r['candidate_max_ms']:.3f} \\\\" for r in selected)
    text=r'''\documentclass[10pt,a4paper]{article}
\usepackage[margin=20mm]{geometry}
\usepackage{amsmath,amssymb,graphicx,hyperref}
\hypersetup{colorlinks=true,urlcolor=blue,linkcolor=blue}
\setlength{\parindent}{0pt}\setlength{\parskip}{7pt}\emergencystretch=2em
\begin{document}
\begin{center}{\Large Supported-contact compiler integration package}\\[5pt]
{\large Verified native interface, portable algorithm and explicit limits}\\7 October 2026\end{center}

\textbf{Result.} A versioned C-compatible batch boundary and indexed implementation
are prepared for a scoped BKF / Vektor Flow import. One million independent
supported-contact responses take \textbf{12.830 ms median with eight workers},
range 12.417--14.606 ms. Every one of seven samples meets the 20 ms target.
54 focused tests pass, including native/reference, worker determinism, wide-size
controls and a C11 header probe. Compiler repositories are unchanged.

\includegraphics[width=\linewidth]{cost.pdf}

\begin{tabular}{rrr}\hline Requested workers & Median ms & Observed range ms\\\hline
@@ROWS@@
\hline\end{tabular}

\textbf{Fair comparison.} Matched-layout single-worker reference: 46.463 ms;
candidate: 50.975 ms. There is no demonstrated single-worker algorithmic speedup.
Four workers miss the controlled median target. Earlier 81.4 ms used a different
harness and is not the paired reference. All attempts remain archived.

\textbf{Measured scope.} Field-major loads, all eleven stores, worker entry and
finite/branch/energy gates are included. Preparation, input validation, allocation,
detection, changing loads/frames, impacts, group solving and contact-history updates
are excluded. Repeated independent fixtures are not a million interacting bodies.
No all-case 2$\times$ gate, compiler/GPU acceptance or experimental gain is claimed.
The measured host is AMD Ryzen 7 3700X, Linux x86-64; the C++17 build uses OpenMP,
\texttt{-O3 -Wall -Wextra -Werror}, without fast-math.
\newpage
\section*{Algorithmic correctness before compiler tuning}
The supported sphere/disk has scalar central inertia $I$, radius $R$, translation
$v$, rolling speed $\omega$, force $f$ and independent rolling moment $M_r$.
Use rim speed $q=R\omega$ and force-equivalent couple $b=M_r/R$:
\[
\begin{bmatrix}\dot u\\\dot q\end{bmatrix}=
\begin{bmatrix}F_d/m\\0\end{bmatrix}+
\begin{bmatrix}1/m+R^2/I&-R^2/I\\-R^2/I&R^2/I\end{bmatrix}
\begin{bmatrix}f\\b\end{bmatrix},\qquad u=v-q.
\]
Both mobility rows now have consistent linear units. Reciprocals and capacities
are computed once per update. Continued nonzero sliding/rolling has known signs
and skips active-set enumeration. Zero motion uses explicit static/onset candidates;
events stop at slip/rolling arrest before constraints are recomputed.

\textbf{Fixed defects.} The old branch tolerances mixed linear and angular units:
a valid frictionless $10^{-8}$ N drive could be rejected for small bodies. The
corrected Python/native paths work for radii $10^{-12}$--$10^6$ m. Axial spin is
set to exact zero at its impulse bound, preventing roundoff reversal. Nonfinite
energy/response states are rejected, including static impulse overflow without work.
No material value or restitution coefficient is fitted or changed.

\textbf{Impulse identity.} Independent angular impulse remains separate from the
linear-impulse lever moment. The scalar branch satisfies
\[
I(\omega^+-\omega^-)=-R\,\delta p_t+\delta L.
\]
The output rolling moment channel is a physical independent impulse, not a
replacement for the lever term or a new definition of the full spin direction.
The user's full contact-relative $\hat{\boldsymbol t}$ and full angular
$\hat{\boldsymbol s}$ remain binding in allowed spatial embeddings. Their
general static/partial-arrest closure is still unresolved.

\textbf{Verification.} 400 updated controls pass at maximum scaled native/Python
error $1.04\cdot10^{-15}$. Another 2,000 controls span masses $10^{-6}$--$10^6$ kg
and radii $10^{-9}$--$10^6$ m with maximum scaled error $1.02\cdot10^{-15}$.
Worker counts produce bit-identical responses, and every timed million-response
output is compared against the single-worker output. Energy, composition and
100 rotated supported controls remain passing. An initial C-probe test used
noncontiguous storage for a field-major pointer; its failure is retained and the
corrected probe passes the unchanged native contract.

\textbf{Experimental local memory.} A separate one-to-three-mode spring/slider
component uses midpoint elastic storage and a static test followed by a dynamic
convex return map. Small factorizations are cached; at most 27 active sets are
needed. 300 randomized coupled-mode controls check passivity and permutation
invariance. Opening applies no impulse and transfers stored energy to an explicit
internal-mode ledger. The caller must retain or physically relax that energy;
it is not silently deleted or assumed to be measured heat. This component is
not called by the native batch and is not a completed impact-deformation model.
\newpage
\section*{Compiler import boundary and readiness}
\texttt{supported\_backend/batch.h} is C11/C++17 compatible. ABI v1 exposes
version/field counts, host worker limit, input validation and synchronous update.
The machine-readable field names and units are in \texttt{schema.json}; the full
ownership and lowering contract is in \texttt{INTEGRATION.md}.

Use field-major IEEE Float64 arrays and one body index $k$:
\[
\texttt{input[field*count+k]},\qquad\texttt{output[field*count+k]}.
\]
There are fifteen inputs and eleven outputs. The caller owns disjoint buffers;
no pointer is retained and no per-body heap allocation occurs. Explicit workers
operate on independent responses. Success is \texttt{SIZE\_MAX}; global invalid
arguments have a distinct status, otherwise return the earliest failed body.
Discard the entire output after failure. No fallback law is substituted.

Python's \texttt{PreparedBatch} owns an immutable validated input copy and can
write directly into caller-provided output storage. Preparation and allocation
must be measured separately for a full application. Native callers validate when
inputs change and honor the buffer contract. The synchronous update checks
branch, finite-range and energy/work admissibility.

\textbf{Ready for scoped import:} native supported branch, field/units schema,
failure semantics, independent body loop, scalar and wide-scale importer oracles,
ownership contract, warning-free source and explicit benchmark evidence.

\textbf{Not qualified:} general collision/contact groups, arbitrary tensor inertia
inside this scalar branch, evolving full t/s history, a physical coupled patch
capacity law, independent empirical calibration, native/WASM/GPU compiler
equivalence or main-compiler acceptance. The new memory component stays
experimental. Measured inertia input elsewhere in the 3D engine remains separate.
Normal/tangential restitution belongs to impact branches and is not reapplied to
this sustained-contact law.

\textbf{Import procedure.} Start from the compiler's accepted paired checkpoint
and current Section 0 authority. Add new fixtures without editing old oracles.
Compare all eleven channels with the archived physical scales; run actual target,
ownership and preservation checks before claiming a compiler integration.
This package changes no compiler seed/main/baseline or hardware acceptance.

\textbf{Experimental accuracy.} Previous glass/public-data residuals are unchanged.
No new measured improvement is shown. Tangential history/partial slip is
established prior physics, e.g. \href{https://websites.umich.edu/~jbarber/Wear1976.pdf}{Maw--Barber--Fawcett (1976)}
and \href{https://doc.lammps.org/pair_granular.html}{LAMMPS granular documentation};
these motivate local state but do not validate this reduced component or supply
every independent material parameter.

\textbf{Reproduction.} Package README and \texttt{supported\_backend/INTEGRATION.md}
give build/run commands. Benchmark source/receipts are in
\texttt{research/supported-batch-optimization/}; \texttt{run-v1/v2/v3} retain all
attempts. The downloadable bundle includes the native core, Python reference,
focused tests and import oracles. Whole-engine regressions require the full
repository. Source/member hashes and extracted-bundle test results accompany
the final artifact receipt.
\end{document}
'''.replace('@@ROWS@@',rows)
    (out/'report.tex').write_text(text)
    for k in [1,2]:
        result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error','report.tex'],cwd=out,capture_output=True,text=True)
        (out/f'build-{k}.txt').write_text(result.stdout+result.stderr);result.check_returncode()
    print(out/'report.pdf')
