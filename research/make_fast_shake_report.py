"""Render the retained fast-shake diagnosis and evaluate archived candidates."""
import json
from pathlib import Path
import zipfile
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.audit_fast_shake_diagnostic import DIRECTORY,errors,physical

def main():
    directory=DIRECTORY;root=Path(__file__).parents[1]
    scene=json.loads((directory/'scene.json').read_text())['scene']
    plan=json.loads((directory/'fixed-correction-plan.json').read_text())
    receipt=json.loads((directory/'corrected-extension-summary.json').read_text())
    load=lambda name:json.loads((directory/(name+'.json')).read_text())
    fixed=load('fixed_1_25us_corrected');guard=load('eighth_travel')
    comparisons={}
    with zipfile.ZipFile(root/'research/spatial-friction/results/traces.zip') as archive:
        failed=json.loads(archive.read('fast_shake27_spheres/reference_4.json'))
        previous=json.loads(archive.read('fast_shake27_spheres/reference_3.json'))
        for candidate in ('coarse','medium','fine'):
            records=[]
            for i in range(3):
                run=json.loads(archive.read(f'fast_shake27_spheres/{candidate}_{i}.json'))
                d=physical(scene,run)
                e_fixed=errors(run,fixed);e_guard=errors(run,guard)
                passed=all(e[k]<=v for e in (e_fixed,e_guard) for k,v in plan['trajectory_budget'].items())
                passed &= all(d[k]<=v for k,v in plan['physical_gates'].items())
                passed &= receipt['fixed']['qualified'] and receipt['travel']['qualified']
                records.append(dict(qualified=bool(passed),errors_vs_fixed=e_fixed,errors_vs_guard=e_guard,
                                    physical=d,step_s=run['step_s']))
            comparisons[candidate]=dict(all_three_qualified=all(r['qualified'] for r in records),runs=records)
    comparisons['cross_reference_errors']=errors(fixed,guard)
    (directory/'candidate-comparison.json').write_text(json.dumps(comparisons,indent=2)+'\n')
    times=np.asarray(failed['times']);a=np.asarray(failed['states'])
    fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for name,result in [('preceding guard .0075',previous),('same guard, half output frame',load('half_output_frame')),('finer guard .00046875',guard),('fixed1.25 μs',fixed)]:
        index={round(t,12):i for i,t in enumerate(result['times'])};b=np.asarray([result['states'][index[round(t,12)]] for t in times])
        for ax,columns in zip(axes,(slice(7,10),slice(10,13))):
            value=np.sqrt(np.mean(np.sum((a[:,1:,columns]-b[:,1:,columns])**2,axis=-1),axis=1))
            ax.plot(times*1000,value,label=name)
    for ax in axes:
        ax.axvline(80,color='.5',ls='--');ax.set_xlabel('simulation time (ms)')
    axes[0].set_ylabel('RMS velocity difference (m/s)');axes[1].set_ylabel('RMS spin difference (rad/s)')
    axes[0].set_title('Difference from retained .00375 trajectory');axes[1].set_title('Second wall reversal occurs at 80 ms')
    axes[1].legend(fontsize=8);fig.savefig(directory/'frame-sensitivity.png',dpi=180);plt.close(fig)
    lines=['# Fast shaking: retained failure, resolved references','',
           'The same 27 spheres and six-wall container were tested at wall speeds ±20 m/s, with gravity, synthetic pair friction 0.4 and zero normal restitution. Both prospectively declared finer references now meet the unchanged quarter trajectory budgets and physical gates. The original failed study remains unchanged.','',
           'The finest old guard fraction 0.00375 had a nonmonotonic trajectory jump. A reproduction with the corrected current backend preserves this branch within 0.00000178 m/s overall RMS. Merely halving output-frame duration adds three internal updates, while keeping the travel cap and physical model unchanged; its prefix agrees through 80 ms to roughly 5e-14 m/s and then follows a different branch after the second reversal. Both solutions satisfy the native contact residual below 1e-8 m/s.','',
           '![Frame partition sensitivity](frame-sensitivity.png)','',
           '| Declared reference | Position edge (m) | Velocity edge (m/s) | Spin edge (rad/s) | Orientation edge (rad) | Pass |',
           '|---|---:|---:|---:|---:|---:|']
    for group,record in receipt.items():
        for edge in record['edges']:
            labels={'fixed_5us':'5','fixed_2_5us':'2.5','fixed_1_25us_corrected':'1.25',
                    'half_travel':'0.001875','quarter_travel':'0.0009375','eighth_travel':'0.00046875'}
            e=edge['error'];label=('Fixed: ' if group=='fixed' else 'Guard: ')+labels[edge['left']]+' → '+labels[edge['right']]+(' us' if group=='fixed' else '')
            lines.append(f'| {label} | {e["position_m"]:.8f} | {e["velocity_m_s"]:.6f} | {e["omega_rad_s"]:.6f} | {e["orientation_rad"]:.7f} | {"yes" if edge["passed"] else "no"} |')
    lines+=['','Each edge must meet quarter budgets: 0.00125 m position, 0.0125 m/s velocity, 0.025 rad/s spin and 0.0025 rad orientation. All three runs in each reference also pass quaternion norm, energy minus measured wall work and internal-update container surface gates. The two finest references agree with each other within these quarter budgets.','',
            '| Original candidate | Spin RMS error versus fixed reference | Spin RMS error versus guard reference | All 3 runs pass full budgets and physical gates |',
            '|---|---:|---:|---:|']
    for name in ('coarse','medium','fine'):
        record=comparisons[name];r=record['runs'][0]
        lines.append(f'| {name} | {r["errors_vs_fixed"]["omega_rad_s"]:.6g} | {r["errors_vs_guard"]["omega_rad_s"]:.6g} | {"yes" if record["all_three_qualified"] else "no"} |')
    lines+=['','The archived fine candidate, travel fraction 0.015, passes against both newly qualified references in all three retained repetitions. Its full-budget velocity error is about 0.00172 m/s and spin error 0.0668–0.0751 rad/s. This is an offline comparison against new evidence, not a rewrite of the original candidate verdict. Medium and coarse candidates remain unqualified.','',
            'The demonstrated failure mechanism is sensitivity to the internal timestep partition near simultaneous contacts. The deeper cause is not yet established: contact birth/retention thresholds, position projection, variable-step warm-start pressure and nonunique Coulomb pressure selection are candidates. This evidence does not distinguish them or prove mathematical nonuniqueness. A frozen contact residual is not a trajectory certificate.','',
            'A useful engineering next step is to make the internal step calendar independent of output-frame sampling, then preserve the same contact law and repeat changing-output-frame tests. Fixed steps already supply a qualified reference for this scene. Variable travel guards require whole-trajectory refinement checks; reducing the travel fraction alone was not monotonic. Changing physical compliance or inventing a pressure regularizer would require a separately declared model and its own validation.','',
            'Two planning errors are retained explicitly. The first plan incorrectly described fixed 5/2.5 microsecond steps as finer than the failed guard: the feature is wall half-thickness 0.025 m, and that guard averaged about 2.21 microseconds. The unchanged extension documents the correction. The requested 8000 primary steps exceeded the adapter limit 4096 and was rejected before simulation. A separately frozen equivalent 1.25 microsecond control uses output dt 0.005 s with 4000 primary steps; comparisons align common 10 ms output times.','',
            'Eight accepted histories and the input-validation rejection are archived. Source snapshots, source-commit checks, binary hash, dependency/input hashes and independent physical/gate recomputation are audited. Three evidence-integrity tests detect consistently rehashed forged qualification and initial geometry.','',
            'Timings came from concurrent research executions and are descriptive only. Neither a speed ranking nor experimental material authenticity is inferred. The horizon is 0.12 s and this reference concerns these 27 spherical bodies; arbitrary hulls and elastic independent torque impulses need their separate evidence.']
    (directory/'report.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':main()
