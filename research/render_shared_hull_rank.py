"""Truthful report for the source-pinned bounded-rank-retry hull follow-up."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.audit_shared_hulls import audit,ROOT
DIRECTORY=ROOT/'research/shared-hull-rank-followup'
SOURCE='0862e30736af34b3736e53154f0679a790113f9f'

def main():
 data=audit(DIRECTORY,SOURCE);summary=json.loads((DIRECTORY/'results/summary.json').read_text());plan=json.loads((DIRECTORY/'plan.json').read_text())
 table=['| Scene | Travel fraction | Rows | Outcome | Residual (m/s) |','| --- | ---: | ---: | --- | ---: |'];plot_rows=[]
 failures={f"{r['scene']}/{r['lane']}":r for r in data['rejections']}
 for config in plan['scenes']:
  for i,fraction in enumerate(config['fractions']):
   key=f"{config['id']}/reference_{i}";r=failures.get(key)
   rows=r['rows'] if r else '-';res=r['residual_m_s'] if r else None;status='Rejected' if r else 'Completed; subject to refinement gates'
   table.append(f"| {config['id']} | {fraction:.3g} | {rows} | {status} | {res:.8g} |" if res is not None else f"| {config['id']} | {fraction:.3g} | {rows} | {status} | — |")
   short='8 hulls, seed42' if config['seed']==42 else '27 hulls, seed7301, rotating';plot_rows.append([short,f'{fraction:.3g}',str(rows),status.split(';')[0],f'{res:.6g}' if res is not None else 'See audit'])
 qualified=sum(v['reference_qualified'] for v in data['scenes'].values())
 text=f'''# Shared-contact hull ladder with bounded Jacobian rank retry

The six predeclared attempts yield **{data['history_count']} completed histories**,
**{data['rejection_count']} retained rejections**, and **{qualified} qualified references**.
All original contact, passivity, physical and quarter-budget trajectory gates remain.

{chr(10).join(table)}

Execution source: `{SOURCE}`. Geometry, material parameters, shared contact-point
construction, wall schedules, spin, effort levels and thresholds are unchanged
from the shared-contact source38 study. The numerical change is one relative
singular cutoff1e-10 retry on the projected-equation Jacobian after the ordinary
1e-12 Newton search stalls. The physical mobility matrix A is unchanged, and
recovery remains subject to the same global256-SVD-call ceiling and full original
contact/passivity gate. This is numerical search, not physical compliance.

The frozen earlier48-row seed42 system independently and natively passes after
this retry. The seed42 fraction0.03 trajectory progresses beyond that particular
failure, then encounters a later rejection. A captured-system success therefore
does not imply completion or accurate refinement of the whole trajectory.

Every attempt, rejected matrix, exact reason, source, authored scene, binary
identity and checkpoint remains archived. The independent auditor checks source
identity, paired controls and scenes, snapshot hashes, all rejection gates and
reference receipts. It independently derives mass/inertia, trajectory errors,
energy and authored surface containment for completed histories. No accuracy
claim is possible for an incomplete history. Earlier failed studies remain intact.

The reported timings are descriptive under recorded collaborative workload;
no new speed ranking or mechanical infeasibility claim follows from this study.

Audit: `python -m research.audit_shared_hulls --directory research/shared-hull-rank-followup --source-commit {SOURCE}`.
'''
 (DIRECTORY/'report.md').write_text(text)
 fig=plt.figure(figsize=(11.7,8.3));fig.patch.set_facecolor('white')
 fig.text(.07,.91,'Shared-contact hulls: bounded rank retry',fontsize=21,weight='bold')
 fig.text(.07,.855,f"{data['rejection_count']} rejections  •  {data['history_count']} histories  •  {qualified} qualified references",fontsize=16,color='#943d35')
 ax=fig.add_axes([.07,.38,.86,.4]);ax.axis('off');t=ax.table(cellText=plot_rows,colLabels=['Scene','Fraction','Rows','Outcome','Residual (m/s)'],cellLoc='left',loc='center',colWidths=[.4,.12,.09,.15,.24]);t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2.1)
 for (row,col),cell in t.get_celld().items():
  cell.set_edgecolor('#dddddd');cell.set_facecolor('#25394b' if row==0 else '#f4f6f8' if row%2 else 'white')
  if row==0:cell.set_text_props(color='white',weight='bold')
 fig.text(.07,.29,'Same shared contact geometry, material and strict 1e-8 m/s law gate as the preceding study.',fontsize=11)
 fig.text(.07,.245,'Only numerical Jacobian increment rank changes; physical mobility and friction remain unchanged.',fontsize=11)
 fig.text(.07,.20,'Solving the earlier captured contact system does not qualify the full hull trajectory.',fontsize=11,weight='bold')
 fig.text(.07,.145,'All levels and failed matrix snapshots retained; independent archive audit passes.',fontsize=10)
 fig.text(.07,.09,'Source: '+SOURCE,fontsize=9,family='monospace');fig.savefig(DIRECTORY/'report.pdf');fig.savefig(DIRECTORY/'results.png',dpi=150);plt.close(fig)
 print('Rendered',DIRECTORY/'report.pdf')
if __name__=='__main__':main()
