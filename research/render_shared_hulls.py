"""Render truthful source-pinned shared-contact hull results after audit."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from research.audit_shared_hulls import audit,DIRECTORY,SOURCE

def main():
 data=audit();rows=data['rejections'];table=['| Scene | Travel fraction | Rows | Rejected residual (m/s) |','| --- | ---: | ---: | ---: |']
 for r in rows:table.append(f"| {r['scene']} | {r['fraction']:.3g} | {r['rows']} | {r['residual_m_s']:.8g} |")
 text=f'''# Shared contact-point hull follow-up

All **six** predeclared trajectory attempts reject. There are **zero** completed
histories and **zero** qualified reference ladders. The unchanged circular-law
residual threshold remains **1e-8 m/s**; no failed refinement level is skipped.

{chr(10).join(table)}

Execution source: `{SOURCE}`. Both three-level ladders use the original authored
hulls, density-derived mass/inertia, coefficient mixing, zero restitution,
gravity, 20 m/s shaking, optional 10 rad/s container spin, and travel fractions
0.06/0.03/0.015. Trajectory budgets and physical gates remain unchanged.

The contact discretization is explicitly changed: both finite dynamic bodies
use the midpoint of their surface endpoints; a finite body against a static or
kinematic body uses the finite-body endpoint. Both lever arms, angular mobility,
free-velocity and split RHS, transported warm angular impulses, and boundary
work use the same world point. This corrects wrench geometry, but these results
show that the correction plus current bounded recovery is insufficient to
complete either challenging hull scene.

Every rejection has a matrix snapshot and exact reason in `results/rejections/`.
Source, authored geometry, partial checkpoints, final traces, hashes and binary
identity are retained. Independent audit verifies source provenance, paired
controls/geometry, all rejection snapshots and both failed reference receipts.
It recomputes trajectory/physical gates independently for any completed history;
there are none in this study, so no trajectory-error estimate is available.

Costs are descriptive: the controlled timing study finished before these runs;
independent archive/documentation work could run concurrently. No speed ranking,
mechanical infeasibility, convergence order, or material authenticity follows
from these six rejections. Earlier archives remain unchanged.

Audit: `python -m research.audit_shared_hulls`.
'''
 (DIRECTORY/'report.md').write_text(text)
 fig=plt.figure(figsize=(11.7,8.3));fig.patch.set_facecolor('white')
 fig.text(.07,.91,'Shared contact-point hull follow-up',fontsize=21,weight='bold')
 fig.text(.07,.855,'6 attempts rejected  •  0 histories  •  0 qualified references',fontsize=16,color='#943d35')
 ax=fig.add_axes([.07,.38,.86,.4]);ax.axis('off')
 cells=[[r['scene'].replace('fast_rotate_shake27_hulls7301','27 hulls, seed 7301, rotating').replace('fast_shake8_hulls42','8 hulls, seed 42'),f"{r['fraction']:.3g}",str(r['rows']),f"{r['residual_m_s']:.6g}"] for r in rows]
 table=ax.table(cellText=cells,colLabels=['Scene','Travel fraction','Rows','Rejected residual (m/s)'],loc='center',cellLoc='left',colWidths=[.43,.17,.12,.28]);table.auto_set_font_size(False);table.set_fontsize(11);table.scale(1,2.1)
 for (row,col),cell in table.get_celld().items():
  cell.set_edgecolor('#dddddd')
  if row==0:cell.set_facecolor('#25394b');cell.set_text_props(color='white',weight='bold')
  else:cell.set_facecolor('#f4f6f8' if row%2 else 'white')
 fig.text(.07,.29,'Residual gate: 1e-8 m/s. Original geometry, material, effort ladder and physical gates retained.',fontsize=11)
 fig.text(.07,.245,'Changed discretization: both bodies apply each contact force at one shared world point.',fontsize=11)
 fig.text(.07,.20,'This fixes wrench geometry; it does not qualify either full random-hull shaking trajectory.',fontsize=11,weight='bold')
 fig.text(.07,.145,'Every rejection and matrix snapshot is retained. No skipped levels or friction-law fallback.',fontsize=10)
 fig.text(.07,.09,'Source: '+SOURCE,fontsize=9,family='monospace')
 fig.savefig(DIRECTORY/'report.pdf');fig.savefig(DIRECTORY/'results.png',dpi=150);plt.close(fig)
 print('Rendered',DIRECTORY/'report.pdf')
if __name__=='__main__':main()
