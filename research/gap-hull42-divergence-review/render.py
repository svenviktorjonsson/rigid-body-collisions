"""Independent static refinement-error plot from retained analysis only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;d=json.loads((P/'analysis.json').read_text());fig,axes=plt.subplots(2,2,figsize=(9,6));labels=dict(position_m='Position RMS (m)',velocity_m_s='Velocity RMS (m/s)',omega_rad_s='Angular velocity RMS (rad/s)',orientation_rad='Orientation RMS (rad)')
for ax,key in zip(axes.flat,labels):
 for p,color in zip(d['pairs'][:2],['#32658f','#ce483e']):ax.semilogy([v['time_s']for v in p['perframe_RMS']][1:],[v[key]for v in p['perframe_RMS']][1:],marker='o',ms=3,color=color,label=p['left'].replace('reference_','level ')+ ' → '+p['right'].replace('reference_','level '))
 ax.axhline(d['quarter_budget'][key],ls='--',color='#444',lw=1,label='Original qualification threshold')
 for t in [.04,.08]:ax.axvline(t,ls=':',lw=1,color='#888')
 ax.set_ylabel(labels[key]);ax.set_xlabel('Output time (s)');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
axes[0,0].legend(fontsize=8);fig.suptitle('Random hulls: contact gates pass, trajectory refinement fails',fontsize=13);fig.text(.06,.015,'Dotted lines: wall reversals. Divergence is already visible at 0.01 s, before either reversal.\nOriginal source/scene/material preserved; no causal identification from output traces alone.',fontsize=9);fig.tight_layout(rect=(0,.07,1,.94));fig.savefig(P/'refinement-errors.png',dpi=180);fig.savefig(P/'refinement-errors.pdf');plt.close(fig)
