"""Standalone audited CPU timing figure; observed ranges are not confidence intervals."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent
audit=json.loads((HERE/'independent-audit.json').read_text());assert audit['passed']
names=['2d_shake9','2d_shake25','3d_sphere'];labels=['2D · 9 disks','2D · 25 disks','3D · 27 spheres']
fig,ax=plt.subplots(figsize=(9,5.5));x=np.arange(3);width=.34
for offset,variant,color,label in [(-width/2,'reference','#5b6f86','Fine reference'),(width/2,'candidate','#168c80','Qualified setting')]:
    medians=[];lower=[];upper=[]
    for name in names:
        record=audit['qualified_benchmarks'][name];summary=json.loads((HERE/record['record_directory']/'summary.json').read_text())[name]['benchmark']
        values=summary['samples_s'][variant];median=record['median_s'][variant];medians.append(median);lower.append(median-min(values));upper.append(max(values)-median)
    ax.bar(x+offset,medians,width,color=color,label=label,yerr=np.array([lower,upper]),capsize=4)
    for at,value in zip(x+offset,medians):ax.text(at,value*1.12,f'{value:.3f}s',ha='center',fontsize=9)
ax.set_yscale('log');ax.set_ylim(.1,30);ax.set_xticks(x,labels);ax.set_ylabel('Native physics time (seconds, log scale)');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(loc='upper left')
ax.set_title('Qualified rapid shaking with friction 0.4\n±20 m/s reversals · 0.12 s simulated',pad=15)
for i,name in enumerate(names):ax.text(i,.12,f"{audit['qualified_benchmarks'][name]['ratio']:.2f}×",ha='center',fontweight='bold')
fig.text(.07,.03,'Three alternating repetitions; bars are medians, whiskers observed min/max. External host load uncontrolled.\n2D timer excludes per-frame output/diagnostics; 3D includes state recording. Compare settings within each scene.',fontsize=8)
fig.tight_layout(rect=[0,.1,1,1]);fig.savefig(HERE/'report.pdf');fig.savefig(HERE/'timings.png',dpi=180)
