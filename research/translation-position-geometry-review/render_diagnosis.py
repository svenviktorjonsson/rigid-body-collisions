"""Static independent illustration of the captured position-row conflict."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).parent;D=P/'results';g=json.loads((D/'rejected-normal-system.json.geometry.json').read_text());w=json.loads((P.parents[1]/'research/translation-position-certificate/witness.json').read_text());support=w['support'];weights=[int(v['numerator'])/int(v['denominator'])for v in w['weights']]
fig,axes=plt.subplots(1,2,figsize=(10,4.1));colors=['#32658f','#ce483e','#568144'];point=np.zeros(2)
for i,weight,color in zip(support,weights,colors):
 force=-weight*np.array(g['rows'][i]['normal_world_on_b'])[:2];axes[0].annotate('',xy=point+force,xytext=point,arrowprops=dict(arrowstyle='->',color=color,lw=2.5));middle=point+.5*force;axes[0].text(middle[0],middle[1]+.018,f'row {i}',color=color,fontsize=10);point+=force
axes[0].plot([0],[0],'ko',ms=4);axes[0].set_aspect('equal');axes[0].set_xlim(-.50,.035);axes[0].set_ylim(-.09,.16);axes[0].axis('off');axes[0].set_title('Positive contact directions nearly cancel',fontsize=12);axes[0].text(-.49,-.072,'Weighted impulse directions on hull 20 (XY projection)\nLinear closure norm: 1.45e−8; rotation remains available.',fontsize=9)
for j,(i,color) in enumerate(zip(support,colors)):
 gap=g['rows'][i]['signed_distance_m'];axes[1].barh(j,abs(gap)*1e9,color=color,height=.55);axes[1].text(max(abs(gap)*1e9*1.2,3),j,('gap ' if gap>0 else 'penetration ')+f'{abs(gap)*1e9:.4g} nm',va='center',fontsize=10)
axes[1].set_xscale('log');axes[1].set_xlim(.5,1e8);axes[1].set_yticks(range(3),[f'row {i}'for i in support]);axes[1].invert_yaxis();axes[1].set_xlabel('Magnitude of actual signed distance (nm)');axes[1].set_title('Separated rows were given zero position target',fontsize=12);axes[1].spines[['top','right']].set_visible(False)
fig.suptitle('Container 0 / random hull 20: translation-repair conflict',fontsize=14);fig.text(.04,.01,'Same frozen 74-row matrix. This is a numerical position-repair diagnosis, not physical collision-law infeasibility.',fontsize=9);fig.tight_layout(rect=(0,.08,1,.94))
fig.savefig(D/'position-conflict.png',dpi=180);fig.savefig(D/'position-conflict.pdf');plt.close(fig)
