"""Error plots from retained results; distinguish reality and synthetic checks."""
from pathlib import Path
import json,numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;R=P.parent/'restitution-validation-report';D=P/'figures';D.mkdir(exist_ok=True)
load=lambda p:json.loads(p.read_text())
balls=load(P.parent/'contact-moment-identification/comparison.json')['rows'];fits=load(R/'fits.json')
fig=plt.figure(figsize=(12,9));grid=fig.add_gridspec(2,3,height_ratios=[1.4,1]);ax=fig.add_subplot(grid[0,:])
labels=[r['ball']+' / '+r['surface'] for r in balls];central=np.array([100*(r['point_law_spin_factor_rad_m']/r['observed_spin_factor_rad_m']-1) for r in balls]);lower=np.array([100*(r['point_law_range'][0]/r['observed_spin_factor_rad_m']-1) for r in balls]);upper=np.array([100*(r['point_law_range'][1]/r['observed_spin_factor_rad_m']-1) for r in balls])
x=np.arange(8);ax.bar(x,central,color=['#427a9c' if r['overlap'] else '#bd5148' for r in balls]);ax.errorbar(x,central,yerr=[central-lower,upper-central],fmt='none',color='black',capsize=3)
for i,r in enumerate(balls):
 e=100*.1/r['observed_spin_factor_rad_m'];ax.plot([i-.3,i+.3],[e,e],color='green');ax.plot([i-.3,i+.3],[-e,-e],color='green')
ax.axhline(0,color='black',linewidth=.7);ax.set_xticks(x,labels,rotation=25,ha='right');ax.set_ylabel('Spin prediction error (%)');ax.set_title('Measured balls: conditional on supplied restitution; assumed sphere inertia\nBlack: angle/et range; green: measured spin uncertainty; red: no overlap')
keys=['fixed','size','angle']
for col,(metric,unit,title) in enumerate([('normal_velocity_rmse_m_s','m/s','Normal velocity'),('tangent_velocity_rmse_m_s','m/s','Tangential velocity'),('angular_speed_rmse_rad_s','rad/s','Angular speed')]):
 ax=fig.add_subplot(grid[1,col]);ax.bar(keys,[fits['models'][k]['heldout'][metric] for k in keys],color=['#427a9c','#8a9e5e','#b89955']);ax.set_ylabel('Held-out RMSE ('+unit+')');ax.set_title('Limestone / '+title);ax.tick_params(axis='x',rotation=15)
fig.suptitle('Agreement with measurements — errors are not all in the same units',fontsize=14);fig.tight_layout(rect=[0,0,1,.96]);fig.savefig(D/'experimental-errors.png',dpi=180);fig.savefig(D/'experimental-errors.pdf');plt.close(fig)
g=load(R/'geometry-sweep/summary.json')['records'];groups={}
for r in g:groups.setdefault(str(r['dimension'])+'D '+r['shape'],[]).append(r)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for ax,metric,title,unit in [(axes[0],'normal_target_error','Maximum normal restitution residual','m/s'),(axes[1],'angular_impulse_error','Maximum angular impulse identity residual','kg m²/s')]:
 labels=list(groups);values=[max(abs(r[metric]) for r in groups[k]) for k in labels];ax.bar(labels,np.maximum(values,1e-15),color='#427a9c');ax.set_yscale('log');ax.set_ylabel(unit);ax.set_title(title);ax.tick_params(axis='x',rotation=15)
 for i,k in enumerate(labels):ax.text(i,values[i]*1.4,f'{len(groups[k])}/{len(groups[k])} pass',ha='center',fontsize=9)
fig.suptitle('Synthetic non-spherical checks — implementation accuracy, not real-world accuracy');fig.tight_layout(rect=[0,0,1,.93]);fig.savefig(D/'verification-errors.png',dpi=180);fig.savefig(D/'verification-errors.pdf');plt.close(fig)
print('Generated experimental and verification plots')
