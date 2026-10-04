"""Regenerate the synthetic local contact result and energy plot."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compliant_contact import integrate_contact,calibrate_normal_damping

root=Path(__file__).resolve().parent
kn=10000.;cn=calibrate_normal_damping(.6,kn)
K=np.array([[2,-.3,.2],[-.3,3,-.4],[.2,-.4,4.]])
result=integrate_contact(K,[-1,.7,2],kn,cn,4000,20,30,2,.6,.4,.03,.02)
summary={'scope':'synthetic local frozen-geometry compliant contact, not calibrated material',
'isolated_normal_restitution_target':.6,'calibrated_normal_damping':cn,
'coupled_effective_normal_restitution':result['effective_normal_restitution'],
'contact_duration':float(result['time'][-1]),'post_velocity':result['post_velocity'].tolist(),
'initial_kinetic_energy':float(result['kinetic'][0]),'final_kinetic_energy':float(result['kinetic'][-1]),
'remaining_elastic_energy':result['residual_elastic_energy_at_opening'],
'dissipated_energy':float(result['dissipation'][-1]),
'maximum_energy_accounting_residual':float(np.max(np.abs(result['energy_accounting_residual'])))}
(root/'compliant-results.json').write_text(json.dumps(summary,indent=2))
fig,axes=plt.subplots(1,2,figsize=(10,4),dpi=160)
for index,label in enumerate(['Normal speed','Tangential speed','Relative spin']):
    axes[0].plot(result['time'],result['states'][index],label=label)
axes[0].set_xlabel('Time');axes[0].set_title('Synthetic local contact');axes[0].legend(fontsize=9)
for values,label in [(result['kinetic'],'Kinetic'),(result['stored'],'Stored elastic'),
                     (result['dissipation'],'Dissipated'),
                     (result['kinetic']+result['stored']+result['dissipation'],'Accounted total')]:
    axes[1].plot(result['time'],values,label=label)
axes[1].set_xlabel('Time');axes[1].set_ylabel('Energy');axes[1].legend(fontsize=9)
fig.tight_layout();fig.savefig(root/'compliant-energy.png')
print(json.dumps(summary,indent=2))
