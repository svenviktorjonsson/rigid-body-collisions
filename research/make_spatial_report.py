"""Render the 3D evidence report and actual random-shape surfaces."""
import itertools
import json
from pathlib import Path
import subprocess
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from spatial_engine import prepare

DIRECTORY=Path(__file__).parent/'spatial-validation'


def draw(ax,scene,result,half,index):
    bodies,_,_,axes,_=prepare(scene);frame=np.asarray(result['states'])[index];box=Rotation.from_quat(frame[0,3:7]).as_matrix()
    for i,b in enumerate(bodies[1:],1):
        R=Rotation.from_quat(frame[i,3:7]).as_matrix()@axes[i]
        color=plt.cm.viridis(i/max(1,len(bodies)-1))
        for s in b['shapes']:
            center=box.T@(frame[i,:3]+R@np.asarray(s['center'])-frame[0,:3]);S=box.T@R@Rotation.from_quat(s['orientation']).as_matrix()
            if s['kind']=='sphere':
                u=np.linspace(0,2*np.pi,12);v=np.linspace(0,np.pi,8)
                p=np.stack([np.outer(np.cos(u),np.sin(v)),np.outer(np.sin(u),np.sin(v)),np.outer(np.ones_like(u),np.cos(v))],axis=-1)*s['radius']+center
                ax.plot_surface(p[:,:,0],p[:,:,1],p[:,:,2],color=color,linewidth=0,alpha=.9)
            else:
                p=np.asarray(s['vertices']) if s['kind']=='hull' else np.asarray(list(itertools.product([-1,1],repeat=3)))*s['half_extents']
                p=p@S.T+center;hull=ConvexHull(p)
                ax.add_collection3d(Poly3DCollection(p[hull.simplices],facecolor=color,edgecolor='#273b46',linewidth=.25,alpha=.95))
    for a,b in itertools.combinations(itertools.product([-half,half],repeat=3),2):
        if sum(x!=y for x,y in zip(a,b))==1:ax.plot(*zip(a,b),color='#516679',alpha=.5,lw=1)
    for setlim in [ax.set_xlim,ax.set_ylim,ax.set_zlim]:setlim(-half,half)
    ax.set_box_aspect((1,1,1));ax.set_xlabel('x [m]');ax.set_ylabel('y [m]');ax.set_zlabel('z [m]');ax.view_init(25,-55)


def main():
    data=json.loads((DIRECTORY/'results/summary.json').read_text());plan=json.loads((DIRECTORY/'plan.json').read_text());scenes=json.loads((DIRECTORY/'results/scenes.json').read_text())
    with zipfile.ZipFile(DIRECTORY/'results/traces.zip') as z:
        fig=plt.figure(figsize=(13,4.5))
        for i,(name,index) in enumerate([('translate64_spheres',-1),('shake_rotate27_hulls42',0),('shake_rotate27_hulls42',-1)],1):
            item=scenes[name];result=json.loads(z.read(f'{name}/standard/0.json'));ax=fig.add_subplot(1,3,i,projection='3d');draw(ax,item['scene'],result,item['half'],index)
            ax.set_title(('64 spheres, 20 m/s, t=1 s' if i==1 else f'27 random hulls, t={0 if index==0 else .12:g} s')+'\nContainer coordinates',fontsize=10)
        fig.tight_layout();fig.savefig(DIRECTORY/'shapes3d.png',dpi=200,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    names=list(data['scenes']);fig,axes=plt.subplots(1,2,figsize=(12,4.5));x=np.arange(len(names))
    for offset,mode in enumerate(plan['candidate_modes']):
        values=[data['scenes'][name]['candidates'][mode]['median_step_s'] for name in names]
        axes[0].bar(x+offset*.18,values,width=.18,label=mode)
    axes[0].set_yscale('log');axes[0].set_ylabel('Median native time [s]');axes[0].legend(fontsize=8);axes[0].set_title('Timing alone does not certify accuracy')
    worst=[max(e['normalized'] for e in data['scenes'][name]['reference_edges']) for name in names]
    axes[1].bar(x,worst,color=['#247756' if data['scenes'][n]['reference_qualified'] else '#aa4444' for n in names]);axes[1].axhline(1,color='black',ls='--');axes[1].set_yscale('log');axes[1].set_ylabel('Worst normalized reference edge');axes[1].set_title('Reference gate: all edges must be <= 1')
    labels=['64 row\n100 m/s','64 spheres\ntranslate 1 s','27 spheres\nshake','27 boxes\ntranslate','27 hulls 42\nshake/rotate','27 hulls 7301\nshake/rotate']
    for ax in axes:ax.set_xticks(x+.27 if ax==axes[0] else x,labels,fontsize=8);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(DIRECTORY/'accuracy-cost.png',dpi=180);plt.close(fig)
    qualified=sum(s['reference_qualified'] for s in data['scenes'].values())
    lines=['# Real 3D rigid collisions: fast moving walls and many bodies','',f'Executed 2026-10-05. **{qualified}/6 references qualify** under the frozen protocol; all 102 retained histories are archived and independently audited.','',
           'The new engine is a genuine 3D, Float64 CPU implementation using unmodified pinned Bullet 3.25. It integrates arbitrary convex hulls, spheres, boxes and compounds with full inertia tensors and quaternion rotation. Twelve analytic and mechanics tests cover mass/inertia, off-center angular impulses, free spin, restitution, friction, 100 m/s walls driving 64 bodies, and 20 m/s containers with translation, shaking and rotation. This is public Python/C++ research; it is not a Vektor language port or compiler acceptance proof.','',
           '![Actual 3D fixture surfaces in container coordinates](shapes3d.png)','',
           '## Frozen protocol','',
           'The six scenes include a 64-body touching row at 100 m/s; a 64-sphere container translating 20 m/s for one second; 27-sphere shaking; 27-box translation; and random asymmetric convex 3D hulls at seeds 42 and 7301 in a container translating/reversing at 20 m/s and rotating/reversing at 10 rad/s. Other horizons are 0.04 s (row) and 0.12 s (containers). All contents are independently integrated; prescribed walls do not teleport them. Friction is synthetic pair coefficient 0.4, restitution zero, gravity 9.81 m/s² except the row. No material calibration or experimental authenticity is claimed.','',
           'Two adjacent travel refinements (fractions 0.06 -> 0.03 -> 0.015) and two adjacent velocity-iteration refinements (64 -> 128 -> 256) must all pass. References use the sequential solver consistently. Candidate RMS budgets are position 0.005 m, velocity 0.05 m/s, angular velocity 0.1 rad/s and quaternion geodesic orientation 0.01 rad; reference edges use one quarter of these values. Bodies and output times match exactly. Physical gates include actual surface containment within 0.002 m, quaternion norm error within 1e-12, nonpositive energy change after subtracting wall work (1 J tolerance), and row final velocity maximum error within 0.015 m/s. Native contact penetration and closing residuals are retained diagnostics; they were not silently added to or removed from the frozen qualification rules.','',
           'Wall motion is integrated in double precision and steps split exactly at velocity commands. The guard bounds relative translation and angular tip travel by a fraction of the smallest fixture half-width/radius, including gravity. It explicitly detects a fast wall crossing a stationary object; the unguarded one-step negative control tunnels. This is a conservative travel heuristic, not a general exact swept-CCD proof. Every internal update checks actual fixture support against every container plane. An independent offline audit reconstructs sampled supports, full kinetic plus gravitational energy, quaternions and every qualification decision.','',
           '## Results','',
           '| Scene | Worst reference edge / budget | Reference | Fastest verified candidate | Native median [s] |',
           '|---|---:|---|---|---:|']
    for name,s in data['scenes'].items():
        choice=s['choice'];cost=f'{s["candidates"][choice]["median_step_s"]:.4f}' if choice else '--'
        lines.append(f'| {name} | {max(e["normalized"] for e in s["reference_edges"]):.3g} | {"pass" if s["reference_qualified"] else "FAIL"} | {choice or "none"} | {cost} |')
    lines += ['', '![Measured timing and reference qualification](accuracy-cost.png)','', '## Limits and interpretation','',
              'A failed reference produces no verified candidate and no accuracy-qualified speed claim. A passing gate certifies only these authored scenes, finite horizons, sample times and four RMS observables; it supplies no universal convergence order, experimental error bound or guarantee for every moving container. Containment and absence of tunneling are weaker claims than trajectory accuracy. The selected fastest mode is an offline decision from one warmup and three retained timing repetitions; adaptive selection remains an online heuristic. Timings include travel control, native mechanical diagnostics and solver choice, but exclude startup and JSON serialization.','',
              'Bullet friction has two independently limited tangent impulses, a pyramid approximation. Its tangent resultant can exceed an isotropic Coulomb circle by up to sqrt(2). Body coefficients multiply at a contact. Separate static/dynamic coefficients, elastic tangential history and rolling/twisting moments are absent from this adapter. Synthetic coefficients are not validated against real material experiments.','',
              'The coupled mode uses Dantzig MLCP with Bullet sequential fallback. The adapter raises the impulse sanity limit from 1000 to 1e30 N·s because high-speed rows can exceed 1000. Fallbacks are retained in every trajectory. Adaptive uses eight sequential iterations until 12 positive-impulse contacts or a closing residual above 0.01 m/s triggers the requested coupled iteration count, with 24-update dwell. It may inherit the coupled fallback and overhead, and is not assumed faster or more accurate.','',
              '| Scene | Fast/coupled/adaptive MLCP fallbacks | Standard max penetration [m] | Standard max surface excess [m] |',
              '|---|---|---:|---:|']
    for name,s in data['scenes'].items():
        c=s['candidates'];d=c['standard']['diagnostics'];counts='/'.join(str(c[m]['diagnostics']['coupled_fallbacks']) for m in ['fast','coupled','adaptive'])
        lines.append(f'| {name} | {counts} | {d["max_contact_penetration_m"]:.4g} | {d.get("container_surface_excess_m",float("nan")):.4g} |')
    lines += ['', '## Reproduction and provenance','',f'Execution source: `{data["execution_source_commit"]}`. Bullet source: `{data["bullet_commit"]}`. Build pins include SHA-256 archives for Bullet and JSON. Full authored geometry, all states, update counts, wall work, fallbacks, timing and source snapshots are retained. The source/data/audit distinguish 3D results from earlier 2D studies.','',
              '```sh','cmake -S spatial_backend -B build/spatial -G Ninja -DCMAKE_BUILD_TYPE=Release','cmake --build build/spatial --target spatial_runner -j 2','python -m unittest tests.test_spatial_engine -v','python -m research.audit_spatial_study','python -m research.run_spatial_study --directory /tmp/fresh-3d-study','python -m research.make_spatial_report','```','',
              'Audit does not require a native build. A fresh execution requires a clean source checkout; new checkpoints reject a different source, binary or plan. Different compiler/platform timing is not assumed identical.','',
              'Sources: [Bullet 3.25 pinned implementation](https://github.com/bulletphysics/bullet3/tree/2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5); [contact solver and friction implementation](https://github.com/bulletphysics/bullet3/blob/2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5/src/BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.cpp); [Dantzig MLCP and fallback](https://github.com/bulletphysics/bullet3/blob/2c204c49e56ed15ec5fcfa71d199ab6d6570b3f5/src/BulletDynamics/MLCPSolvers/btMLCPSolver.cpp). Existing mechanics and solver families are prior art; this verification is not a novelty claim.','']
    (DIRECTORY/'report.md').write_text('\n'.join(lines))
    subprocess.run(['pandoc','report.md','-o','report.pdf','--pdf-engine=pdflatex','-V','geometry:margin=0.7in','-V','fontsize=10pt'],cwd=DIRECTORY,check=True)

if __name__=='__main__':main()
