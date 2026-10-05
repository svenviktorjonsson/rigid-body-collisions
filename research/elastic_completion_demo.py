"""Render recorded elastic contact evidence; never simulate a new trajectory.

Creates a standalone interactive HTML playback and publication-ready plots
from the completed immutable archive. Sphere orientation is absent from the
prototype state; an optional stripe illustrates integrated normal-axis spin.
"""
import argparse,hashlib,html,io,json,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEFAULT_CASES=('elastic-normal-axis-spin','vertical-floor-ceiling','same-floor-oblique-mixed-positive',
               'same-floor-oblique-mixed-negative','low-friction-no-spin-reversal','weighted-high-spin-low-friction-budget')


def load(directory):
    summary=json.loads((directory/'summary.json').read_text());plan=json.loads((directory/'plan.json').read_text())
    if not summary['complete']:raise ValueError('Archive incomplete; render only after every declared attempt is retained')
    specs={row['name']:row for row in plan['cases']};rows={row['name']:row for row in summary['cases']};records={}
    with zipfile.ZipFile(directory/'traces.zip') as archive:
        for name in DEFAULT_CASES:
            if not rows[name]['qualified']:raise ValueError('Unqualified selected case: '+name)
            with np.load(io.BytesIO(archive.read(name+'/fine.npz')),allow_pickle=False) as trace:
                data={key:trace[key].copy() for key in trace.files}
            data['spec']=specs[name];data['events']=rows[name]['runs'][-1]['metrics']['events'];records[name]=data
    return summary,records


def plots(records,out):
    selected=('elastic-normal-axis-spin','vertical-floor-ceiling','same-floor-oblique-mixed-positive')
    figure,axes=plt.subplots(3,3,figsize=(12,8),constrained_layout=True)
    for column,name in enumerate(selected):
        r=records[name];t=r['times'];s=r['states']
        axes[0,column].plot(t,s[:,2],label='centre height');axes[0,column].set_title(name.replace('-',' '),fontsize=10)
        axes[0,column].set_ylabel('Height (m)')
        axes[1,column].plot(t,s[:,8],label='normal-axis spin');axes[1,column].plot(t,s[:,7],label='tangent-axis spin',alpha=.7)
        axes[1,column].set_ylabel('Angular velocity (rad/s)');axes[1,column].legend(fontsize=8)
        axes[2,column].plot(t,s[:,3],label='horizontal');axes[2,column].plot(t,s[:,5],label='vertical')
        axes[2,column].set_ylabel('Velocity (m/s)');axes[2,column].set_xlabel('Time (s)');axes[2,column].legend(fontsize=8)
        for row in range(3):
            axes[row,column].axhline(0,color='#999',linewidth=.5);axes[row,column].grid(alpha=.2)
            for event in r['events']:
                if event['kind']=='lift_off':axes[row,column].axvline(event['time_s'],color='#888',alpha=.35,linewidth=.8)
    figure.suptitle('Recorded sphere/fixed-plane elastic wrench trajectories\nSynthetic material; vertical and same-floor chains are distinct experiments',fontsize=12)
    figure.savefig(out/'spin-and-bounces.png',dpi=180);figure.savefig(out/'spin-and-bounces.pdf');plt.close(figure)
    r=records['elastic-normal-axis-spin'];figure,axes=plt.subplots(2,1,figsize=(8,6),constrained_layout=True)
    axes[0].plot(r['times'],r['kinetic_J'],label='kinetic')
    axes[0].plot(r['times'],np.sum(r['normal_stored_J'],axis=1),label='normal spring')
    axes[0].plot(r['times'],np.sum(r['twist_stored_J'],axis=1),label='twisting spring')
    axes[0].plot(r['times'],r['dissipated_J'],label='dissipation');axes[0].set_ylabel('Energy (J)');axes[0].legend()
    axes[1].plot(r['times'],r['couple_impulse_N_m_s'][:,2]);axes[1].set_ylabel('Independent couple impulse (N m s)');axes[1].set_xlabel('Time (s)')
    for ax in axes:ax.grid(alpha=.2)
    figure.suptitle('Spin reversal stores and releases torsional energy\nThe couple impulse is separate from the moment of the point force')
    figure.savefig(out/'spin-energy.png',dpi=180);figure.savefig(out/'spin-energy.pdf');plt.close(figure)


def playback(records,out,summary):
    cases={}
    for name,r in records.items():
        t=r['times'];s=r['states'];angle=np.r_[0,np.cumsum(.5*(s[1:,8]+s[:-1,8])*np.diff(t))]
        cases[name]=dict(times=t.tolist(),states=s.tolist(),angle=angle.tolist(),radius=r['spec']['material']['radius'],
                         planes=r['spec']['simulation'].get('planes',[{'normal':[0,0,1],'offset':0,'name':'floor'}]),
                         events=r['events'])
    data=json.dumps(cases,separators=(',',':')).replace('<','\\u003c')
    content=r'''<!doctype html><meta charset="utf-8"><title>Elastic spin and bounce evidence</title>
<style>body{font:16px system-ui;margin:24px auto;max-width:960px;color:#192c3d;background:#fafcff}canvas{width:100%;border:1px solid #ddd;border-radius:10px;background:white}button,select{font:inherit;padding:8px;margin:8px 8px 8px 0}input{width:100%}.scope{background:#edf2f8;padding:14px;border-radius:8px}#values{font-variant-numeric:tabular-nums;white-space:pre-wrap}</style>
<h1>Spin and bounce playback</h1><p>Recorded numerical trajectories of a sphere contacting fixed planes with an elastic force and an independent twisting couple.</p>
<p class="scope">Synthetic rubber-like material; no measured rubber calibration. These elastic sphere/plane experiments are a separate prototype from the native arbitrary-body, inelastic Coulomb solver. Vertical floor/ceiling chains and oblique same-floor chains are separate verified cases. The stripe illustrates integrated normal-axis spin; sphere orientation is not a simulated state.</p>
<select id="case"></select><button id="play">Play</button><label>Playback <select id="speed"><option value="1">1×</option><option value=".2">0.2×</option><option value=".05">0.05×</option></select></label>
<canvas id="canvas" width="960" height="460"></canvas><input id="time" type="range" min="0" value="0" step="1"><p id="values"></p>
<p>Blue path: recorded centre trajectory. Orange arrow: horizontal velocity. Green arrow: vertical velocity. Dashed lines: planes.</p>
<p>Source checkpoint: COMMIT. No physics is recomputed in this viewer.</p>
<script>const cases=DATA;const select=document.getElementById('case'),slider=document.getElementById('time'),canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');let playing=false,last=null,index=0;Object.keys(cases).forEach(name=>{const option=document.createElement('option');option.value=name;option.textContent=name.replaceAll('-',' ');select.append(option)});
function draw(){const d=cases[select.value],i=Number(slider.value),s=d.states[i],R=d.radius;let xmin=Math.min(...d.states.map(s=>s[0]))-1.5*R,xmax=Math.max(...d.states.map(s=>s[0]))+1.5*R,zmin=0,zmax=Math.max(...d.states.map(s=>s[2]))+1.5*R;for(const p of d.planes){if(p.normal[2]<0)zmax=Math.max(zmax,-p.offset+R*.3)}const scale=Math.min(820/(xmax-xmin),340/(zmax-zmin)),X=x=>480+(x-(xmin+xmax)/2)*scale,Z=z=>410-(z-zmin)*scale;ctx.clearRect(0,0,960,460);ctx.strokeStyle='#8795a5';ctx.setLineDash([7,5]);for(const p of d.planes){if(Math.abs(p.normal[2])===1){let z=p.offset/p.normal[2];ctx.beginPath();ctx.moveTo(50,Z(z));ctx.lineTo(910,Z(z));ctx.stroke();ctx.fillStyle='#526477';ctx.fillText(p.name,60,Z(z)-7)}}ctx.setLineDash([]);ctx.beginPath();ctx.strokeStyle='#abc9e3';for(let k=0;k<=i;k++){const q=d.states[k];k?ctx.lineTo(X(q[0]),Z(q[2])):ctx.moveTo(X(q[0]),Z(q[2]))}ctx.stroke();ctx.fillStyle='#3b82b8';ctx.beginPath();ctx.arc(X(s[0]),Z(s[2]),R*scale,0,2*Math.PI);ctx.fill();ctx.save();ctx.beginPath();ctx.arc(X(s[0]),Z(s[2]),R*scale,0,2*Math.PI);ctx.clip();ctx.strokeStyle='#eaf5ff';ctx.lineWidth=4;ctx.beginPath();let stripe=Math.sin(d.angle[i])*R*.7*scale;ctx.ellipse(X(s[0])+stripe*.5,Z(s[2]),Math.max(2,Math.abs(Math.cos(d.angle[i]))*R*.28*scale),R*scale,0,0,Math.PI*2);ctx.stroke();ctx.restore();function arrow(dx,dz,color){let x=X(s[0]),z=Z(s[2]),length=Math.hypot(dx,dz);if(length<1e-12)return;let a=50/Math.max(1,length);dx*=a;dz*=-a;ctx.strokeStyle=color;ctx.lineWidth=3;ctx.beginPath();ctx.moveTo(x,z);ctx.lineTo(x+dx,z+dz);ctx.stroke()}arrow(s[3],0,'#e58d23');arrow(0,s[5],'#218a64');ctx.fillStyle='#152d46';ctx.font='20px system-ui';ctx.fillText('Normal-axis spin: '+s[8].toFixed(4)+' rad/s',40,35);document.getElementById('values').textContent='t = '+d.times[i].toFixed(6)+' s    vx = '+s[3].toFixed(6)+' m/s    vz = '+s[5].toFixed(6)+' m/s\nωy = '+s[7].toFixed(6)+' rad/s    ωz = '+s[8].toFixed(6)+' rad/s';}
function reset(){playing=false;index=0;slider.max=cases[select.value].times.length-1;slider.value=0;last=null;draw()}select.onchange=reset;slider.oninput=()=>{index=Number(slider.value);draw()};document.getElementById('play').onclick=()=>{playing=!playing;last=null};function tick(now){if(playing){const d=cases[select.value];if(last!==null){let target=d.times[Number(slider.value)]+(now-last)/1000*Number(document.getElementById('speed').value);while(index<d.times.length-1&&d.times[index]<target)index++;slider.value=index;if(index===d.times.length-1)playing=false;draw()}last=now}requestAnimationFrame(tick)}reset();requestAnimationFrame(tick);</script>'''
    (out/'demo.html').write_text(content.replace('DATA',data).replace('COMMIT',html.escape(summary['execution_source_commit'])))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,default=Path('research/elastic-completion'));parser.add_argument('--output',type=Path,default=Path('research/elastic-completion-visuals'))
    args=parser.parse_args();summary,records=load(args.archive);args.output.mkdir(parents=True,exist_ok=True)
    plots(records,args.output);playback(records,args.output,summary)
    provenance=dict(archive=str(args.archive),source_commit=summary['execution_source_commit'],selected_cases=list(records),
                    archive_sha256={name:hashlib.sha256((args.archive/name).read_bytes()).hexdigest() for name in ('summary.json','plan.json','traces.zip')},
                    outputs_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.suffix in ('.png','.pdf','.html')},
                    scope='Recorded synthetic sphere/fixed-plane prototype; numerical validation is separate from native arbitrary-body Coulomb and from experimental rubber authentication.')
    (args.output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Rendered six qualified cases, two scientific figures and standalone playback')


if __name__=='__main__':main()
