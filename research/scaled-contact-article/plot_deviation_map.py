"""One measured-deviation map in direction/magnitude coordinates.

q=(|v_n|, |v_t|, ell*|omega|), ell=reported radius fixed by policy.
This direction describes the mix of measured observables, not spatial heading.
"""
from pathlib import Path
import json,math,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
P=Path(__file__).resolve().parent;D=P/'figures';D.mkdir(exist_ok=True)
fits=json.loads((P.parent/'restitution-validation-report/fits.json').read_text());balls=json.loads((P.parent/'contact-moment-identification/comparison.json').read_text())['rows']
def deviation(observed,predicted):
 o=np.asarray(observed);p=np.asarray(predicted);no=np.linalg.norm(o);np_=np.linalg.norm(p)
 if no<=0 or np_<=0:raise ValueError('direction/magnitude needs nonzero signatures')
 cosine=np.clip((o@p)/(no*np_),-1,1)
 return float(np.degrees(np.arccos(cosine))),float(100*(np_/no-1))
records=[]
for row in fits['models']['fixed']['heldout']['predictions']:
 radius=row['diameter_m']/2;o=np.abs(row['observed']);p=np.abs(row['predicted']);o[2]*=radius;p[2]*=radius;a,e=deviation(o,p)
 records.append({'id':f"Rock row {row['source_row']}",'type':'10 cm limestone' if row['diameter_m']==.1 else '20 cm limestone','ell_m':radius,'observed_signature':o.tolist(),'predicted_signature':p.tolist(),'mix_angle_error_deg':a,'magnitude_error_percent':e,'measured_comparison':'Held-out impact; sphere approximation; initial spin omitted','source_row':row['source_row']})
for row in balls:
 R=row['radius_m'];en=row['normal_restitution'];et=row['tangential_restitution'];S=row['observed_spin_factor_rad_m']
 def vectors(angle,tangent,spin,normal=en):
  theta=math.radians(angle);predS=(1+tangent)*math.sin(theta)/(1.4*R)
  # Divide all components by incoming speed, which cancels in both metrics.
  # vt/vin reconstructed from published et=-(vt-R*omega)/vin_t.
  o=np.abs([normal*math.cos(theta),R*spin-tangent*math.sin(theta),R*spin])
  p=np.abs([normal*math.cos(theta),R*predS-tangent*math.sin(theta),R*predS])
  return o,p
 o,p=vectors(25,et,S);a,e=deviation(o,p);ranges=[]
 for angle in [24,26]:
  for tangent in [et-.01,et+.01]:
   for spin in [S-.1,S+.1]:
    for normal in [en-.01,en+.01]:ranges.append(deviation(*vectors(angle,tangent,spin,normal)))
 records.append({'id':row['ball']+' / '+row['surface'],'type':'Superball' if row['ball']=='Superball' else 'Golf ball','ell_m':R,'observed_signature':o.tolist(),'predicted_signature':p.tolist(),'signatures_divided_by_incoming_speed':True,'mix_angle_error_deg':a,'magnitude_error_percent':e,'uncertainty_envelope':{'angle_error_deg':[min(v[0] for v in ranges),max(v[0] for v in ranges)],'magnitude_error_percent':[min(v[1] for v in ranges),max(v[1] for v in ranges)]},'measured_comparison':'Conditional spin comparison; velocity reconstructed from target-supplied restitution; assumed inertia','surface':row['surface']})
assert len(records)==33
assert deviation([1,2,3],[1,2,3])[0]<1e-6
for o,p in [([1,2,3],[2,3,5]),([.1,.5,4],[.2,.4,3])]:
 assert np.allclose(deviation(o,p),deviation(np.array(o)*11,np.array(p)*11))
colors={'10 cm limestone':'#2876a3','20 cm limestone':'#6b5aa5','Superball':'#d77829','Golf ball':'#2d946c'}
fig,ax=plt.subplots(figsize=(10,7))
for kind,color in colors.items():
 rows=[r for r in records if r['type']==kind];ax.scatter([r['mix_angle_error_deg'] for r in rows],[r['magnitude_error_percent'] for r in rows],s=65,c=color,label=kind,alpha=.85,edgecolors='white',linewidths=.7,zorder=3)
for r in records:
 if 'surface' not in r:continue
 a=r['mix_angle_error_deg'];e=r['magnitude_error_percent'];ranges=r['uncertainty_envelope'];lo,hi=ranges['angle_error_deg'];low,high=ranges['magnitude_error_percent'];color=colors[r['type']]
 ax.plot([lo,hi],[e,e],color=color,alpha=.45,linewidth=1,zorder=2);ax.plot([a,a],[low,high],color=color,alpha=.45,linewidth=1,zorder=2)
 if 'pad' in r['surface'] or (r['type']=='Golf ball' and 'strings' in r['surface']):ax.annotate(r['id'],(a,e),xytext=(7,-15 if r['type']=='Superball' else 9),textcoords='offset points',fontsize=8,color=color)
ax.scatter([0],[0],marker='*',s=200,c='black',label='Exact match',zorder=4)
ax.axhline(0,color='#777',linewidth=.8);ax.set_xlim(-.8,max(r['mix_angle_error_deg'] for r in records)*1.13);ax.grid(alpha=.18);ax.set_axisbelow(True)
ax.set_xlabel('Error in velocity–spin mix (degrees)\nAngle between observed and predicted magnitude signatures')
ax.set_ylabel('Error in combined motion magnitude (%)\nAbove zero: too much; below zero: too little')
ax.set_title('Prediction versus reality — one point per comparison',pad=15);ax.legend(frameon=False,loc='upper right',fontsize=9)
fig.text(.1,.015,'Signature: (|v_normal|, |v_tangent|, ell |omega|), with ell = radius. 25 held-out rocks + 8 ball–surface summaries.\nBall lines: reported input/spin uncertainty envelope, not confidence intervals. Rock uncertainties unavailable.',fontsize=8,color='#555')
fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(D/'deviation-map.png',dpi=180);fig.savefig(D/'deviation-map.pdf');plt.close(fig)
artifact={'signature_definition':'q=(abs(vn),abs(vt),ell*abs(omega)); ell=reported radius, fixed policy, not fitted','x':'acos(q_observed dot q_predicted / norms), degrees; angle in observable-signature space, not spatial heading','y':'100*(norm(q_predicted)/norm(q_observed)-1), percent','points':records,'case_count':33,'uncertainties':'Ball angle +/-1deg, et +/-0.01, spin factor +/-0.1 corner envelope; en +/-0.01 varied consistently in both signatures. Not confidence intervals. Rock error bars unavailable.','limitations':['Ball components reconstructed from supplied restitution and spin; not independent whole-vector observations','Rock inputs omit small initial spin and exact shape/inertia','Magnitude signatures omit unreported sign/direction of omega','Different true states can share error-map coordinates; no jitter or artificial separation','Only a two-coordinate summary: norm/mix angle does not identify which component is wrong','Synthetic checks excluded from measured reality plot']}
(P/'deviation-map.json').write_text(json.dumps(artifact,indent=2)+'\n')
# Standalone interactive SVG: hover reveals actual observable signatures and source.
data=json.dumps(records).replace('</','<\\/');palette=json.dumps(colors)
page='''<!doctype html><meta charset="utf-8"><title>Prediction deviation map</title><style>body{font:16px/1.5 system-ui;max-width:1050px;margin:25px auto;padding:15px;color:#24313c}svg{width:100%;background:#fbfcfd;border:1px solid #eee}.hint{color:#52606d}#detail{min-height:130px;background:#f2f5f7;padding:15px}button{margin-right:8px;padding:7px;border:1px solid #ddd;background:white}</style><h1>Prediction versus reality</h1><p>One dot per measured comparison. Exact agreement is at the star. Horizontal position shows a different mix of translation and spin; vertical position shows too much or too little combined motion. Hover or select a dot for details.</p><div id="legend"></div><svg id="chart" viewBox="0 0 1000 640" role="img" aria-label="Direction and magnitude error scatter plot"></svg><div id="detail">Select a point.</div><p class="hint">Signature: (|v normal|, |v tangent|, ell |omega|); ell is the reported radius, fixed before comparison. The angle is in this signature space, not the object's spatial heading. There are25 held-out limestone impacts and8 ball/surface summary comparisons. Ball velocities are reconstructed from reported restitution and spin; they are not independent full-state measurements. Rock geometry and initial spin are approximated. Different cases may overlap; points are never moved to separate them.</p><script>const rows=DATA, colors=COLORS;const svg=document.querySelector('#chart'),NS='http://www.w3.org/2000/svg';const xmin=-.8,xmax=Math.max(...rows.map(r=>r.mix_angle_error_deg))*1.15,ymin=Math.min(...rows.map(r=>r.magnitude_error_percent))-5,ymax=Math.max(...rows.map(r=>r.magnitude_error_percent))+5;const X=x=>85+(x-xmin)/(xmax-xmin)*855,Y=y=>550-(y-ymin)/(ymax-ymin)*500;function node(tag,attrs,text){const n=document.createElementNS(NS,tag);Object.entries(attrs).forEach(([k,v])=>n.setAttribute(k,v));if(text)n.textContent=text;svg.appendChild(n);return n}for(let i=0;i<=5;i++){let x=xmax*i/5,y=ymin+(ymax-ymin)*i/5;node('line',{x1:X(x),x2:X(x),y1:50,y2:550,stroke:'#e6e9eb'});node('text',{x:X(x),y:575,'text-anchor':'middle','font-size':13},x.toFixed(1));node('line',{x1:85,x2:940,y1:Y(y),y2:Y(y),stroke:'#e6e9eb'});node('text',{x:75,y:Y(y)+4,'text-anchor':'end','font-size':13},y.toFixed(1));}node('line',{x1:85,x2:940,y1:Y(0),y2:Y(0),stroke:'#777'});node('text',{x:500,y:617,'text-anchor':'middle'},'Velocity–spin mix error (degrees)');node('text',{x:20,y:305,transform:'rotate(-90 20 305)','text-anchor':'middle'},'Combined motion magnitude error (%)');node('text',{x:X(0),y:Y(0)+7,'text-anchor':'middle','font-size':25},'★');for(const r of rows){const n=node('circle',{cx:X(r.mix_angle_error_deg),cy:Y(r.magnitude_error_percent),r:6,fill:colors[r.type],stroke:'white','stroke-width':1,tabindex:0});const title=document.createElementNS(NS,'title');title.textContent=r.id;n.appendChild(title);function show(){document.querySelector('#detail').textContent=r.id+' | '+r.mix_angle_error_deg.toFixed(2)+'° mix error | '+r.magnitude_error_percent.toFixed(2)+'% magnitude error. Observed ['+r.observed_signature.map(x=>x.toFixed(3)).join(', ')+']; predicted ['+r.predicted_signature.map(x=>x.toFixed(3)).join(', ')+']. '+(r.signatures_divided_by_incoming_speed?'Components normalized by incoming speed. ':'Components in m/s. ')+r.measured_comparison;}n.addEventListener('mouseenter',show);n.addEventListener('focus',show);n.addEventListener('click',show);}const reset=document.createElement('button');reset.textContent='All types';reset.onclick=()=>svg.querySelectorAll('circle').forEach(n=>n.style.opacity='1');document.querySelector('#legend').appendChild(reset);for(const [kind,col] of Object.entries(colors)){const b=document.createElement('button');b.style.color=col;b.textContent='● '+kind;b.onclick=()=>{[...svg.querySelectorAll('circle')].forEach((n,i)=>n.style.opacity=rows[i].type===kind?'1':'.15')};document.querySelector('#legend').appendChild(b);}</script>'''.replace('DATA',data).replace('COLORS',palette)
(P/'deviation-map.html').write_text(page)
print('Deviation map:33 measured comparisons; no synthetic passes mixed in')
