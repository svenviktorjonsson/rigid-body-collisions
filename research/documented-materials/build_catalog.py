"""Extract selected published parameter profiles; do not fit outcomes."""
from pathlib import Path
from html.parser import HTMLParser
import json,re,hashlib
P=Path(__file__).resolve().parent;CACHE=Path('/home/viktor/.cache/physics-documented-materials-20261006')
class Tables(HTMLParser):
 def __init__(self):super().__init__();self.rows=[];self.row=[];self.cell=None
 def handle_starttag(self,t,a):
  if t=='tr':self.row=[]
  elif t in ['td','th']:self.cell=[]
 def handle_endtag(self,t):
  if t in ['td','th'] and self.cell is not None:self.row.append(' '.join(' '.join(self.cell).split()));self.cell=None
  elif t=='tr' and self.row:self.rows.append(self.row)
 def handle_data(self,s):
  if self.cell is not None:self.cell.append(s)
raw=CACHE/'cornell-results.html';table=Tables();table.feed(raw.read_bytes().decode('cp1252'))
selection=[('glass-soda-binary','Binary','Soda lime glass','Same','3.18'),('glass-fresh-binary','Binary','Fresh Glass','Same','2.97'),('glass-spent-binary','Binary','Spent Glass','Same','2.97'),('acetate-binary','Binary','Cellulose acetate','Same','5.99'),('chrome-steel-binary','Binary','Chrome Steel Hoover Precision','Same','3.18'),('delrin-smallparts-binary','Binary','Delrin Small Parts Inc.','Same','6.35'),('polyurethane-binary','Binary','Polyurethane','Same','4.00'),('glass-soda-aluminum','Flat Plate','Soda lime glass sph.','Smooth Aluminum','3.18'),('acetate-aluminum','Flat Plate','Cellulose acetate','Smooth Aluminum','5.99')]
profiles=[];section=''
def param(text):
 vals=re.findall(r'[-+]?\d*\.?\d+',text)
 if len(vals)!=2:raise ValueError(text)
 return {'value':float(vals[0]),'reported_uncertainty':float(vals[1]),'uncertainty_type':'as printed; exact statistical coverage not supplied by summary page','reported_text':text,'zero_uncertainty_means_exact':False}
for cells in table.rows:
 if len(cells)!=8:continue
 if cells[0]:section=cells[0]
 for id,group,a,b,d in selection:
  if section==group and cells[1:4]==[a,b,d]:
   profiles.append({'id':id,'body_material':a,'other_material':a if b=='Same' else b,'geometry':'sphere_sphere' if group=='Binary' else 'sphere_plane','sphere_diameter_m':float(d)/1000,'sphere_density_kg_m3':float(cells[4])*1000,'normal_restitution':param(cells[5]),'tangential_restitution':param(cells[6]),'sliding_friction':param(cells[7]),'source_url':'https://grainflowresearch.mae.cornell.edu/impact/data/Impact%20Results.html','source_group':group,'source_label':'Fall1999 parameter chart','origin':'published experimental characterization; no coefficients fitted by this project','convention':'u_n_after=-en*u_n_before; beta0 maps to et for nonsliding endpoint; sliding limited by mu','inertia_model':'homogeneous sphere assumption; not a measured tensor','conditions':{'surface_and_supplier':a+' / '+b,'temperature_K':None,'humidity':None,'normal_speed_range_m_s':None,'unknown_conditions_are_defaults':False},'prediction_status':'documented coefficients available; cross-condition predictive accuracy unqualified'})
assert len(profiles)==len(selection)
# Primary Caltech manuscript p78: air,12.7mm spheres,normal velocities50–380mm/s.
for kind,et,de,mu,dm in [('steel',.34,.07,.11,.003),('glass',.39,.06,.106,.008)]:
 profiles.append({'id':kind+'-zerodur-air','body_material':kind+' sphere','other_material':'Zerodur','geometry':'sphere_plane','sphere_diameter_m':.0127,'sphere_density_kg_m3':None,'normal_restitution':{'value':.97,'reported_uncertainty':.02,'uncertainty_type':'reported experimental normal spread'},'tangential_restitution':{'value':et,'reported_uncertainty':de,'uncertainty_type':'author reported uncertainty of line fit'},'sliding_friction':{'value':mu,'reported_uncertainty':dm,'uncertainty_type':'author reported uncertainty of line fit'},'source_url':'https://authors.library.caltech.edu/records/ks4y3-4ct92','doi':'10.1017/S002211200400919X','source_location':'p78; dry collisions in air, Figure4 discussion','origin':'author experimental characterization, not project fit','convention':'author beta maps to signed tangential restitution; mu is sliding friction','inertia_model':'homogeneous sphere assumption','conditions':{'environment':'air','normal_speed_range_m_s':[.05,.38],'diameter_m':.0127,'surface_roughness':'specimen/target described by source; no universal steel/glass property claim'},'prediction_status':'documented coefficients available; raw per-impact files not verified'})
report={'schema':1,'profiles':profiles,'source_sha256':{str(f.name):hashlib.sha256(f.read_bytes()).hexdigest() for f in [raw,CACHE/'foerster-thesis.pdf',CACHE/'joseph-hunt.pdf']},'unsupported_requested_pairs':[{'pair':'limestone / C25 concrete','reason':'Wang2018 publishes COM normal/tangential ratios and elastic properties, not complete signed contact en/et and same-pair mu; no substitute fitted values.'},{'pair':'Cross2010 Superball / granite,rubber,pad,strings','reason':'normal/tangent restitution reported but independently specified same-pair sliding friction missing; mu0.9 remains hypothesis and is excluded from documented-only profiles.'}],'selection_policy':'Explicit specimen/surface/profile id required; no generic material-name fallback. Published uncertainties and unknown conditions retained. Numerical controls do not qualify real-world prediction.'}
(P/'catalog.json').write_text(json.dumps(report,indent=2)+'\n');print('Documented complete three-coefficient profiles',len(profiles))
