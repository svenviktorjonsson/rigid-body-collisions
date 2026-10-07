"""Measured limestone impact/rotation consistency against a declared sphere approximation.
No fitted friction or restitution; this does not test the unavailable actual
polyhedron geometry/contact point. COM ratios are not contact-slip restitution.
"""
import argparse,hashlib,json,math
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('xlsx',type=Path);args=p.parse_args();ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
 with ZipFile(args.xlsx) as z:
  strings=[''.join(t.itertext()) for t in E.fromstring(z.read('xl/sharedStrings.xml'))]
  rows=E.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row',ns)
  current={};records=[]
  for row in rows:
   cells={}
   for cell in row:
    node=cell.find('s:v',ns)
    if node is None:continue
    value=node.text
    if cell.get('t')=='s':value=strings[int(value)]
    col=''.join(x for x in cell.get('r') if x.isalpha());cells[col]=value
   if int(row.get('r'))<=17 or cells.get('A')=='D':continue
   for key in ('A','B','C'):
    if cells.get(key):current[key]=float(cells[key])
   if not all(cells.get(k) for k in ('D','E','H','I','N')):continue
   diameter=current['A']/100;r=diameter/2;vni,vti,vnr,vtr,omega=[float(cells[k]) for k in ('D','E','H','I','N')]
   # Exact sphere/flat-plane tangential impulse identity, conditional on
   # collinear measured tangential components and homogeneous inertia.
   induced_spin=abs(vti-vtr)/(.4*r)
   lower,upper=max(0.,induced_spin-3.),induced_spin+3.
   relative_error=abs(induced_spin-omega)/omega if omega else None
   energy_ratio=(vnr*vnr+vtr*vtr+.4*r*r*omega*omega)/(vni*vni+vti*vti)
   records.append({'source_row':int(row.get('r')),'diameter_m':diameter,'slope_angle_deg':current['B'],'release_height_m':current['C'],
      'observed_normal_COM_ratio':float(cells['J']),'observed_tangent_COM_ratio':float(cells['K']),
      'tangent_COM_ratio_is_contact_tangential_restitution':False,
      'observed_outgoing_rotation_speed_rad_s':omega,'sphere_induced_spin_rad_s':induced_spin,
      'sphere_outgoing_spin_range_for_incoming_norm_le3':[lower,upper],
      'observed_spin_inside_conditional_sphere_range':lower<=omega<=upper,
      'relative_spin_discrepancy_at_zero_initial_spin':relative_error,
      'energy_ratio_reconstructed_under_author_sphere_inertia':energy_ratio,
      'published_energy_ratio':float(cells['O']),
      'full_actual_shape_model_validation':False})
 report={'source':'https://doi.org/10.5194/nhess-18-3045-2018','supplement':'https://nhess.copernicus.org/articles/18/3045/2018/nhess-18-3045-2018-supplement.zip',
   'xlsx_sha256':hashlib.sha256(args.xlsx.read_bytes()).hexdigest(),'case_count':len(records),
   'inside_conditional_sphere_spin_range':sum(x['observed_spin_inside_conditional_sphere_range'] for x in records),
   'COM_normal_ratios_above_one':sum(x['observed_normal_COM_ratio']>1 for x in records),
   'max_published_energy_formula_reconstruction_error':max(abs(x['energy_ratio_reconstructed_under_author_sphere_inertia']-x['published_energy_ratio']) for x in records),
   'inertia':'Author sphere approximation; actual irregular polyhedron tensor absent',
   'assumptions':['Homogeneous sphere rather than actual faceted limestone shape','Undeformed-radius point lever','Collinear incoming/outgoing tangential components','Reported incoming rotation norm bound3rad/s, not per-trial signed angular velocity','No independent pair-friction inference'],
   'independent_material_validation_passed':False,'records':records}
 Path(__file__).with_name('comparison.json').write_text(json.dumps(report,indent=2)+'\n')
 print({k:report[k] for k in ('case_count','inside_conditional_sphere_spin_range','COM_normal_ratios_above_one','max_published_energy_formula_reconstruction_error')})
