"""Rebuild standalone HTML/PDF and audit from retained experiment JSON."""
from pathlib import Path
import json, math, hashlib, html, base64, textwrap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
P=Path(__file__).resolve().parent
load=lambda f:json.loads((P/f).read_text())
f=load('fits.json'); n=load('native-heldout/summary.json'); g=load('geometry-sweep/summary.json'); r=load('rubber-surfaces/summary.json')
keys=['fixed','size','angle']
cv={k:math.sqrt(sum(x['models'][k]['test']['count']*x['models'][k]['test']['normalized_joint_rmse']**2 for x in f['leave_one_plate_angle_out'])/75) for k in keys}
assert (f['training_count'],f['heldout_count'])==(50,25)
for k in keys:
 a=f['models'][k]['train']['predictions'];b=f['models'][k]['heldout']['predictions']
 assert not ({x['source_row'] for x in a}&{x['source_row'] for x in b})
 assert all(x['release_height_m']==4.5 for x in b)
for fold in f['leave_one_plate_angle_out']:
 for k in keys:
  assert all(x['plate_angle_deg']==fold['held_out_plate_angle_deg'] for x in fold['models'][k]['test']['predictions'])
assert n['native_law_pass_count']==n['case_count']==75
assert g['pass_count']==g['case_count']==96
assert r['spin_interval_overlap_count']==3 and r['native_run_count']==36
assert all(x['analytic_native_error']<1e-10 for row in r['rows'] for x in row['native_runs'])
sections=[]
def add(title,*paras):sections.append((title,list(paras)))
add('Restitution validation report — 6 October 2026',
'The normal and tangential restitution model is implemented in the native 3D solver and an isolated simultaneous 2D build. Implementation checks pass. Experimental comparisons show partial agreement, including an unresolved rubber-pad spin discrepancy. These results do not establish an exact real-rock replay or independently validated material friction.',
'This report separates measured comparisons, analytic/native verification and synthetic geometry tests. The original 13-case accuracy-qualified baseline remains OPEN. The user removed the 2x performance requirement; priority is realistic prediction against independent measurements.')
add('1. Model and scope',
'At a shared contact point, u_n(after) = -e_n u_n(before) and u_t(after) = -e_t u_t(before) when friction capacity permits. Normal restitution is restricted to [0,1]; tangential restitution to [-1,1]. A tangential value of -1 preserves contact slip, 0 requests sticking, and positive values request slip reversal. Both coefficients must be supplied together.',
'Tangential impulses obey the circular Coulomb bound |J_t| <= mu J_n. The full contact Jacobian includes lever arms and body inertia. Simultaneous contacts retain coupled solving. A separate actual kinetic-energy minus boundary-work gate rejects energy injection; the shifted restitution target is not used as incoming velocity in that audit.',
'For a homogeneous sphere against a fixed plane with initially zero spin: J_t/m = clip(-(1+e_t)*v_t/3.5, +/-mu*(1+e_n)*v_n); outgoing v_t = v_t + J_t/m; angular speed = 2.5*|J_t/m|/R. These simplifying assumptions apply to the fitted rock comparison, not to the actual faceted specimens.',
'Prior verification: 167 engine tests, 11 manually invoked native checks, 144 native restitution/spin/friction controls and 24 byte-identical default trajectories. Three rotated-box controls also pass. CTest had no registered tests and is not counted as a passing suite. The isolated 2D build has not qualified the complete benchmark baseline.')
add('2. Rock data and prospective validation',
'Wang et al. (2018), DOI 10.5194/nhess-18-3045-2018, publish 75 limestone impacts on concrete, nominal diameters 0.10 and 0.20 m and mean masses 1.2 and 10 kg. The supplement supplies incoming/outgoing velocity magnitudes and outgoing spin. Exact facet meshes, individual inertia tensors, attitude and signed angular-velocity vectors are unavailable. Incoming spin is approximated as zero; the paper reports norms up to 3 rad/s.',
'The prospective plan was pushed before these fits. Training uses 50 impacts at release heights 2.5/3.5 m; the 25 impacts at 4.5 m are held out. Additional validation withholds one whole landing-plate-angle group in each of four folds. Candidate laws have fixed coefficients (3 parameters), separate coefficients by diameter (6), or linear impact-angle coefficients (6), clipped to physical bounds. Three optimization starts are selected by training loss only.',
'Joint RMSE weights normal velocity, tangential velocity and R times angular speed equally after division by each incoming total speed. It is a dimensionless prediction error, not an experimental uncertainty interval. Published COM restitution ratios must not be interpreted as signed contact-slip coefficients.')
lines=[]
for k in keys:
 h=f['models'][k]['heldout'];lines.append(f"{k}: joint RMSE {h['normalized_joint_rmse']:.6f}; normal {h['normal_velocity_rmse_m_s']:.4f} m/s; tangent {h['tangent_velocity_rmse_m_s']:.4f} m/s; spin {h['angular_speed_rmse_rad_s']:.4f} rad/s; pooled angle-group CV {cv[k]:.6f}.")
add('3. Rock results',*lines,
'Fixed fitted coefficients: e_n = 0.492986, e_t = -0.191582, mu = 0.811484. These are effective fit parameters under the sphere approximation, not independently measured material properties. Separate-size parameters have Jacobian rank 5 of 6, so at least one parameter is locally unidentifiable.',
'The size model improves the height-held-out joint error by 1.4%; the angle model improves it by 9.7%. Across held-out angle groups, the size model is about 0.6% worse and the angle model about 9.1% worse than fixed coefficients. There is no robust evidence here to promote a location- or direction-dependent material law. Geometry and missing initial state can also cause the measured variation.',
'All 75 native checks (25 held-out impacts under each of three fitted laws) reproduce analytic endpoints and pass energy/work checks. Maximum endpoint discrepancy is 3.13e-13. This verifies implementation of the assumed sphere law, not the actual experimental rock geometry. No independent same-pair friction measurement was available for a friction-parameter match.')
add('4. Rubber: a discriminating spin comparison',
'Cross (2010), Enhancing the Bounce of a Ball, Table I, reports the same 58 mm, 103 g Superball across four surfaces. Incoming speed is approximately 4 m/s, incidence 25 +/-1 degrees to vertical, initially without spin. Restitution uncertainty is +/-0.01 and spin factor uncertainty +/-0.1 rad/m.',
'The model uses the published normal AND tangential restitution, sphere inertia factor 0.4, and disclosed mu=0.9 hypothesis. The friction cap is inactive, so these runs do not identify mu. The 36 native variations reproduce the analytic endpoints. Spin-factor intervals below vary incidence and tangential restitution across the reported ranges; they are deterministic ranges, not statistical confidence intervals.',
*[f"{x['surface']}: e_n={x['normal_restitution']:.2f}, e_t={x['tangential_restitution']:.2f}; measured spin factor {x['observed_spin_factor']:.1f} +/-0.1; predicted central {x['predicted_central_spin_factor']:.3f}, range {x['predicted_range_over_angle_and_et_uncertainty'][0]:.3f} to {x['predicted_range_over_angle_and_et_uncertainty'][1]:.3f}; overlap {'YES' if x['spin_intervals_overlap'] else 'NO'}." for x in r['rows']],
'Granite, rubber sheet and tennis strings overlap the stated uncertainty ranges. The Superball-material pad does not: 16.34 predicted versus 18.2 +/-0.1 measured. No single sphere inertia factor explains all four within the reported incidence/restitution/spin ranges. Finite contact patches, normal-force offsets or compliant contact dynamics are plausible missing mechanisms, but were not independently measured here. Inferred offsets retained in JSON are diagnostic fits to the same outcomes and are not validated geometry.',
'Supplying restitution measured in the target experiment does not independently predict those restitution values. Public elastic constants and a friction coefficient alone do not specify impact dissipation or hysteresis. An independently specified contact/material law and independent validation trials are needed for the requested predictive material test.')
add('5. Ball size and energy accounting',
'Cross (2002), Grip-slip behavior of a bouncing ball, supplies a separate 46 mm, 46.4 g Superball experiment with a 340 g instrumented support block. The 58 mm and 46 mm records are not a verified same-compound, same-surface multi-size validation set. Their different supports and specimens prevent attributing differences only to radius.',
'For the reported 46 mm trial, reconstructed ball translational energy decreases from 0.167878 J to 0.086278 J while outgoing rotation contains 0.047533 J. Ball kinetic energy changes by -0.034066 J. Translational energy is not separately conserved: rotation, support motion and dissipation must be included. The experimentally constrained moving support has not been replayed as an exact full native constraint model.')
add('6. Non-spherical contact geometry',
'96 synthetic cases vary 2D rectangles/eight-vertex convex shapes and 3D boxes/asymmetric six-vertex hulls over four attitudes, three incoming directions and two fixed friction levels (0.2 and 2). Both restitution coefficients remain 0.6 in every case. All 96 pass normal-target, tangential-target-or-capacity, cone, angular-impulse and energy checks. 45 cases bind the friction cap.',
'Maximum normal target error is 7.85e-9 m/s; maximum angular-impulse error is 8.16e-13. Contact location, direction and inertia alter the outgoing motion without changing restitution. This is evidence that geometry must be tested before adding multiple material values. It is synthetic verification, not a real-rock experiment.',
'The initial reporter failed for 48 planar cases because NumPy removed 2D cross products; the 48 spatial cases passed. The reporter was corrected to the scalar cross product and all 96 cases rerun. The initial errors are retained, and are not classified as physics rejections.')
add('7. Other public data and remaining identification gaps',
'Tschamut2014 (EnviDat) yielded 2,219 before/after scalar rotation records from 74 tests, joined to specimen mass. These natural-rock records lack the full signed angular vectors, orientation/contact normals and independently measured contact parameters needed for complete forward prediction. Derived database licensing and attribution are retained separately.',
'Chant Sura (EnviDat DOI 10.16904/envidat.174) contains manufactured concrete EOTA shapes and richer motion reconstructions: 82 CSV records, 41 with finite gyro/rotation energy. Mixed angular units and sensor-to-shape orientation require verification. Energy-derived diagonal inertia would only reconstruct the authors calculation, not establish a measured inertia tensor.',
'The public rock-friction compilation at Mendeley DOI 10.17632/55hr73zc4y.1 is a literature source, not a matched specimen/impact calibration. Its files were not downloaded after access returned 403. No friction match is claimed.',
'A decisive test requires measured mesh, mass and full inertia tensor; incoming/outgoing vector motion; attitude and contact geometry; support response; independently sourced normal/tangential restitution and friction with uncertainty. Fit on one condition set, freeze all parameters, then predict unseen conditions. Repeat across radius and incidence for the same rubber compound/surface.')
add('8. Engineering status and reproducibility',
'The report evidence was pushed at commit 4ffacf7 on research/adaptive-benchmark-validation, draft PR #1. Numerical model source was frozen from the restitution implementation; plan.json names d999256 and the raw XLSX SHA256 328d727800aba8a00f1ec6c6ef46b85f440f0bad9335c395d88727ca1963e77f. Native binary hashes are retained in experiment summaries.',
'Rebuild with the repository Python environment: python research/restitution-validation-report/build_report.py. fit_rocks.py accepts --xlsx for the source supplement. native_holdout.py, geometry_sweep.py and rubber_surfaces.py retain the native replay construction; existing results must be preserved before rerunning. Publisher PDFs remain in the external cache and are not redistributed.',
'The requested all-13-case working/accuracy baseline is unqualified. Large simultaneous systems and time-refinement checks retain genuine failures. The user removed the 2x performance gate. It is no longer an acceptance requirement. Physical and accuracy checks still apply. This report completes the present comparison study, not those outstanding engineering acceptance gates.')
add('Contact improvement experiment — shear and rocking compliance', 'An isolated passive shear-plus-horizontal-rocking spring hypothesis was tested with shared rocking stiffness fitted on three surfaces and spin predicted on the withheld fourth. Tangential stiffness was selected from the supplied restitution only, on the first root branch. No per-case spin correction was used.', 'Pooled held-out spin RMSE worsened from 0.999 to 1.235 rad/m (23.7%). The withheld pad prediction worsened from 16.343 to 15.841 rad/m versus 18.2 measured. Thirty-six analytic controls and forty independent integration/history-energy checks pass; physical contact-patch admissibility is not established. The candidate is rejected for production adoption. Evidence is retained in research/shear-rocking-contact.', 'A fixed-lever tangential spring cannot change the spin/restitution identity by altering force history alone. Improving this discrepancy requires independently supported contact deformation, inertia or an additional horizontal-axis contact moment, rather than an arbitrary stiffness adjustment.')
add('9. Primary sources',
'https://nhess.copernicus.org/articles/18/3045/2018/ — Wang et al., limestone experiments and public supplement (CC BY 4.0).',
'https://physics.usyd.edu.au/~cross/PUBLICATIONS/48.%20EnhanceBounce.pdf — Cross, Enhancing the Bounce of a Ball (2010).',
'https://physics.usyd.edu.au/~cross/Gripslip.pdf — Cross, Grip-slip behavior of a bouncing ball (2002).',
'https://www.envidat.ch/metadata/tschamut2014 — natural-rock measurements.',
'https://doi.org/10.16904/envidat.174 — Chant Sura dataset.',
'https://data.mendeley.com/datasets/55hr73zc4y/1 — rock-property compilation (access limitation above).')
fig,ax=plt.subplots(figsize=(8,4));x=range(3)
ax.bar([i-.18 for i in x],[f['models'][k]['heldout']['normalized_joint_rmse'] for k in keys],.36,label='Height held out')
ax.bar([i+.18 for i in x],[cv[k] for k in keys],.36,label='Angle groups held out')
ax.set_xticks(list(x),keys);ax.set_ylabel('Normalized joint RMSE (lower is better)');ax.legend();fig.tight_layout();fig.savefig(P/'rock-validation.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(8,4));
for i,row in enumerate(r['rows']):
 lo,hi=row['predicted_range_over_angle_and_et_uncertainty'];c=row['predicted_central_spin_factor']
 ax.errorbar(c,i,xerr=[[c-lo],[hi-c]],fmt='o',color='tab:blue',label='Prediction range' if i==0 else None)
 ax.errorbar(row['observed_spin_factor'],i,xerr=.1,fmt='s',color='tab:orange',label='Measured' if i==0 else None)
ax.set_yticks(range(4),[x['surface'] for x in r['rows']]);ax.set_xlabel('Outgoing spin / incoming speed (rad/m)');ax.legend();fig.tight_layout();fig.savefig(P/'rubber-spin.png',dpi=180);plt.close(fig)
comparison_rows=[]
for row in r['rows']:
 lo,hi=row['predicted_range_over_angle_and_et_uncertainty']
 comparison_rows.append([row['surface'],f"{row['observed_spin_factor']:.1f} +/-0.1",f"{row['predicted_central_spin_factor']:.2f} ({lo:.2f}–{hi:.2f})",'Overlaps reported uncertainty' if row['spin_intervals_overlap'] else 'Mismatch'])
comparison_rows.extend([
 ['Limestone normal velocity','25 held-out measured impacts','RMSE 1.004 m/s','Approximate sphere; actual facets unavailable'],
 ['Limestone tangential velocity','Same 25 impacts','RMSE 1.305 m/s','Approximate sphere; fitted coefficients'],
 ['Limestone angular speed','Same 25 impacts','RMSE 12.876 rad/s','Approximate sphere; initial spin omitted'],
 ['Actual irregular-rock full motion','No complete matched dataset verified','Not validated','Mesh, inertia, attitude and contact geometry missing'],
 ['Independent friction prediction','No matched same-pair measurement','Not validated','Rubber mu assumed; rock mu fitted']])
body=''.join('<section><h2>'+html.escape(t)+'</h2>'+''.join('<p>'+html.escape(p)+'</p>' for p in ps)+'</section>' for t,ps in sections)
body+='<h2>Comparison with reality</h2><p>Rubber values are spin factor in rad/m. Restitution is supplied from the same experiment; these are conditional spin predictions, not independent material predictions.</p><table border="1" cellpadding="8"><tr><th>Case / quantity</th><th>Measured</th><th>Model</th><th>Assessment</th></tr>'+''.join('<tr>'+''.join('<td>'+html.escape(v)+'</td>' for v in row)+'</tr>' for row in comparison_rows)+'</table>'
for title,path in [('Rock generalization','rock-validation.png'),('Rubber spin comparison','rubber-spin.png')]:
 body+='<h2>'+title+'</h2><img alt="'+title+'" src="data:image/png;base64,'+base64.b64encode((P/path).read_bytes()).decode()+'">'
(P/'report.html').write_text('<!doctype html><meta charset="utf-8"><title>Restitution validation report</title><style>body{max-width:950px;margin:40px auto;padding:20px;font:17px/1.6 system-ui;color:#17212b}h2{margin-top:2em}img{max-width:100%}section{break-inside:avoid}</style>'+body)
sections.append(('Comparison with reality', ['Rubber spin factors are rad/m; restitution comes from the same target experiment.']+[' | '.join(row) for row in comparison_rows]))
with PdfPages(P/'report.pdf') as pdf:
 page=0
 for title,paras in sections:
  chunks=[]
  for para in paras:chunks+=textwrap.wrap(para,width=91)+['']
  for start in range(0,len(chunks),42):
   page+=1;fig=plt.figure(figsize=(8.27,11.69));fig.text(.075,.94,title if start==0 else title+' (continued)',fontsize=13,weight='bold')
   fig.text(.075,.895,'\n'.join(chunks[start:start+42]),fontsize=9.4,linespacing=1.5,va='top',family='DejaVu Sans');fig.text(.075,.04,f'Restitution validation | 6 October 2026 | Page {page}',fontsize=8);pdf.savefig(fig);plt.close(fig)
 for path in ['rock-validation.png','rubber-spin.png']:
  fig,ax=plt.subplots(figsize=(11.69,8.27));ax.imshow(plt.imread(P/path));ax.axis('off');pdf.savefig(fig);plt.close(fig)
audit={'training_count':50,'heldout_count':25,'native_checks_passed':75,'geometry_checks_passed':96,'rubber_native_checks_passed':36,'rubber_surfaces_overlap':3,'pooled_angle_cv':cv,'all_13_case_baseline_qualified':False,'twofold_performance_gate_status':'removed_by_user','independent_material_friction_validated':False,'actual_rock_geometry_validated':False,'input_sha256':{str(path.relative_to(P)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [P/'plan.json',P/'fits.json',P/'native-heldout/summary.json',P/'geometry-sweep/summary.json',P/'rubber-surfaces/summary.json']}}
(P/'report-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
