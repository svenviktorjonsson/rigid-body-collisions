# Measured properties and sustained-contact repair — October 7, 2026

Latest user: find missing model physics, fix it, continue. This checkpoint repairs
two concrete gaps, not the full authentic collision model. First code checkpoint
26d77f3 pushed on research/adaptive-benchmark-validation; latest code/evidence/report
commit identified by git history. No missing remote pushes at task start. Do not
spawn new agents under current developer instructions; no language repo edits.

spatial_engine.py accepts explicit mass_properties with all three SI keys:
mass_kg, center_of_mass_m, inertia_body_kg_m2 (about supplied local COM, body axes).
Maps full measured tensor into native principal axes, preserves collision geometry
and authored position-as-COM semantics. Finite/symmetry/positive/triangle/shape
extent bounds; no proof of exact-shape realizability. Legacy defaults match old
512a1b2 preparation bitwise for three scenes. material_profiles.py now checks
authoritative density and its declared homogeneous-sphere COM/inertia assumptions
instead of allowing measured overrides to bypass specimen guards.

New supported_contact.py public primitive: exact constant-load/drive supported
scalar-I disk/sphere dynamics; mu_s vs mu_d, independent rolling/spin moments with
physical moment lengths, event integration to slip/rolling/axial arrest, no
through-zero reversal. Separate impulse/work/loss channels and moving-plane work.
advance_spatial rotates the motion-plane branch and retains FULL t/s definitions.
Rejects general noncollinear/normal-impact states. Partial angular arrest can need
transverse static moment outside s/n span: explicit rejection test, no relabeled s.
Complete static reactions flagged; original zero/static directional law unresolved.
Independent angular capacities phenomenological, not coupled finite-patch budget.
Native production contacts do NOT automatically use this supported primitive.

39 focused tests PASS (15new/24existing), verification receipt/logs in
research/contact-gap-fix/verification/. Initial coarse free-rotation native test
failed same1e-7 momentum tolerance; 32/64/128/256 refinement retained, 256 passes
8.24e-8. audit-v2:400randomcontrols, energy1.22e-15/composition2.80e-16 maximum
scaled errors, max4intervals; 100rotated/moving-planecontrols angularimpulse2.01e-15
Nms. Tiny motions1e-14 preserved; static constraints projected at exact arrest to
avoid roundoff chatter. No source coefficient fits.

glass-repeat-v1 and reproducible v2 retain 24 native analytic/energy passes and
exact old RMSE:normal.0366739897m/s,COMtangent.0178047683m/s,joint.0241563367.
No empiricalaccuracy gain. Historical contact-tangent target reconstructs spin,
source parameter/evaluation independence incomplete. No rolling/spin calibration.

Native C++ supported kernel:400Pythoncontrols match exactly, O3WallWextraWerror,
no-fast-math. Flat SoA body index k; median100/10k/100k/1M independent response
cost48.6/46.94/80.94/81.41ns, 1Mtotal81.4ms,208MBarrays. 5repeats+warmup,
inputload/all11outputstores/energy/branch checks timed; allocation/detection/poses/
changingloads/impacts/groups excluded. Not a many-body scene or baseline speedup.
User removed2xgate; no restoration. Newreport5pages, syntheticmotiontraces and
glassCOMvelocity magnitude/angle error dots, no invented angular targets.

Host disk filled during benchmark source creation; only our downloaded156426196B
RockMasonry1.rar moved to /tmp/physics-public-cache-preserved-20261007/ after SHA
a2423df131acf78692a18f8732d4054b1e87b5d798f8d322adb5d96d773952f4 verification.
Original cache file absent, no symlink; receipt/source URL in cache-relocation.json.
Temporary copy can disappear on reboot; reproducibly re-download with fetch script.
No measurements deleted, unrelated local-results untouched. Ordinary filesystem
free space remains tight; avoid large builds/copies without checking available.

Next: specify general zero/static-direction convention without widening t/s;
combine unilateral opening and local shear/elastic-mode state with a physical
finite-patch budget/work ledger. Independently characterized contact parameters
and well-resolved incoming/outgoing spin/timing still missing for clean held-out
data. Maw1976 tangential compliance/partial stick is relevant prior art, not a
complete parameter source. Existing all-case contact-group qualification remains
open; do not claim full realism or infer a coefficient merely to erase residuals.

# Public experimental data extension — October 7, 2026

Latest user asks for more public observations to improve authenticity. New isolated
research/public-validation-data/ imports GAUGE22bounce/59sliding/160spatialimpact
trials, MIT1718signedplanarstates, and135limestonerockingtrials (134unique processed
pairs; group1_n13 exactlyduplicates1_n7). All published materialvalues unchanged;
no new fit or empiricalaccuracy gain. Protocol checkpoints80d893e/1ec0648 pushed
before baseline/candidate outcomes; data/evidence checkpointa49eafd pushed.

GAUGE metadata e=.576733 and friction wood/plastic/metal=.273673/.275018/.262758.
Nominal30deg four-frame prefix-conditioned sliding forecasts:56evaluationtrials,
478forecastpoints, positionRMSE6.962/10.824/2.880mm. Measured prefix-board orientation
29.56–29.67deg makes ALL56trials worse, RMSE9.010/12.709/4.573mm; retainfailedcandidate.
Quaternion/ZYXEuleragreement1.74e-6rad. Separate mu_s/mu_d/rolling/tangentiale absent;
pair/calibration/evaluation independence incompletelydocumented.

Bounce audit corrected outcome dependency: original jointarcintersectionusespost
trajectorytoinferinputtime; originalRMSE.382254m/s retainedonlyasconsistencydiagnostic.
marker-v1 uses observedvalleytimestamp and incoming-only fit:42evalevents/21trials,
RMSE.380733m/s,max.671462,mean-.350346. Oneframe sensitivity ±.515592m/s includeszero
for35/42; NOTconfidenceinterval. GAUGEreleased30Hz vs180Hzpapercapture; deformable
ball/plank,contactduration and sceneorigin/pairmatch prevent cleanmaterial inference.
MEASUREMENT-ERRATUM.md explicitlysupersedesoriginalprotocolinput-independenceclaim.

MITmass.0364kg,gyrationradius.0192m,I=1.3418496e-5kgm2; all1718rowsimported. Inferred
freecoupleresidualRMS.00118023Nms NOTindependenttorquemeasurement; finitecontact,
contactpointmotion,guide/externalreactions/gravity/noise cancontribute. Geometry/Jac
mismatch1.72mm; authorcoeffsfitonoutcomes, matchedindependentpropsmissing. v1bodyindex
wascaseindex; v2usescase_index plusbody_index=0; equivalencecheckedall1718, otherCSV/
metricsbitidentical, sourcesnapshotpreserved. Noauthorcode/originalMATredistributed.

Rockinggeometry-onlyHousnerangularratio comparator:134firsteventsRMSE.039013,max.125871,
22ratios>1retained. 7545sourceprocessedrows,7519unique-finite/2820>1 (correlated, not
independentmaterialrestitution). Nominalinertiageometric; effectivegeometryoutcome;
fc=.7 assumedenergycorrection, NOTindependentfriction. 2026release135trials vsassociated
paper120. RAR6 archive156426196bytes SHAa2423df131acf78692a18f8732d4054b1e87b5d798f8d322adb5d96d773952f4;
270processedTXTsextractedviaunrar (authorizedordinaryhostsetup); failed7zemptyfolder
retainedoutsideGitandignored. fetch_rocking.py verifiescontentandcompleteness.

Spatialfullimport160trials,280bodyrecords,4686poses; flatcase/bodyindices,explicit
poseoffset/count, sourcexyzwquatsnormerror8.98e-7. Nofullspatialendpointpredictions:
measuredinertiatensorsabsent; nonsmoothfolders task1/2/3 vsmetadata3/4/5 mismatch,
restitutiondictspreservedwithoutguessmapping. Keep user'sfullt/s/independentdeltaL;
productionunchanged andzero-slipclosureoriginal13scenes/hosted393issuesremainopen.

New2026RemondABSshell/siliconepaper relevant tocheaplocalmemory/spin; originalPDF
cached, sourcecatalogonly, Fig8notdigitized. mu~.92/localstiffness/effectivemassfit
fromsameoutcomes, derivedGnotindependentinput. Do notclaimfixedindependentcharacterization.
24analyticbounceextraction+9planecontrols pass (max1.47e-14m/s); noenginequalification.
Report5pages withzero-error dots, materialtable, failed-candidateplot and provenance;
reproduction/sourcehashes retained. Download package and finalpush noted by latest git.
Next:resolveGAUGEscene/materialmapping, uncertainty-awareimpactstates, thenpassive
smalllocalhistory comparisonon declaredsplits. Noarbitrarycorrection or2xgate.

# Indexed pressure-patch continuation — October 7, 2026

Latest user: continue authentic/efficient work; use indices suitable for later
Vektor Flow. Read spec/bootstrap AGENTS + handovers and authoritative Section0;
no language repo edits/port/compiler/GPU claims. Physics remote fetched: no missing
main/branch pushes; ADMIN push permissions confirmed. Same research branch retained.
New isolated study research/indexed-pressure-patch/; single owner body k, flat
contact/incidence/site ranges, shared footprint templates, SoA fields, explicit
ordered gather/scatter. INDEXED-DESIGN.md describes later-port boundary and material
vs state inputs. No dense pair-body tensors or hidden sums.

Flat unilateral normal foundation + existing dynamic friction now generates
force/couple for mixed sliding/twist/transverse rotation/nonuniform pressure and
2D interval/3D circular/ellipse/irregular patches. Cache3x3Gram pressure moments;
loaded zero-axial-twist branch eliminates site loops exactly. Open/clipped/mixed
branches preserve site-level physics. K,C,patch inputs synthetic, no new fit.
Static/shear history, moving patch/yield, restitution consistency, coupled group
step and experimental calibration still unimplemented; production unchanged.

audit-v2:240 mechanical controls,50loaded/190opening-clipped, scaled power1.92e-15,
indexed work/global momentum4.91e-16; analytic Hertzspin/cachedGram pass; normal
transient stored/lost energy residual1.34e-9J of.5J. reference-point-v1:100origin/
300ell controls,1.89e-15; centerofpressure removes transverse freecouple but does
NOT generally align mixed force with user's full t. Fixed-origin residual alone
does not prove every directional closure incompatible; full zero/static/directional
law remains unresolved. Never relabel t/s or declare conventional patch equivalence.

benchmark-v1:44synthetic local indexed batches100/10k/100k/1Mresponses, gainsall
1.41to27.93x,48native/Python controls2.27e-15. v2alternate-order22confirm10k/100k
allimprove1.42to28.94x,48controls; warning-freeO3WallWextraWerror/no-fast-math.
Gather/scatter/outputreset included, identityframes; allocations/detection/history/
integration/group solving excluded. Responses are NOT interacting body scenes.
v1incorrectmemoryestimate/orderbias retained; v2memorycounts include verification
snapshots. No2xgate restored and original13scenequalification remains open.

refinement-v2:18mixedsyntheticstates against147456site reference;12estimatesadmit,
6decline atworkbudget. Returnednumericerrorsall<=1e-4,max2.06e-5, but declines stay
declines; no general bound. Nearlocal-slipzero canneed65536sites, tooexpensive;
next efficient reduction must preserve force/couple errors and positive work ledger.
audit/refinement-v1 factor2reference-site LABEL mistakes retained with v2 correction,
originalsource snapshots and strict numeric-equivalence receipt; no physics changes.
Initial optionalbooktabs/rowescape PDFbuildfailures retained, corrected report6pages.

No new measured comparisons or materialfits this checkpoint: earlier107collision+
17rolling tables unchanged, prior rock damping/sharedball fits remain rejected.
Next: reconcile pressure origin/directions, small shear/mode history and independent
matched characterization; avoid performance/experimental claims beyond this scope.

# Efficient deformation reduction search — October 7, 2026

Latest user asks to continue finding a more authentic model while retaining efficiency.
New isolated research: research/viscoelastic-relaxation/; report.pdf + signed error-map.
One effective viscoelastic relaxation time per tennis specimen improves existing
8-point evaluation errors: RMSE reductions51.53/67.46/33.71%, all8absolute errors
and all3maximum errors decrease. Same17points and9/8split were already inspected:
EXPLORATORY reuse, not fresh confirmatory validation. Effective time estimates
1.860/2.159/25.039ms are not identified bulk material properties (quasirolling,
apparent belt speed, shell construction, matched material inputs absent).

Related one-parameter normal viscoelastic sphere reduction WORSENS rocks: height
RMSE1.00437to1.13969m/s; pooled angle-fold1.14028to1.19421m/s. Rejected, preserved.
Actual facet attitude/I absent; fixed secondary tangent/sliding inputs historical
estimates. Primary Wang2018section5.1/figure12 documents slab indentation and rim
damage: further rock candidate should encapsulate local yield/indentation geometry,
not just add damping or a raw-impulse polynomial. Photo crater dimensions unmatched
to rows; illustrative conical angles and4passive local-normal controls are not fits.

model.py adds exact supported rolling and repulsive force-zero normal reference.
Remaining elastic energy at release is recorded, not erased; propagation into a
later contact remains open. evidence-v1 sealed; audit-v2 passes200rolling controls
at3ellscales+20ODE+25normal controls, analytic duration/weak damping and data audits.
audit-v1 retains audit-script QR variable-shadowing failure. benchmark-v2 passes16
native/Python controls and warning-free build, no fast-math. Local scalar response
cost about13ns rolling,31nsnormal size/speed+513-node table; table error1.34e-7,
arrays+coefficients24,592bytes. NOT full scenes, contact networks or interacting
million-body benchmarks. benchmark-v1 warning+timings retained;2xgate remainsremoved.

patch_spin.py derives independent axial moment3*pi/16*mu_d*N*a from Hertz pressure,
zero net tangential force but nonzero torque at zero center slip.24rotated traction
quadrature/energy controls plusarrest pass; no new fitted spinningfriction coefficient.
Only pure axial fullsliding with prescribed circular pressure andconstant N/a, not
transient rubber bounce or mixed friction closure. Pressure-field/Hydroelastic
(Elandt/Drake) and reduced plastic contact(Zunker/Kamrin) literature searched as
next comparators; neither implemented/adopted/experimentally validated here.

Production unchanged; native rolling/twisting rejected; original13rapid/irregular
set andhosted393failure remain open. Zero relative velocity leaves user's t
undefined: rolling prototype uses constraint reaction, not redefined t. Need recover
that directional branch before general integration. Updated report retains
notation and explicitly separates mechanical controls from empirical validation.

# Experimental report and local contact-memory core — October 7, 2026

Latest user requests a complete real-data report and cheap rigid-body features encapsulating deformation, including possible pressure/impulse-dependent rock resistance. No fully validated authentic model exists yet. Report: research/full-experimental-report/report.pdf (19 pages), report.html, evidence-v3, per-row CSVs, source refresh receipt. Fresh HTTP200 downloads match glass, tennis and rock source fingerprints. 107 collision records (24 glass, 8 ball/surface summaries, 75 rocks) plus17 rolling points; not124 independent fully characterized events.

Fixed glass comparator RMSE0.036674/0.017805m/s; reconstructed spin remains nonindependent. Ball shared signed-moment conditional leave-one-surface-out RMSE1.047965 to1.242542rad/m, WORSE, rejected. Source et itself contains observed spin: no blind endpoint/material validation. Singleton moment inference is calibration only; negative rolling friction rejected. Rock historical sphere proxy remains1.004366/1.304693m/s,12.876456rad/s on25held-out records, actual mesh/I/signedspin absent. Missing rolling constants for3tennis specimens estimated0.016438/0.017675/0.231073 on9points, tested on8; source curves fitted to allsamepoints are reproduction comparators, not independently held-out.

New rolling.py isolated sustained branch passes200controls at3ell scales. New contact_memory.py keeps rigid bodies, adds small elastic history/stiffness/damping and an implicit-midpoint contact solve including independent angular impulse. contact-memory-v2 passes100planar+100spatial body-energy/scaling controls,20dependent-direction cases and second-order exact-oscillator convergence. Only bilateral/elastic branch: opening, Coulomb/plastic transitions, evolving frames and material calibration remain unimplemented. Do not claim production adoption or empirical improvement. v1 stores initial absolute-symmetry-tolerance rejection. Local Python timing45.4us is not engine benchmarking.

Active article adds full-velocity direction projection of friction capacities and physical normal transfer, normalized rolling resistance and contact-local deformation equations. Bold vectors, hat directions, scalar delta components, Delta combinedP, singleindices, symmetric baseline wedge, double-struck inertia/matrixdigits remain. Numerical evidence is in the separate report. Literature supports contact-local compression/shear/asymmetric pressure; public G10-forceplate, granite bounce and attached-ball mode data have different boundaries and must not be mixed. No pressure-polynomial correction is adopted. All13rapid/irregular baseline remainsunqualified;2xgate removed. Production adapters remain unchanged and reject rolling/twisting.

# Rolling resistance and estimation policy — October 6, 2026

Latest user permits a small fit for missing coefficients, superseding the prior prohibition on project fitting. Documented parameters remain fixed; estimated parameters must be identifiable, physically bounded and separate from held-out validation. Article adds mu_r with physical moment length a_r and coordinate-normalized capacity mu_r a_r/ell, plus sustained no-slip rolling/static-reaction derivation. Do not transfer microsphere coefficients to macroscopic rubber/rocks. General mixed angular closure and source-slip convention remain unqualified; production adapter still rejects nonzero rolling/twisting. Full experimental evidence report is now requested; include source readiness and genuine mismatches rather than hiding them.

# Combined P uses uppercase Delta — October 6, 2026

User explicitly prohibits lowercase delta on combined uppercase P. Active article now writes Delta P_c = [delta p; delta L/ell] for combined contact impulse and Delta P for body/global changes. Lowercase delta remains only on linear/angular impulses and scalar directional components; never on combined P. This is the latest convention and supersedes the broad earlier statement that all impulses use lowercase delta. Static/dynamic friction section is being added in the same checkpoint.

# Static and dynamic friction restored explicitly — October 6, 2026

User requires mu_s (static) and mu_d (dynamic) in the article. Active symbolic derivation and separate input table now include both, scalar static capacity and slip-opposing dynamic force, conditional fixed-direction/single-branch impulse forms, and the dynamic component row [mu_d,1,0,0]. Integration over actual sliding intervals is distinguished from a whole-impact shortcut; slip arrest/reversal and positive tangential restitution require the stated original branch/history law. Static s is distinguished from spin-direction s. No coefficients invented, empirical results restored or production changes claimed. Primary reference: Drake dry-friction/CoulombFriction documentation. The source-slip convention versus full-relative-velocity direction compatibility remains explicit.

# Scalar impulses and unit-direction hats — October 6, 2026

User correctly clarified that delta p_n, delta p_t, delta L_s, delta L_n are scalars, not vectors. Active article reconstructs full bold impulses as scalar times unit direction; extra q_n/q_t/a_s/a_n amplitude symbols are removed. Hats mark normalized n, t, s. Unknown component vector contains these scalar impulses directly (angular entries divided by ell). Identity and zero matrix blocks now use true double-struck digits; ordinary scalar entries remain numeric. A small locally bundled, permissively licensed BBOLD Type 1 font supplies these glyphs, with build.sh local search paths; AMS mathbb I remains the inertia font. Named matrices remain in their prior style pending any broader preference.

# Wedge product spacing — October 6, 2026

User requests more separation after a wedge matrix before the following operand. Active article inserts LaTeX thin space (backslash comma) at wedge-matrix products, without printing commas. Internal r/wedge spacing remains the same on both sides; transpose superscripts stay attached to the matrix.

# Impulse delta versus state-change Delta — October 6, 2026

Latest user reserves lowercase delta for impulses only. Active article uses uppercase Delta for body/stacked momentum increments, body/world angular-momentum changes and kinetic-energy changes. Contact impulses retain delta p, delta L, delta P_c and their directional components. Explicit before/after change definitions and global momentum balances included. Bold vector preference and single indices remain active.

# Bold vectors restored — October 6, 2026

User changed preference: vectors are bold again. Active article restores bold linear/angular/combined vectors and directions, retaining plain scalar components, planar angular scalars, single body index k, relative sums, symmetric baseline wedge, transpose-only reversal and double-struck inertia I. This supersedes the earlier plain-vector instruction. Numerical results remain omitted.

# Single body index and relative sums — October 6, 2026

User rejects paired or stacked subscripts. Active article uses one body index k; relative contact quantities are unindexed sums with signed body contributions. Contact velocity u_k and combined contact motion U_k replace double c,k subscripts. Body pairs are represented by a sum, not k,j labels. Planar component display suppresses body indexing. Plain symbols (no bold vectors) are retained.

# Plain symbols and body indices — October 6, 2026

User requests k for body indexing instead of a/A and plain vector symbols instead of bold. Active article uses k and j for body pairs, plain vector/matrix symbols throughout, and W_d for component mobility instead of A. Admissibility is calligraphic C; distributed traction uses sigma as index. Baseline wedge, symmetric spacing, transpose-only rule, double-struck inertia I and omitted numerical results remain unchanged. This is a presentation change, not new physics implementation.

# Symmetric wedge spacing — October 6, 2026

User requests equal vector/wedge spacing on either side. Active article macros now group each entire matrix symbol as one math atom and use the same fixed 2mu internal gap in r-wedge and wedge-r. This prevents TeX binary-operator spacing from making the two symbols asymmetric. Transpose-only semantics and double-struck inertia I remain unchanged.

# Dimension-independent transpose and inertia notation — October 6, 2026

Latest user explicitly removes the wedge-reversal minus-sign rule: use only transpose, wedge r = transpose of r wedge, for both 2D and 3D. Active article removes both negative-matrix and swapped-operand identities. Cross-product matrix entries retain their necessary signs; body/contact orientation still has physical signs. Inertia uses double-struck I (LaTeX mathbb I), not E or ordinary bold I, throughout tensor and scalar planar/sphere expressions. Numeric results remain omitted.

# Full symbolic article rewrite — October 6, 2026

User requested full rewrite using baseline wedge, lowercase delta/indices and combined uppercase quantities; explicitly requested skipping numerical results pending redo. Active article.tex/pdf completely rewritten: r wedge denotes the cross-product matrix, wedge r its transpose; no surrounding parentheses in matrix chains. Directions t and s follow relative contact velocity and relative angular velocity. Components delta p_n, delta p_t, delta L_s, delta L_n. Upper/lowercase artifacts in user dictation are incidental. Full force+angular impulse, body-versus-contact angular impulse, 2D/3D symbolic examples, explicit mobility, constrained component map, energy and matrix-free shared-contact evaluation included. Material tables remain symbolic. All prior numbers and figures removed from active PDF; artifacts preserved. Exact directional constitutive closure still requires recovery: no native implementation or new empirical validation claimed. Active README distinguishes this scope from archived documentation.

# Wedge notation correction — October 6, 2026

User explicitly requests wedge on cross-product matrices, no redundant parentheses in associative matrix products, and lowercase delta for impulses. Active article uses r_subscript-wedge j and analogous matrix products, preserving factor order. This notation correction does not resolve the directional constitutive model mismatch documented below.

# Directional model mismatch — October 6, 2026

Latest user correction supersedes claims of model fidelity: t is the relative contact-velocity direction; s is the relative angular-velocity direction. No p_s component. Existing conventional force-only endpoint comparisons do not validate this model. Earlier research-assessment.tex already includes an independent free angular impulse. Latest full-wrench article still lacks the exact directional constitutive law. Read research/scaled-contact-article/MODEL-FIDELITY-AUDIT.md before further implementation. Preserve previous results as historical comparators. Recover original equations; do not guess remaining direction, projection, zero-state or capacity conventions. No corrected numerical validation yet.

# Independent contact torque correction — October 6, 2026

User correctly identified the missing independent torque impulse. Article now uses delta L = r cross j + k, with k the free contact torque impulse excluding the force lever-arm contribution. Full scaled wrench maps, paired-body coupling, energy, moving-boundary torque work and planar scalar moments are derived. Central normal axial-spin example demonstrates force-only spin unchanged versus prescribed torque changing spin. This is illustrative algebra, not a fitted or measured material law. Educational matrix_tools.py includes the general wrench map. Latest wrench-impulse-v2 evidence checks 100 random 3D pairs plus 100 planar controls, scaling/reference-point invariance and energy/duality, with maximum energy error 4.27e-14 J. v1 is retained preceding evidence. Production solver and measured comparisons are unchanged: independent rolling/torsional moment prediction remains unimplemented and requires physically justified, independently documented inputs. Do not claim native full-wrench validation.

# Literature novelty review — October 6, 2026

User requested literature search for notation uniqueness and publishable contribution. Closest primary precedent: Vose/Umbanhowar/Lynch RSS2011, appendix footnote1 PDFp7, explicitly scales planar angular velocity by radius of gyration and torque by its reciprocal. Zhang/Su/Liao2014 section3.3 eq21 scales spatial deformation/moment similarly. Exact momentum symbol pairing not found, but dual impulse integration is established; do not claim novelty from that absence. Normal/tangential restitution, coupled Delassus, matrix-free evaluation and dissipation issues have substantial prior art (Chatterjee/Ruina1998, SIGGRAPH2022, Acary/Collins-Craft2025). Review and publication assessment in research/literature-novelty-review/report.pdf and review.json; derivation article updated with closest references. No completed new research contribution established; attributed tutorial is plausible. Strongest proposed research route is independently characterized fixed material inputs predicting held-out full motion, with shape/inertia/uncertainty and established model comparisons. No new fitting, production changes, submission or 2x gate. GAUGE August2026 preprint added as recent related benchmark/candidate validation source; complete signed tangential restitution and spin-state suitability unverified, no GAUGE run performed. Original source PDFs cached outside repo at /home/viktor/.cache/physics-literature-novelty-20261006.

# Article 3D example and prior art — October 6, 2026

User requested an explicit worked3D example and asked whether the notation is new. Article now includes a rotated cuboid/plane corner impact with non-diagonal world inertia, full6component state, full3x3 contact coupling, impulse, outgoing translation/spin and energy tables plus geometry/motion figure. Prescribed inputs are separate in worked_3d_inputs.json; independent unscaled impulse/energy checks and reference-length/tangent-basis invariance are retained in research/scaled-contact-article/worked-3d-v1/. This is a frozen algebra example, not experimental or native trajectory validation. Negative tangential restitution convention is clarified. All fitted rock comparisons and the old conditional deviation map are moved to a historical appendix; documented24glass comparisons remain in the main article. Primary-source review finds established spatial-vector duality and characteristic-length normalization; do not claim new mechanics or notation priority. Sources and scope are recorded in notation-prior-art.json. Production physics and material profiles remain unchanged.

# Active realism policy — October 6, 2026

Latest user rejected fitted parameters as the route to authenticity. Stop geometry/Pareto parameter searches. Select documented material-pair profiles and independently evaluate unchanged source values. Do not offer legacy fitted rock coefficients or assumed rubber friction as published material properties. New catalog research/documented-materials/catalog.json contains11complete published triples; pair loader material_profiles.py requires explicit profile IDs and preserves conditions/uncertainties. Cornell fresh/spent glass and Caltech dry Zerodur profiles available. Missing limestone/concrete signed contact parameters and Cross2010 same-pair friction remain unavailable.

# Current user scope override — October 6, 2026

The user explicitly removed the 2x performance gate. Focus on realistic behavior and a comparison table against measured reality. Historical 2x plans below are superseded; physical/accuracy requirements remain. Do not claim independent material validation from fitted or target-supplied restitution.

# Physics-engine continuation

## Active user request — October6,2026

Finish larger irregular2D/3D cases and make EVERY rapid-friction example pass original model/accuracy gates. Then freeze a working baseline and seek >=2x end-to-end performance improvement on EACH case, preserving full physics and accuracy. Continuously push live checkpoints. Exact13-case scope and prospective timing rules: research/rapid-friction/performance-gate-plan.json. Earlier fine-reference speed ratios are not the requested new2x gate. Baseline is not yet qualified: original rotating/irregular examples and new larger scenes retain genuine failures. Six larger ladders from published6b63cb2 are COMPLETE:20 full histories/10 genuine rejections (nine velocity, one translation-only position), no qualified references. Independent state/provenance/rejection audits pass. Preserve outputs; never restart over existing files. Isolated exact-component recovery pilots and rotating planar discovery trials are declared under research/large-contact-recovery and research/rapid-friction/planar-discovery-plan.json. Solver/accuracy remediation precedes performance claims.

## Latest independent checkpoints — October6

Six additional125-sphere full histories pass independent physical and frozen
provenance audits. Finer312.5/156.25/78.125ns ladder fails BOTH spin edges
0.0347795/0.0353242rad/s (quarter limit0.025). Stricter1e-11 numerical search
at1.25/.625/.3125us passes first edge but fails second spin0.0355317.
Original physical gates remain unchanged; neither ladder qualifies. Evidence:
research/large-sphere-finer and research/large-sphere-strict-search.

Fresh component19264-hull world still declines later396-row contact at0.02s;
independent accepted-prefix geometry/pose/energy and actual rejection checks
pass. Preserve component192 binary snapshot before changing production source.
Joint planar prototype10 full histories pass physical gates but all8 refinement
edges fail; independent archive audit passes. Prototype remains isolated.

Hosted CI required393 replay FAILS despite local acceptance. Both hosted
sixteen-seed and stricter six-seed variants also decline393;162/297 controls
pass. Downloaded immutable diagnostic artifact37398436313 retained under
research/ci-contact-portability/hosted-37398436313 (generated binaries excluded).
Do not remove required gate or claim portability. Full-island planar prototype
8 analytical controls pass; original rotating group ladder COMPLETE:9 full physical passes,7 failed refinement edges; initialdisk
level declines actual-body residual1.0000048e-10 above1e-10. Retain all real
failures. Global13-case working baseline and requested per-case2x remain OPEN.

## Production component192 checkpoint — October6

The actual135-row component now passes the production helper and independent
original-law gate at9.745044e-9m/s. Production cap192 is selected with the same
warm plus six seeds and2048 SVD/iteration limits PER search. Default projection
tail remains64 rows; normal pressure384 and position512 remain unchanged.
All23 original replays pass, all23 newhelper bypassed, prior22 bytes/counters
exact.163 engine tests and11 native checks pass. CI includes actual396 capture.
Evidence: research/component-cap192/production396-independent.json, live23/receipt.json.

The three125-sphere refinement histories are COMPLETE and independently pass
all physical/provenance gates. First edge passes; second spin0.03614876rad/s
exceeds the original quarter-budget0.025rad/s. Reference remains unqualified.
The cap128 executable is preserved for their historical audit, which passes.
Next fresh histories must preserve every failure and qualify both edges before
any per-case timing acceptance. The13-case global baseline/2x gate stays OPEN.

## Further original-scene evidence — October6

Component128 fresh64-hull1.25us retry clears393 rows but rejects later396 rows
at the same0.02s output prefix; the failing component has135 rows. Independent
prefix geometry/inertia/pose/energy and actual rejection checks pass. Isolated
component192 accepts this exact396 system at9.745044e-9m/s;162/297/393 controls
still accept,423/276 still decline. Production128 remains unchanged until the
active125-sphere refinement source/binary guard releases. Keep six seeds.

Isolated temporal hard-contact Float64 Box2D study completes10 rotating planar
histories and passes all physical gates, but fails every refinement edge. It
preserves polygon-core mass and exact libm authored rotations; upstream rounded
mass and Bhaskara approximation would change the original authored model.
Independent signed per-impulse boundary-work controls, both body orders,
angular momentum identities and observer state parity pass. Archives/audit:
research/temporal-hard-contact/. Never adopt or claim qualified/performance.

The276-row exact30-component face-angle search and diverse null-seed control
still decline; their outputs remain retained. The423 Python root is accepted
again by the original zero-budget gate, but its large null-pressure search is
not a robust integrated solver. The first two fresh125-sphere histories complete
and pass physical gates; their first refinement edge passes. The finest third
history remains running. No all13 working baseline or requested2x acceptance.

## Component128 recovery checkpoint — October6

The null-traction helper now supports exact components up to128 rows, retaining
warm plus six seeds and2048 iteration/SVD limits per search. Default projection
tail remains64 rows. Actual393-row finer64-hull rejection contains a126-row
component: production and independent original-law gates accept it at
5.5972095e-9m/s. All23 contact replays pass, prior22 bytes/counters exact,
163 engine tests and11 native checks pass. CI includes this actual capture.
See research/component-cap128/production393-independent.json and live23/receipt.json.

The original six-seed cap128 pilot accepts162/297/393 but declines423/276.
Sixteen seeds gives the same outcomes and is not adopted. Two fresh64-hull
histories from source66d0f2a clear earlier failures but stop later: 10us at0.01s
on276 rows,1.25us at0.02s on393 rows. Both accepted-prefix physical audits pass;
neither is a full history. Their preserved source66 runner is recorded in
research/large-velocity-world/binary-snapshot.json. Preserve its initial audit;
its live guards are historical after the cap128 build.

The joint1um/2048 velocity/128 position-iteration planar control completes all
five original9-polygon histories but fails every original refinement edge.
Finite-difference CMINPACK and native trust-region pilots retain their declines;
one trust-region162 root passes instantaneous gates only. All13 references
remain unqualified as a set, and the requested all-case2x gate remains pending.
New scaled minimax contact search is isolated under research/minimax-contact;
no adoption or performance claim is justified before its original-law checks.

## Native larger velocity fallback — October6

Bounded mobility-null traction seeds now integrate after ALL existing lanes
including the projection tail decline. Full4096/component64 rows, exact graph,
warm plus six seeds, fresh2048 iteration/SVD caps PER search. Right derivative
at zero cone radius is isolated to this helper; old projection default unchanged.
Original eager-cone production gate accepts actual162/297-row saved failures.
423-row125-hull velocity capture still declines. All23 original replays bypass
the helper, all prior22 bytes/counters exact;163engine+10contact+11native checks,
two actual-failure regressions,390-row position and no-LAPACK control pass.
Read research/large-contact-recovery/README.md and validation/receipt.json.
Source/binary guards for the earlier completed125-hull run are now historical;
a preserved88438b8 runner and frozen Git blobs recheck that provenance.

Five finer original9-polygon histories through39.0625ns also fail every original
refinement edge. Rounded-union seam prototype retains default state bytes but
fails refinement and one energy gate; neither prototype is adopted. Archives
and independent audits remain in research/planar-refinement and rounded-boundary.
Fresh trajectories and all13-case reference qualification remain necessary
before the requested2x per-case performance gate.

## Latest larger-case remediation — October6

All30 larger reference attempts are sealed at e945742:20 histories/9 velocity
rejections/1 translation-only position rejection, zero qualified references.
Independent archive audit now covers189 retained records including15 rotating
planar slop/seam controls; those controls also fail original accuracy edges.

Position recovery is now integrated with512 normal rows only, same128 active
states, original target/mobility/bounds/absolute1e-8 gate. Velocity normal-pressure
limit remains384. Actual390-row rejection now passes full native pipeline at
3.55e-15m/s; independent geometry/primal witness passes. LP success flag alone
failed direct slack validation and is not the witness. Regression tests163/35
subtests+10contact+11native checks, live23 and exact prior22 velocity bytes/counters
pass. Read research/large-position-recovery/README.md. Archived original binaries
are preserved and historical source hashes checked through their frozen Git pin.

Fresh original125-hull finest trajectory at integrated88438b8 completes all
192000 updates/0.12s and passes original full physical gates. Independent saved
geometry/energy/source/runtime audit passes. See world-results/final.json and
world-independent-audit.json under large-position-recovery. This is one full
history, still without both adjacent refinement edges or a2x performance claim.
Elapsed is descriptive because concurrent research/brief profiling was allowed.

Large velocity pilots: projection/direct/component and FB merit all decline.
Null-traction seeds plus bounded LM/TRF find strict original-law roots for162-row
and423-row captures; unchanged native budget0 independently accepts both. The
297-row capture's failing21-row component still declines after single/pair removal,
inactive-face and cone-traction probes. All outputs remain under
research/large-contact-recovery/. Do not integrate these exploratory roots as a
solver or claim full histories; further bounded sliding-face/root/discovery work
and fresh trajectory qualification are needed before sealing the13-case baseline.

## Current Ubuntu continuation — October 5, 2026

Remote pushes through a799038 were merged; production projection-tail integration is published at 8c7065eb8219cb0735aa3e742aa82b99521a26c1. Live23 original-law checks pass with prior22 response bytes/counters preserved. Current verification: 163 engine tests and35 subtests, ten contact-model tests and eleven native checks pass.

The six original adaptive hull lanes retain three full histories and three later original-law rejections, with zero qualified references. Rapid-friction research now qualifies translational +/-20m/s shaking with mu0.4 over0.12s: 2D9/25 disks achieve23.07x/15.65x native median gains against qualified fine references; 3D27 spheres achieves7.58x. Every timed repeat passes unchanged gates and repeats state bytes. FullFloat64 planar qualification uses explicit1um numerical slop, authored geometry/material unchanged; supported builder and exact settings are documented in research/rapid-friction/README.md and verified-settings.json. Default planar slop remains unchanged. Independent archive audit passes for144 retained records, including actual failed trials.

Rotating groups, mixed polygons, boxes and hulls remain unqualified. Next work must resolve those original trajectory accuracy failures; do not relax gates or promote the successful disks/spheres to general shape qualification. Cache-reset/tighter-cache prototypes did not qualify and were not adopted. Read research/rapid-friction/README.md, report.pdf and independent-audit.json before extending; preserve all failed outputs. No physical rubber calibration, VKF port or compiler/GPU acceptance is claimed. Historical sections below retain source pins and are superseded where they say integration/benchmarks are pending. Direct authenticated git push currently works.

User requested a push and handover on October5,2026 to continue in another chat. Continue on `svenviktorjonsson/rigid-body-collisions`, branch `research/adaptive-benchmark-validation`. Root coordinates commits/publication; do not merge main or force-push. User authorizes frequent pushes and parallel research/review. Keep original material and physical/accuracy gates; preserve actual failures and source provenance.

## Verified production and rubber-like examples

Live production source is **bca35c3103a78c731b37ea8a467e5fc71d13e8aa**; later checkpoints add evidence and a corrected test fixture. The bounded projection-tail proposal below is NOT integrated.

- 162 engine +10 contact-model regressions pass. Native mechanical checks and all22 retained original-law contact replays pass. Independent no-LAPACK/API/rotating-wall review passes17 checks. A corrected1kg opposing-wall fixture passes its11-test friction suite; the earlier body-level density fixture error is retained.
- Combined translation repair subtracts actual accepted final physical contact motion, including angular/warm/external/gyro increments, from every desired position normal rate. This fixes double consumption of clearance; actual1kg ball control changes rightgap−49.999µm to+50.001µm, samevx.95m/s/K.45125J, no new numerical pose displacement.
- Hull margin is set before recalculating cached AABBs;8 independent geometry controls pass. Contact discovery intentionally changes, so solver endpoint parity is not legacy world-trajectory identity.
- `early_component_recovery=False` remains default. Opt-in early component search uses the actual first256 rejected iterate and unchanged original full-row gates. Isolated66 replies pass, all22 defaults are exact,17 early successes/5 declines; declined work is retained and caps are fresh/separate from the later tail.
- Elastic sphere/fixed-plane prototype: **18 scenarios/54 histories/0 rejections**, both twist signs reverse,5 alternating floor/ceiling impacts,3 same-floor back-and-forth bounces undergravity. Stored elastic energy and independent angular couple impulses are audited. Lowcapacity controls correctly retain spin. These are synthetic material parameters, not measured rubber; native arbitrary-body/many-contact elastic couples remain unverified.

Paper/visuals: `research/elastic-completion/report.pdf`, `floor-ceiling-preview.gif`, `same-floor-preview.gif`; `research/elastic-completion-visuals/demo.html` and typeset figures. User dislikes raw LaTeX in chat: show typeset figures or plain readable values.

## Dense-shape accuracy is still unresolved

Predecessor frozen source **108a9bb4c7899f75d760b27b179cc56557904a08**: `research/hull-gap-completion` completes6 histories/0 rejects/0 qualified references. Individual physics/contact/position/ledger gates pass; both original refinement edges fail in both scenes. The old74-row bounded position-system inconsistency certificate remains valid for its old targets; it is not physical geometry or Coulomb-law infeasibility.

Frozen live-source **bca35c3**: `research/hull-combined-completion/RESULTS.md` retains6 uninterrupted attempts, **5 full histories/1 actual42-row velocity rejection/0 qualified references**. All5 completed lanes pass individual gates. Eight-body seed42 middlelevel rejects at residual3.102141818e−5m/s after0.08s, blocking both original refinement edges. Twenty-seven-body seed7301 completes all3 but fails every quarter-budget edge: position.0940804/.0736703m, velocity5.78974/6.67731m/s, spin82.8643/82.5177rad/s, orientation1. 34852/1.16538rad. Do not skip the rejected level or promote accepted contact equations into trajectory accuracy.

All bca study archives and independent full-source/current-runtime/physical/prefix/geometry/inertia/pose-ledger audits are sealed and published; guards released. Three numerical changes were declared together. Costs are descriptive, not a controlled speed ranking. No CCD is active for these top-level compound hull bodies; contact/manifold timing remains a separate accuracy issue.

## New42-row numerical recovery: evidence and current running check

Capture: `research/hull-combined-completion/results/rejections/fast_shake8_hulls42/reference_1.json`, SHA256 `53e45c4004be16d58ab921cc6ed360af93342d06170fc5e28e519231f381914c`.

`research/new-combined-contact-review/run-20261005T194211Z` finds strict original-law roots with SciPy LM/TRF and unchanged native budget0 acceptance. Near-redundant normal/tangent response directions agree to roundoff, not bitwise equality. A numerical face-seed experiment also accepts with originalPGS, but its generic native relation proposal is unexecuted and not chosen for integration.

Chosen helper research: frozen **170d798b113863dc4bee3df515bb8ab63175ea21**, `research/new-combined-projection-review/projection_more_v2.h`. A warm original-projection/column-scaled More search on≤64 fullrows with fresh **2048SVD/2048iteration** caps accepts the new42 at **7.6059686e−9m/s**,1283SVD/steps, passivity−3.60146098J. Nine controls pass. All four1024-budget variants and an exact-component/global1024 trial genuinely fail and remain archived. No merit/scaling superiority is established; greater bounded numerical effort is the supported explanation.

Both cheap independent audits pass:

```
python research/new-combined-projection-review/audit_failed_trials.py
python research/new-combined-projection-review/audit_v2.py
```

The **23-input baseline/candidate preservation run is COMPLETE**. It executed46 planned replies from exact published **9e546715d14a5b0468f984f3b5f18eff02dfb6dd**. Candidate23/23 pass native and independent original-law checks; baseline22/23 accepts and retains the new42 rejection. All prior22 impulse/response arrays have identical Float64bytes including signedzero, old counters match and new helper is bypassed. The new42 candidate tail uses890SVD/iterations at residual2.424537764e−9m/s; the1283-call standalone result used a different explicitly recorded seed.

Outputs/receipts are `.../23-results/`, summary `23-results/summary.json`, logs `23-run.stdout/stderr`, compile receipts `23-compile*`; signed-zero audit is in the results directory. The separate stricter `23-independent-audit.json` passes all122 checks: finite-energy, all original physical laws, caps and source/executable/compiler/library guards. Root sealed `23-completion-manifest.json`; the isolated-run source/runtime guard is released. No native trials remain active. Production still has no projection tail.

**First next-chat action:** inspect the final strict audit and guard report, then review the unapplied production proposal below. Candidate acceptance is instantaneous-system evidence, not production or trajectory qualification. Do not rerun completed drivers over existing artifacts. If an environment is replaced, preserve immutable outputs and build prospective integration in a new tree; old runtime attestation is archival evidence rather than a claim about newly installed libraries. Candidate helper changed only include paths/namespace alias from exact170. All native23 simulation work is done; remaining work is integration, verification and fresh histories.

## Concrete next steps after23 proof

`research/projection-tail-integration/proposed/` contains **unapplied/uncompiled** reviewed-by-root draft production files, with manifest and README. Do not treat them as implementation acceptance. Review against current production, then integrate ONLY after23 proof/guard release:

1. Copy validated helper to `spatial_backend/projection_more.h`; insert its tail after every existing velocity recovery lane declines, using actual final rejectedPGS seed. Preserve accepted old22 endpoints. Independent original full projection/bounds/finite-passivity gate remains mandatory. No material/tolerance changes.
2. Add LAPACK-enabled `projection_tail_checks` CMake/CI target, Float64/Bullet/LAPACK links; update Python numerical metadata and23 live replay validator. Draft runner adds final/atomic-prefix `projection_tail_policy`: stage=`after_all_existing_pipeline_failure`, compiled/enabled, caps64/2048/2048 and actual attempts/solves/declines/SVD/iteration/Newton counters. Root draft additionally rechecks original residual and normal bounds before applying impulses. Compile/test all affected native targets, live23 replays and appropriate engine regressions; independently review.
3. `research/hull-search-completion` and `research/audit_hull_search_completion.py` prepare the next **six unchanged full trajectories**. Common/scenes/material/gravity/wall schedules/quarter gates remain exactlyoriginal52/bca. Fourteen earlier archive pins stay immutable. A separately declared bounded numerical-search tail is the only additional search change. Protocol remains PENDING until reviewed integration, BUILDREADY,23default proof, finalized evidence gates/config (these are verification conditions, not requests for new user permission) and an exact **published40-character source SHA**. Then run all6 exactlyonce, retain all rejects/prefixes/ledgers, audit full provenance/runtime/physics and BOTH edges. Never claim convergence without them.
4. Update README/strategy/draftPR and paired organization handovers from actual outcomes. DraftPR: https://github.com/svenviktorjonsson/rigid-body-collisions/pull/1 (still needs final scope refresh). Do not merge it.

## Publishing and organization handover

Directgitpush historically returns401. Exact nonforced GitData publication:

```
python /tmp/publish-physics-commit.py HEAD
```

Wait for a publisher to finish before movingHEAD. If a new environment lacks this helper, use available GitHub tools/credentials or reconstruct exact GitData publication; never force references. Keep ordinary source/evidence checkpoints frequent.

Correct integration target **vektor-flow/bootstrap, working**, paired **vektor-flow/spec, working**. Latest paired docs-only checkpoints: spec **2a29d03aab6cd29b98b6730d38b309d7b1fa0950**, bootstrap **21d153e561aa9d3b00ea32fe0f8318b376c0259e**. Bootstrap owns full handover; spec is a short pointer. Their organization branches advanced to consolidated **0130unverified compiler candidate** during this work; preserve latest handover/compiler/fixture/patch bytes. Read current AGENTS/HANDOVER and fetch/fast-forward before editing; publish spec first then exactpin bootstrap. This research does not implement a VKF port or qualify Section0/main/release. October5 GPU resumption is documented in their latest handover; do not substitute older deferral prose or initiate hardware work for this physics handover.

HostedCI: paired source108/e2/evidence579 runs passed. Current bca Python3.11 jobs pass; Python3.12 jobs were cancelled without a failed step; rerun-failed-jobs was requested for run37364665223. Newer pushes are queued. Check actual latest job conclusions; do not claim all current hosted matrices green.

Publication fallback at final handover: the shellGitHub credential expired during
the last upload and environment-status refresh did not renew it. GitHub connector
GitData tools succeeded: create_blob → create_tree → create_commit → update_ref
with force=false. Verify every blob and the complete tree against local Git, then
verify remote branch and reconcile the identical local tree. Connector commit
metadata can produce a different commitSHA: final sealed evidence initially local
4aa139e1539473fbfde9443b24fa6ef340bb4222 was published canonically as
a237f91ba332ce13f5414d8ff9b97c56a7de87c1 with identical tree
a08899dcfb43f755b731c98e2bdfac4742433ddb. The local original is preserved on
archive/local-handover-4aa139e. Frozen research sources170/9e/bca are unaffected.
Use the GitHub connector if the old shell publisher still returns401; do not ask
for approval to repeat the already-authorized nonforced publication.

## Numerical continuation controls — October6

Isolated frictionless-to-original friction homotopy tests393/latest396/276/423.
Initial numerical negatives near roundoff fail exact normal bounds; retained
first outputs. Fresh cone-cleaned candidates are rechecked with unchanged
original equations, bounds, passivity and zero-budget native gate. All four
full systems still decline. Never apply intermediate altered-friction roots
to bodies. Evidence:research/friction-continuation/results and results-v2.

## Mobility search breakthrough — October6

Python and bounded native diagonal-regularization continuation recover original
393/276/423 saved systems with moderate impulses; latest396 still declines.
Numerical intermediate matrices NEVER become physical body mobility. Native
17 stages,2048 iteration/SVD limits EACH, exact192-row components; only final
original zero-budget gate plus independent original gates accepts. Python final
max residuals2.21e-9/1.84e-17/1.42e-15; native4.41e-9/9.95e-9/1.88e-9.
Native remains ISOLATED. Hosted comparator declared in required CI diagnostics;
production required393 null-seed test remains unchanged and currentlyfails.
Local Clang19 and GCC15 accept all earlier portability controls, so compiler
alone has not explained hosted GCC13/LAPACK behavior. All records preserved.
Do not edit native helper/portable driver or spatial sources while hosted run
is collecting diagnostics; snapshot nativebinary and original192source exist.
All-case accuracy/baseline/2x task remains OPEN; do not claimworldcompletion.

## Hosted native continuation reproduced — October6

Artifact37400982980/11384849365 reproduces native original-law393/276/423
acceptance on hosted GCC13/LAPACK and latest396 decline. Artifacts archived
under research/ci-contact-portability/hosted-37400982980; binaryexcluded.
Production integration can now proceed only after ALL existing lanes decline,
with authoritative original zero-budget gate before any body writes. Required
393 physical acceptance must remain mandatory; old bare null-only search is
known nonportable, so validate actual combined production pipeline. Preserve
original bare failure evidence. Local and hosted saved roots are instantaneous
proofs, not full histories or reference/performance qualification.

## Production numerical-mobility tail integrated — October6

After ALL old lanes including null-traction seeds decline, a new bounded
search-only diagonal continuation tail supports full4096/exact192 components,
17 fixed numerical diagonal levels,2048 iteration/SVD caps PER stage. Original
Coulomb coefficient, physical mobility, original absolute tolerance/bounds/
passivity remain unchanged; authoritative original zero-budget eager-cone
gates are mandatory before atomic candidate/body application. Existing23
replays are ALL byte/counter exact and bypass newhelper;163 engine tests and
11 native checks pass; no-LAPACK direct build/replay passes without BLAS links.
Full production replays accept393 (newtail4.41e-9),276 (existinglane6.10e-9),
423 (newtail1.88e-9) and earlier396 (existinglane9.91e-9); latest396 still
rejects1.23e-7. Required CI393/earlier396 now exercise COMPLETE production
pipeline and same original physical acceptance, with276/423 added mandatory.
Bare null-only393 portability failure stays archived, not hidden or relabeled.
These are saved systems, not full-world qualification or2x gate completion.
Validation:research/mobility-continuation/integration/. No active world runs
currently; declare source/plan before fresh64/125-hull histories.

## Large-component cap/support correction — October6

Three new full-horizon hull attempts stop later:64-h10us279rows,125-h10us
759rows,125-h5us693rows. All accepted-prefix/rejection/geometry/pose audits
pass; no full histories. Snapshot original3afb runner/replay before rebuilding.
759 contains an ALREADY accepted249-row component, but the cap previously
stopped before finding failed57 rows. Nonlinear search cap remains192;
accepted larger components now pass ORIGINAL zero-budget gate without search.
Production759 accepts4.54e-9.693 has failed645-row component but only93 active
rows; new final large-only reduced-support tail uses full4096/reduced192,
8 support passes,17 fixed diagonal stages/2048 iteration+SVD limits PER stage,
retaining zero impulses outside selected triples as SEARCH ONLY. Every original
full component/global eager-cone/bounds/passivity gate mandatory; production
693 accepts6.86e-9. Existing23 bytes/counters EXACT, allnewtailsbypassed;
163engine/11native tests pass. Latest396 stilldeclines. Saved279 surprisingly
accepts full fresh old polisher at1.24e-12; initial contrary expectationreceipt
remains retained beside corrected independent receipt. It is not a fullworld.

Hosted facc054 production/engine/conformance checks pass on BOTH Python3.11/3.12;
job3.12 fails ONLY later research comparator namespace redefinition after
production integration. Raw failure artifact37402546398 retained; portable
comparator now builds separately renamed frozen prototype source, preserving
canonical archived header bytes; local controls393/276/423 pass,396declines.
Keep all13 reference/baseline/2x gates OPEN.

## Terminal rejected-seed component polisher — October6

Final bounded tail after every existing lane declines: full4096 exact components,
nonlinear search192, original256 PGS then ONE unchanged256SVD-cap existing
polisher per failed component. Original eager-cone/bounds/residual/passivity
component/global gates mandatory before atomic writes. Existing23 saved endpoints
and counters remain byte-exact, helper bypassed.163engine/11native checks pass;
optional no-LAPACK build/replay passes without BLAS/LAPACK runtime dependencies.
Validation research/mobility-continuation/integration-v3; actual8 replays retain
latest396 decline. New prospective original125-sphere tight1pm numerical slop
ladder retains strict1e-11 search and ALL original physics/quarter-budget gates.
New isolated planar simultaneous-contact experiment: CCDoff only enabled,
retain both prepared points, internal2.5e-11 search/final1e-10 gates unchanged.
Eight analytical controls and disabled original state parity pass. Both studies
require published source freeze before full worlds. All13 baseline/2x gates OPEN.

## Hosted checkpoint passes; further numerical studies retained — October6

Run37405326424 forfaf8e95 passes BOTH Python3.11/3.12 including renamed frozen
research portability comparator; artifact11386722895 archived with file hashes,
research binaries excluded. New checkpoints retain running hosted jobs instead
of canceling evidence collection. Current production terminal restart source6414
passes23 exact replays/163engine/11native/optional no-LAPACK; complete actual8
original gates pass except latest396, intentionally retained as decline.
Fresh125-h10us passes earlier759 then fails792rows at accepted prefix.03s;
failing492 component selected60 support rows, caps not exceeded, searchdeclines.
Fresh125-h5us clears earlier693 repeatedly then stops906rows at prefix.06s.
Full original prefix/geometry/rejection audits retained.64-h10us ongoing: terminal
restart actually clears earlier279; guardedsource/binary still FROZEN.
Sphere tight1pm slop full first2 levels passphysics, firstspin edge.03628rad/s
FAILS original.025. Finest level ongoing. Pure100000-sweep PGS withoutNewton
also declines latest396/later792; no adoption. Two complete planar studies
(simultaneousCCDoff/bothpoints and follow-up1pm slop) each10 full physicsPASS,
ALL8 accuracy edgesFAIL; independent archive auditsPASS. New isolated zero-position
projection test declared, all original geometry/energy/quarter gates unchanged,
8 analytical controls and disabled-original state parityPASS. All13 working
baseline/2x gates OPEN; no performance ranking from these descriptive timings.

## Mixed sliding/sticking guide integration and latest full worlds — October6

New FINAL recovery tail after every old lane including terminal component polish:
exact full components4096, selected physical192, weak12, LPvariables384,
constraints4096,4096 LPcalls and100000 SHARED pivots PER full solve;128 direction
iterations/face,32 INNER cone facets guide only. Original EXACT Coulomb circle,
eager cone, bounds, absolute residual and finite passivity component/global gates
mandatory before atomic application. Search allowance .5*originaltol never changes
physical gate. Cumulative LP counters unsigned64. No added library dependency;
standard two-phase simplex attribution/license included. All23 old endpoint bytes
and original counters EXACT, helper bypassed;163engine/11native pass; original22
replays pass, optional no-LAPACK original gate passes with no BLAS/LAPACK dependency.
Complete actual8 captures now ALL accept, latest396 through new tail at7.7827e-9
(34LP/1130pivots, moderate maximpulse.18995). Later792/906 remain genuine declines;
906 exhausts shared pivot cap. Integration proofs:research/mixed-face-native/integration.
Hosted prior2fbe checkpoint37409918312 BOTH jobsPASS, artifact11389126819 archived;
new integration needs its own hosted latest396 required replay. Main fetch has no
unincluded commits. Keep PR1 draft and all13 working-baseline/2x gates OPEN.

Frozen6414 production64-h10us full .12s completes and passes ALL physics. Finer
64-h5/h2.5 histories stop with393/294-row genuine rejects; independent outcome
audit authenticates reused10us anchor and both accepted prefixes/rejections.
New native mixed-face helper also declines both finer captures; retain them.
125-h10/h5 still stop later792/906. Tight1pm125sphere first ladder edgesFAIL/PASS;
next finer full physicsPASS but newest edgeFAIL. BOTH original edges remain required.
Small current8box/hull and analytic-clock studies each6 full physicsPASS/all4
accuracy edgesFAIL. Planar simultaneous/tight-slop/no-position/cold-start and
rounded-union studies each10 full physicsPASS/all8 accuracy edgesFAIL, independently
audited, no adoption. Rounded-union analytical8/disabled-original bytesPASS.
All authored scenes/material/horizons and original gates preserved. These elapsed
times are descriptive, not accepted performance. New isolated bounded strong-contact
release search declared; no root/world claim before native original-gate proof.
Next:published final-source full hull histories, all-case accuracy qualification,
then freeze baseline and pursue >=2x EVERY case with five alternating end-to-end
repetitions, original gates per repeat, and continuous ordinary pushes.

## Rubber/rock calibration and completed guarded studies — October6

User added public real-impact calibration for natural rocks and different-sized
rubber balls, including incoming/outgoing linear/angular motion and energy.
Public Chant Sura and Tschamut archives downloaded to external cache; provenance
and extraction inventories retained, raw data not redistributed. Chant includes
3D linear and gyro components, but coordinate alignment/attitude/contact normals
need verification; ideal concrete shapes are not natural-rock specimens. Tschamut
has natural scans/masses and scalar rotational speed, not verified full angular
vectors. No exact real-impact replay or fitted friction claimed.
Rubber primary manuscripts verify 46mm/46.4g and 58mm/103g Superball experiments;
no clean matched-compound diameter sweep verified. Derived 2002 energy ledger
and prospective calibration protocol recorded in research/rubber-ball-calibration.
Grip/deformation may require compliant tangential physics beyond rigid Coulomb.

Planar analytic clock:10 full histories physicsPASS, all8 refinement edgesFAIL;
independent archive auditPASS. New integrated mixed-face production worlds:
64-h5/h2.5 reject393/294rows,125-h10 rejects792rows; prefix/rejection auditsPASS.
64-h10 still running, guarded spatial source/binary/runtime FROZEN; publish only
completed outcomes. Strong-contact release search also genuinely declines these
later/finer captures; no adoption. Hosted df3fbc5 both jobsPASS; new integration
checkpoint ec9e2dc still running. ALL13 baseline/2x gates remain OPEN.

Final5c75fa9 original64-h10us world now COMPLETE .12s/physicsPASS;939.744s
concurrent descriptive time, not performance measurement. Final independent audit
covers full64-h10 plus genuine393/294/792 rejects and accepted prefixes, PASS.
Spatial source/binary/runtime guard remains authenticated; no world is active now.
All13 accuracy-baseline/2x gates remain OPEN. Public Chant inventory inspects82
CSV files,41 valid gyro/energy files; mixed resultant/component angular units and
energy-implied masses recorded. Inferred energy-formula inertia is not measured
inertia; full coordinate/contact inputs still unverified, no friction fit claimed.

Rubber size/spin synthetic controls COMPLETE:24 cases,46/58/100mm diameters,
2D/3D,peripheral spin−2/0/1/2m/s,declared uniform masses/disk-vs-sphere inertia,
independent normal/tangential impulse+energy checks. Current production3D and
isolated planar rounded-union joint solver ALL24PASS,maxstateerror1.0502e−9.
Original planar comparator9cases miss tangential response at first contact;
all failures and initial NumPybool serialization error retained. This is not a
rubber calibration, production2D/global-world acceptance or performance claim.

## Independent material prediction requirement — October6

User clarified material/contact data must predict measured motion; no fitting to
same validation endpoint. Native rubber adequacy comparison now COMPLETE7runs,
58mm103gSuperball/granite,Cross2010TableI. μsweep0/.05/.1/.2/.4/.9/1.5 with
measured normalrestitution.78 prescribed. Native rigid-law independent impulse
and energy auditPASS; experimental spin FAIL:predicted maxspin factor10.4093
vs14.9±.1,even26deg uncertaintymax10.7973<14.8. Tangentialrestitution0vs.49±.01.
Normal.78 matches by construction,NOTindependentmaterialprediction. Same-pair
friction/measuredinertia absent,fixed14kggranite approximation explicit. No fittedμ.
Retain initial KeyError reporting failure and all outcomes. Concrete gap requires
rubber tangential elasticity/dissipation and nonzero restitution+isotropicfriction;
current Coulomb lane still e0 only. No completeexperimentalvalidation/all13baseline
or2x claim. All prior authored cases/gates unchanged, continuous branchpushes.

## User's two-channel restitution restored — October6

User clarified BOTH normal/tangential restitution are CENTRAL to model. Prior
zero-tangent Coulomb adequacy comparison was a scope mistake; do not present it
as failure of user's complete model. Native3D Coulomb opt-in explicit normal/
tangential endpoint targets now implemented,sharedcontact/fullgraph/freegyro,
circularimpulsecap plus separate REALkinetic-minus-wallworkgate before application.
Both run APIs accept normal_restitution/tangential_restitution; isolated2D
simultaneous whole-island build auto-selected for explicitcoeffs. Unsupportedold
binary acknowledgement rejects instead of silently ignoring inputs. Defaults
retainlegacyphysics/outputselection; old24controls Float64bytesEXACT.
144 native controlsPASS (bothcoeffs,3radii,3spins,2caps),3offcentre orientedbox
controlsPASS fullinertia impulse/energy/contacttargets. Optional3Dpercontact
point/basis/velocity/impulse diagnostics ready for non-spherical data. Coefficients
currently uniform,NO fittedcontactpoint/directionmap. FournewregressionsPASS;
11manualnativechecksPASS. CTest has NO registered tests,not counted as nativePASS.
163 priorengine testsPASS; final167engine tests ALLPASS after restoration.
Measured58mm rubber with BOTH.78/.49 predicts spinfactor15.5099central25deg;
14.927–16.088 over24–26deg overlaps observed14.9±.1. Normal/tangentinputvalues
fromsameimpact,μ.9 hypothesis,homogeneousinertia/fixedplane approximation; not
independentmaterialprediction. All historiesandcomparisonsretained. Rockparser
imports2219before/after scalarrotation impacts74tests,massjoined2219; signed3D
omega/contactframes and independentcoeffs absent,NO fake rockvalidationclaim.
All13accuracybaseline andrequested2x gates remain OPEN. Continuous ordinarypushes.

Actual rock comparison extended with Wang2018 public limestone/concrete75impact
workbook,10/20cmdiameters. IMPORTANT:their Rt is COMtangent velocity ratio,not
our signedcontact-slip e_t. Reconstructed published energyformula agrees1.11e−16;
conditional sphere-point angularimpulse approximation matches only6/75 within
reportedincoming spin bound3rad/s. Exactfacetedshape/contactorientations absent:
this identifies insufficientsphereapproximation,NOTfullshape-modelfailure orproof
that materialcoefficientsdependonlocation. Twelve COMRn>1 mustnotbe interpreted
as intrinsiccontact restitution>1. Rawpubliczipandpaperexternalcache;download
hashes/sourceattributionandallcomparisonrecordsretained. InitialguessedPDF404,
actualpublisherzipdownloadPASS,repeated workbookheader importererrorcorrected
andretained. Independentrockfullstatevalidation remains OPEN.

## Full restitution comparison report — October 6, 2026

Full standalone HTML/PDF and plots: research/restitution-validation-report/report.html and report.pdf. Rebuild script and report-audit.json retain checks and input hashes. Rock fitting uses 50 training/25 held-out impacts; fixed e_n=.492986, e_t=-.191582, mu=.811484 are effective sphere-fit coefficients, not measured material properties. Height-holdout angle-law improvement9.7% reverses to9.1% worse under angle-group CV; do not promote variable material coefficients. Native75/75, synthetic fixed-rest geometry96/96, rubber native36/36 pass. Three of four rubber surface spin intervals overlap; Superball pad remains16.34 predicted vs18.2+/-0.1 measured. No independent friction or actual rock mesh/inertia validation. Hosted restoration run37415426120 succeeded; newer report runs still pending/in progress. All13 working baseline and per-case2x gate remain OPEN. Initial48 planar reporting errors retained separately; scalar cross-product reporter fix yields final96 passes without solver changes.

## Contact realism development — October 6, 2026

Completed isolated passive shear plus horizontal rocking spring experiment under prospectively pushed plan9fb76df. Four leave-one-surface-out tests worsen spin RMSE0.999 to1.235rad/m (+23.7%); pad worsens16.343 to15.841 vs18.2. No production adoption.36 analytical controls and40 independent integration/energy checks pass, integrated budgets only, instantaneous patch not verified. Fixed-lever shear compliance cannot change endpoint spin at supplied restitution. Existing normal-axis twist is wrong axis for this discrepancy. Derive finite patch/support deformation with independent load/inertia data before further law adoption. Evidence: research/shear-rocking-contact; full report updated.2x requirement remains removed.

## Article and extended contact diagnosis — October 6, 2026

User explicitly requested parallel article; one article agent is working only research/scaled-contact-article. User notation: bold uppercase L is angular momentum, lowercase script ell is length scale. Combined velocity (v,ell*omega), momentum (p,L/ell). Keep associative matrix chains without commuting them. Expanded Cross2010 conditional comparison now includes four golf-ball cases and four Superball cases, retained under research/contact-moment-identification. Both pad cases miss spin uncertainty; golf/string case also misses under homogeneous inertia. Missing normal-force offsets are inferred only, never measured/forward applied. texlive-latex-base installed for article PDF; production model remains unchanged after failed shear/rocking candidate.

## Symbolic article and practical matrix calculations — October 6, 2026

Completed user-requested parallel article under research/scaled-contact-article/article.pdf and article.tex. Uses angular momentum bold L and lowercase script ell, symbolic equations; numerical material values separated into input tables. Includes2D/3D scaled-duality/contact/restitution/friction/work derivations, matrix-free coupled contact action,8ball comparisons,held-outlimestone anddatasetavailability tables,experimental-vs-synthetic figures.200coordinatechecks pass;100shared-contact dense/matrixfree comparisons max2.84e-14;actual2D/3D wrapperexamplesexactlyreproduce retained histories. FinalLaTeXbuildtwice clean with no warnings/overfull. No productionphysicsadoption from rejectedcomplianceprototype;reportandarticlehonestlyretainmismatches.2x remains removed.

## User visualization replacement — October 6, 2026

User rejected bar charts and requested a single2D comparison map. Active article replaces bothbars with figures/deviation-map.png/pdf and a symbolic derivation of observable signature q=(abs(vn),abs(vt),ell*abs(omega)), ell=reportedradius policy. Coordinates are angle between measured/predicted signatures (not spatialheading) and relative normerror%;matchorigin.33dots:25held-out limestone+8ball/surface summaries, colored10/20cmrocks/Superball/golf. Interactive deviation-map.html supports inspection; JSONretainsallvectors/limits. Ballcomponents are reconstructedusingreportedrestitution/spin, notindependentfullstate; uncertaintyen/et/incidence/spin cornerenvelopesnotCI. Nojitter;sharedcoordinatesallowed. Native/syntheticchecksexcluded. Metricsdependonscalingpolicybutphysicalpredictionsdo not. ArticleLaTeX twiceclean;33count/metricbounds/JavaScriptsyntaxauditsPASS. Historicalbarsretainedoutsideactivearticle.

## Documented material inputs checkpoint — October 6, 2026

No further fitted-parameter/Pareto execution authorized by latest steering.11source-backed complete profiles under research/documented-materials/catalog.json/html:9Cornell selectedspecimen/pair entries and2Joseph/Hunt2004air-Zerodur entries. material_profiles.py selects explicitprofileIDs and oneisolated3Dpair, en/et fixed; en/et/mu/size/density/context sourced, no guessedfallback. Native81/81analytic+energychecks;source/provenance81sceneauditand4wrong-contextrejectionspass. Initialnewadaptermu²error retained and corrected:sqrt(mu)perbody forBulletproductmixing;Box2Dusesdifferentmixing. Educational3Dwrappercorrected;2D/3Dexamplehistoriesstillbyteexact.

ActualCornellglassbinaryworksheet24recordreproduction usingpublished(.97,.44,.092) unchanged: normalRMSE.03667399m/s,relativeCOMtangentRMSE.01780477m/s,normalizedtranslationjoint.02415634.24native/analytic/energychecks pass. Sourcecontactgt/spin reconstructed byangularmomentum; notindependentangularmeasurement. Publishedchart/worksheetmay sharecharacterizationtrials,notindependentvalidation. NominalRused;rawrecordradiivariations noted. Plateworksheetmetadata differs fromcatalogandnotcomparedblindly. SourcePDFs/XLSoutsidepublicrepo;SHAretained. New24dotmaphollowforinferredspin;oldrockfitteddotsremainhistorical,notsource-profilepredictions. Articleaddsdocumentedprofiletable/resultsandlabelslegacyfits.2x remains removed;actualmesh/independent-materialtransfer/openallcaseaccuracynotclaimedcomplete.
