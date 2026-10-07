"""Prospective auditor controls, no native execution or historical source edits.

Existing native records are read-only fixtures; their use is not a selected-tail
trajectory result. Synthetic modifications exercise rejection of malformed data.
"""
import copy,hashlib,importlib.util,json,tempfile
from pathlib import Path
from research.audit_hull_search_completion import (
    ROOT,STUDY,FIELDS,CHANGE,audit_plan,ledger,position_projection,count,
    prefix_record,validate_model,rejection_capture,refinement_receipt,library_metadata,linked_paths,ledger_agreement,recovery_config,recovery_metadata)
from research.audit_shared_hulls import audit as historical_audit


def rejects(call):
    try:call()
    except (AssertionError,KeyError,ValueError,TypeError,IndexError):return
    raise AssertionError('Malformed or undeclared evidence accepted')


def main():
    checks=[];plan,_,pending=audit_plan()
    if pending:rejects(lambda:audit_plan(require_ready=True))
    else:audit_plan(require_ready=True)
    checks.append('Selected strategy remains blocked until integration/build/default proof; preceding bca archive is sealed')
    with tempfile.TemporaryDirectory(prefix='combined-auditor-controls-') as name:
        temporary=Path(name)
        rejects(lambda:audit_plan(require_ready=True))
        for mutate in (
            lambda p:p['common'].update(early_component_recovery=False),
            lambda p:p['common'].update(position_stabilization='split_translation_gap'),
            lambda p:p['declared_numerical_change'].clear(),
            lambda p:p['retained_numerical_changes'].pop('shape_cache_margin_order'),
            lambda p:p['physical_gates'].update(energy_change_minus_boundary_work_J=2.),
            lambda p:p['physical_gates'].update(container_surface_excess_m=.003),
            lambda p:p['trajectory_budget'].update(orientation_rad=.02),
            lambda p:p['common'].update(contact_tolerance_m_s=1e-7),
            lambda p:p['common'].update(contact_slop_m=1e-8),
            lambda p:p['scenes'][0].update(fractions=[.06,.03,.01]),
            lambda p:p['scenes'][0].update(duration=.11),
            lambda p:p.update(candidate_repetitions=1),
            lambda p:p['preceding_protocol_artifact_hashes'].update({'invented':'0'*64}),
            lambda p:p['numerical_recovery'].update(stage='before_existing_pipeline'),
            lambda p:p['numerical_recovery'].update(approved_caps={'max_rows':64})):
            bad=copy.deepcopy(plan);mutate(bad)
            (temporary/'plan.json').write_text(json.dumps(bad));rejects(lambda:audit_plan(temporary))
        checks.append('Only declared bounded-search tail added; original policies/duration/lanes/slop/contact/energy/containment/quarter gates and14 exact archive pins retained')
        record=dict(zip(FIELDS,[2,.01,-3.,5.,[1.,-2.,2.],4.]))
        assert ledger(record)==record
        for key,value in ((FIELDS[0],True),(FIELDS[0],-1),(FIELDS[1],-1.),
                          (FIELDS[2],float('nan')),(FIELDS[3],2.),(FIELDS[4],[1,2]),(FIELDS[5],2.)):
            bad=dict(record);bad[key]=value;rejects(lambda:ledger(bad))
        zero=dict(zip(FIELDS,[0,0.,0.,0.,[0.,0.,0.],0.]));assert ledger(zero)==zero
        zero[FIELDS[2]]=1.;rejects(lambda:ledger(zero))
        for malformed in (True,-1,1.,float('nan')):rejects(lambda:count(malformed))
        for residual in (float('nan'),float('inf'),-1.,1.001e-8):
            rejects(lambda:position_projection(dict(translation_split_solves=3,translation_split_residual_max_m_s=residual),1e-8))
        # Producer bookkeeping control uses synthetic gates only; this is no native trajectory.
        spec=importlib.util.spec_from_file_location('combined_producer_control',STUDY/'runner.py')
        producer=importlib.util.module_from_spec(spec);spec.loader.exec_module(producer)
        producer.diagnostics=lambda scene,result,half:{key:0. for key in plan['physical_gates']}
        producer.errors=lambda a,b:{key:0. for key in plan['trajectory_budget']}
        good=dict(coulomb_residual_max_m_s=0.,translation_split_solves=1,translation_split_residual_max_m_s=0.,contact_point_policy='shared',numerical_model={'contact_point_policy':'shared'})
        test_scene={'container_interior_half_extents_m':[1.]}
        test_runs={f'reference_{i}':copy.deepcopy(good) for i in range(3)}
        assert producer.qualify(plan,test_scene,test_runs)['reference_qualified']
        for badvalue in (float('nan'),-1.,1.001e-8):
            badruns=copy.deepcopy(test_runs);badruns['reference_1']['translation_split_residual_max_m_s']=badvalue
            assert not producer.qualify(plan,test_scene,badruns)['reference_qualified']
        badruns=copy.deepcopy(test_runs);del badruns['reference_1']['translation_split_residual_max_m_s']
        assert not producer.qualify(plan,test_scene,badruns)['reference_qualified']
        badruns=copy.deepcopy(test_runs);badruns['reference_1']['translation_split_solves']=True
        assert not producer.qualify(plan,test_scene,badruns)['reference_qualified']
        checks.append('Finite ledger/vector/triangle/zero-update, strict projection/counter AND producer position-eligibility controls')
        saved=dict(native_prefix=record,final=record,prefix_available=True,final_available=True)
        ledger_agreement(record,record,saved)
        altered=dict(record);altered[FIELDS[1]]=.02
        rejects(lambda:ledger_agreement(record,altered,saved))
        missing=dict(saved);missing['prefix_available']=False
        rejects(lambda:ledger_agreement(record,record,missing))
        checks.append('Final/native-prefix finite ledger equality and availability controls')
        libs={'/synthetic/lib/libc.so':'0'*64,'/synthetic/lib/ld-linux-x86-64.so.2':'1'*64}
        library_metadata(libs)
        rejects(lambda:library_metadata({'/synthetic/lib/libc.so':'0'*64}))
        rejects(lambda:library_metadata({'relative/libc.so':'0'*64,**libs}))
        rejects(lambda:library_metadata(dict(libs,**{'/synthetic/lib/libbad.so':'invented'})))
        linked='libc.so => /synthetic/lib/libc.so (0x1)\n /synthetic/lib/ld-linux-x86-64.so.2 (0x2)\n'
        assert linked_paths(linked)==set(libs)
        checks.append('Portable runtime manifest includes ALL recorded libraries and loader; malformed hash/path or missing loader rejected')
        expected={'adapter.py','native/helper.h'};payload={'adapter.py':b'python source','native/helper.h':b'helper source'}
        hashes={k:hashlib.sha256(v).hexdigest() for k,v in payload.items()}
        def source_control(contents,manifest):
            assert set(contents)==set(manifest)==expected
            assert all(hashlib.sha256(contents[k]).hexdigest()==manifest[k] for k in contents)
        source_control(payload,hashes)
        bad=dict(payload);bad['native/helper.h']=b'changed';rejects(lambda:source_control(bad,hashes))
        bad=dict(hashes);del bad['native/helper.h'];rejects(lambda:source_control(payload,bad))
        checks.append('Exact source-file set and byte/hash controls reject missing helper or changed native source')
        scenes=json.loads((ROOT/plan['baseline_scenes']).read_text());scene=scenes[plan['scenes'][0]['id']]['scene']
        fixture=json.loads((ROOT/'research/hull-combined-completion/results/progress/fast_shake8_hulls42/reference_0.json').read_text())
        authored,receipt=prefix_record(scene,fixture,plan);assert receipt['trajectory_qualified'] is False
        for mutate in (
            lambda p:p.update(completed_output_frames=True),
            lambda p:p.update(collision_updates=-1),
            lambda p:p.update(boundary_work_J=float('nan')),
            lambda p:p.update(coulomb_residual_max_m_s=1.001e-8),
            lambda p:p.update(translation_pose_ledger_updates=True),
            lambda p:p['states_backend'][0][1].__setitem__(7,1.),
            lambda p:p['wire_bodies'][1]['principal_inertia'].__setitem__(0,999.)):
            bad=copy.deepcopy(fixture);mutate(bad);rejects(lambda:prefix_record(scene,bad,plan))
        partial=copy.deepcopy(fixture)
        partial.update(states_backend=partial['states_backend'][:2],times=partial['times'][:2],completed_output_frames=1,complete=False)
        _,partial_receipt=prefix_record(scene,partial,plan)
        assert partial_receipt['prefix_observation_only'] and partial_receipt['trajectory_qualified'] is False
        physically_bad=copy.deepcopy(fixture);physically_bad['boundary_work_J']-=1e6
        _,bad_receipt=prefix_record(scene,physically_bad,plan)
        assert bad_receipt['diagnostics']['energy_change_minus_boundary_work_J']>plan['physical_gates']['energy_change_minus_boundary_work_J']
        assert bad_receipt['trajectory_qualified'] is False
        checks.append('Native principal/authored conversion and full tensor energy controls; partial or failed physical prefix never qualifies')
        # Synthetic approved config exercises a future metadata contract. It is NOT approval
        # of the production helper and never changes the pending actual plan.
        fixture_plan=copy.deepcopy(plan)
        cfg=fixture_plan['numerical_recovery']
        cfg.update(status='ROOT_APPROVED_FROZEN_NATIVE_CONFIG',preexecution_checks={key:True for key in cfg['preexecution_checks']})
        fixture_plan['preparation_status']='FINALIZED_FOR_ROOT_PUBLISHED_EXECUTION'
        recovery_config(fixture_plan,True)
        for key,value in [('max_rows',True),('max_iteration_steps',0),('max_svd_calls',-1),('max_rows',65)]:
            bad=copy.deepcopy(fixture_plan);bad['numerical_recovery']['approved_caps'][key]=value
            rejects(lambda:recovery_config(bad,True))
        for field,value in [('stage','before_existing_pipeline'),('wire_flag','invented_flag'),('installed_helper',None),('default_enabled',None)]:
            bad=copy.deepcopy(fixture_plan);bad['numerical_recovery'][field]=value
            rejects(lambda:recovery_config(bad,True))
        for key in cfg['preexecution_checks']:
            bad=copy.deepcopy(fixture_plan);bad['numerical_recovery']['preexecution_checks'][key]=False
            rejects(lambda:recovery_config(bad,True))
        model=copy.deepcopy(fixture_plan['common']);model['contact_recovery']={'enabled':True}
        model.update(contact_point_policy='shared',shape_cache_margin_order='margin before recalc',travel_fraction=.06)
        model[cfg['numerical_model_key']]=dict(compiled=True,enabled=True,stage=cfg['stage'],**cfg['approved_caps'],
            attempts=1,solves=1,declines=0,svd_calls=1283,iteration_steps=1283,newton_steps=1283)
        success=dict(numerical_model=model,contact_point_policy='shared',scalar_precision='float64')
        validate_model(success,fixture_plan,.06)
        for mutate in (
            lambda v:v['numerical_model'].update(shape_cache_margin_order='recalc before margin'),
            lambda v:v['numerical_model'].update(early_component_recovery=False),
            lambda v:v['numerical_model'].update(position_stabilization='split_translation_gap'),
            lambda v:v.update(scalar_precision='float32'),
            lambda v:v['numerical_model'][cfg['numerical_model_key']].update(stage='before_existing_pipeline'),
            lambda v:v['numerical_model'][cfg['numerical_model_key']].update(max_rows=65),
            lambda v:v['numerical_model'][cfg['numerical_model_key']].update(compiled=False),
            lambda v:v['numerical_model'][cfg['numerical_model_key']].update(enabled=False),
            lambda v:v['numerical_model'][cfg['numerical_model_key']].update(attempts=True),
            lambda v:v['numerical_model'][cfg['numerical_model_key']].update(declines=1),
            lambda v:v['numerical_model'][cfg['numerical_model_key']].update(svd_calls=2049),
            lambda v:v['numerical_model'][cfg['numerical_model_key']].update(iteration_steps=2049),
            lambda v:v['numerical_model'][cfg['numerical_model_key']].update(newton_steps=1284)):
            bad=copy.deepcopy(success);mutate(bad);rejects(lambda:validate_model(bad,fixture_plan,.06))
        checks.append('Selected tail exact helper/caps/stage/compile/recovery/counter contract enforced; synthetic readiness cannot replace actual integration/build/default proof')
        snapshot=dict(schema='normal-only-position-rejection-v1',phase='position_translation',A=[[1.]],b=[1.],p=[0.],lo=[0.],hi=[1e30],dependencies=[-1],residual_m_s=1.,tolerance_m_s=1e-8,internal_dt_s=.001,iteration_budget=4096)
        rejected={'rejected':'Translation-only position projection failed; repair initial overlap or refine timestep'}
        rejection_capture(snapshot,rejected,plan)
        for mutate in (
            lambda p:p.update(schema='historical-schema'),lambda p:p.update(phase='position'),
            lambda p:p.update(dependencies=[0]),lambda p:p.update(residual_m_s=0.),
            lambda p:p.update(tolerance_m_s=1e-7),lambda p:p.update(A=[[float('nan')]]),
            lambda p:p.update(iteration_budget=True)):
            bad=copy.deepcopy(snapshot);mutate(bad);rejects(lambda:rejection_capture(bad,rejected,plan))
        bad=copy.deepcopy(snapshot);del bad['A'];rejects(lambda:rejection_capture(bad,rejected,plan))
        checks.append('New translation reject requires finite symmetric pure-normal matrix; no historical missing-capture exception')
        authored['physical_setup_id']='synthetic-identical-control'
        runs={f'reference_{i}':copy.deepcopy(authored) for i in range(3)}
        eligible={k:True for k in runs};assert refinement_receipt(plan,runs,eligible)['reference_qualified']
        for lane in runs:
            badeligible=dict(eligible);badeligible[lane]=False
            assert not refinement_receipt(plan,runs,badeligible)['reference_qualified']
        changed=copy.deepcopy(runs)
        for frame in changed['reference_2']['states']:
            for i,mass in enumerate(changed['reference_2']['mass']):
                if mass:frame[i][0]+=.001251
        answer=refinement_receipt(plan,changed,eligible)
        assert answer['edges'][0]['passed'] and not answer['edges'][1]['passed'] and not answer['reference_qualified']
        checks.append('Both exact quarter-budget edges required; missing/rejected/partial lane cannot be skipped')
        for directory,receipt_name,policy in (
            ('hull-active-completion','independent-audit.json',None),
            ('hull-translation-completion','final-independent-audit.json','split_translation')):
            study=ROOT/'research'/directory;receipt_path=study/receipt_name
            before=receipt_path.read_bytes();source=json.loads((study/'results/summary.json').read_text())['execution_source_commit']
            output=temporary/(directory+'.json')
            historical_audit(study,source,position_stabilization=policy,output_path=output)
            assert output.read_bytes()==before and receipt_path.read_bytes()==before
            checks.append(directory+' original receipt reproduced BYTE IDENTICALLY; historical auditor unchanged')
        from research.audit_hull_combined_completion import audit_evidence as combined_evidence
        combined_study=ROOT/'research/hull-combined-completion'
        combined_path=combined_study/'final-independent-audit.json'
        combined_bytes=combined_path.read_bytes()
        combined_report=combined_evidence(combined_study,plan['preceding_protocol_source_commit'],require_all=True)
        assert (json.dumps(combined_report,indent=2,allow_nan=False)+'\n').encode()==combined_bytes
        assert combined_path.read_bytes()==combined_bytes
        assert combined_report['attempt_count']==6 and combined_report['history_count']==5 and combined_report['rejection_count']==1
        assert not any(v['reference_qualified'] for v in combined_report['scenes'].values())
        checks.append('Sealed bca six-attempt5history1reject0qualification receipt reproduced BYTE IDENTICALLY; historical combined auditor unchanged')
    print(json.dumps(dict(schema='prospective-search-auditor-controls-v1',checks=checks,
                         native_execution=False,search_trajectory_executed=False,
                         fixture_scope='Existing historical/native records read only plus synthetic malformed controls; not bounded-search policy evidence',
                         historical_sources_or_receipts_mutated=False),indent=2))

if __name__=='__main__':main()
