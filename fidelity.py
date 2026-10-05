"""Offline accuracy-gated fidelity selection for unchanged physical setups.

References require adjacent refinement checks before candidates can be selected.
This certifies only supplied scenes, observables and horizons, not arbitrary
inputs or online adaptation. Failed qualification returns no accuracy-backed
choice rather than silently increasing budgets or treating the finest run as truth.
"""
from research.run_rigid_study import errors, normalized_error, BUDGET, REFERENCE_BUDGET
import math


def select(reference_runs, reference_key, edges, candidates, *, budget=None, reference_budget=None, costs=None):
    budget = BUDGET if budget is None else budget
    reference_budget = REFERENCE_BUDGET if reference_budget is None else reference_budget
    for values in (budget,reference_budget):
        if set(values)!=set(BUDGET) or any(not math.isfinite(v) or v<=0 for v in values.values()):
            raise ValueError('Positive finite position, velocity and spin RMS budgets required')
    if not edges: raise ValueError('At least one adjacent reference refinement edge required')
    refinements=[]
    for a,b in edges:
        error=errors(reference_runs[b],reference_runs[a])
        refinements.append({'from':a,'to':b,'errors':error,
                            'normalized_error':normalized_error(error,reference_budget)})
    if any(r['normalized_error']>1 for r in refinements):
        return {'status':'unqualified_reference','choice':None,'refinements':refinements,'comparisons':[]}
    reference=reference_runs[reference_key]; comparisons=[]
    for label,run in candidates.items():
        error=errors(reference,run)
        cost=run['engine_and_controller_s'] if costs is None else costs[label]
        if not math.isfinite(cost) or cost<0: raise ValueError('Finite nonnegative measured cost required')
        comparisons.append({'candidate':label,'errors':error,
            'normalized_error':normalized_error(error,budget),
            'cost_s':cost})
    passed=[c for c in comparisons if c['normalized_error']<=1]
    best=min(passed,key=lambda c:c['cost_s']) if passed else None
    return {'status':'qualified' if best else 'no_candidate_within_budget',
            'choice':best['candidate'] if best else None,
            'refinements':refinements,'comparisons':comparisons}
