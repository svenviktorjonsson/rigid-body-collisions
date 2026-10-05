# Normal pressure-null integration checkpoint

The native helper preserves the original PSD mobility and material. It searches
numerical pressure-null directions to release redundant active constraints, then
uses a minimum-norm range step. Every accepted candidate meets the unchanged
absolute normal projection, bounds and finite passivity gate.

The final native replay passes 17 of 20 captured systems: all 16 earlier captures
and the new 291-row position system. The three later 60-, 39- and 51-row velocity
failures remain in the receipt and unresolved in this production checkpoint.
This is captured-system validation, not full-trajectory accuracy or a speed claim.

The same helper supports opt-in translation-only position repair. Recovery requires
an iteration budget of at least 64 and honors disabling contact recovery. Caps are
384 normal rows and 128 active states/SVD calls per attempt. Counters disclose
search work and maximum numerical-null trial velocity change. Failure leaves the
helper output unchanged and does not establish physical infeasibility.

All 155 Python regressions and nine native normal-null controls pass, including
explicit recovery enable/disable, singular redundancy, contradictory constraints,
nonfinite input, bounds and exhausted-work controls. Existing native circular
friction, translation and active-contact controls also pass. The independent
elastic completion audit again passes all 18 cases and 54 histories with no rejects.

`source-validation.json` pins the final implementation and replay binary hashes.
`all-twenty-replays.json` includes native outputs and independent original-law and
passivity recalculation. `pre-control-change` preserves the initial replay before
the explicit enable/disable control was added; those receipts are not relabelled
as final-source evidence.
