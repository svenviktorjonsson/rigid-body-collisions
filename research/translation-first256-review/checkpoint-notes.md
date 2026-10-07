# Paused first-iteration diagnostic checkpoint

This locally source-frozen experiment started before its public plan checkpoint.
The coordinator paused it at108 of132 outputs so the source and plan could be
published before the remaining long operations. paused-prefix.jsonl preserves
that exact accepted prefix; pause-event.json records the interruption. The
in-flight native steady-clock sample includes the pause and cannot support a
normal timing comparison. Earlier output remains a captured-state diagnostic,
not a complete study or whole-engine speed claim. All attempts remain visible.
Resume only after the source checkpoint is published. Treat the full study's
timing ranking as unqualified unless a separately declared fresh experiment
satisfies its execution protocol. Physical gates and numerical work counters
can still be audited independently of elapsed time.
