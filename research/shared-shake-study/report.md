# Corrected shared-contact shaking: accuracy and cost

The prospectively declared source is `38f407d12208654075c07315e96b9fc213612b91`. All nine attempts completed; both fixed-step refinement edges qualify, every timed repetition passes physical/residual gates and full trajectory budgets, and repeated histories are bitwise identical.

The scene contains 27 one-kilogram spheres in a container prescribed to shake at plus/minus20 m/s for0.12 s, with gravity9.81 m/s squared, pair friction0.4 and zero restitution. The common-point convention is applied to complete contact rows, warm angular impulses and boundary work. Parameters are synthetic.

| Setting | Native seconds, all repetitions | Median seconds |
|---|---|---|
| Fixed1.25 microsecond reference | 22.768746, 23.051115, 19.541006 | 22.768746 |
| Motion-based travel guard0.015 | 3.261182, 3.173906, 3.051779 | 3.173906 |

The native median ratio is **7.17 times**. Native timing includes stepping/state recording; whole-process timings are retained separately. Sequential interleaved launches reduce ordering bias but external machine load is uncontrolled.

| Fixed-step refinement | Position RMS, m | Velocity RMS, m/s | Spin RMS, rad/s | Orientation RMS, rad |
|---|---|---|---|---|
| fixed\_5us->fixed\_2\_5us | 8.3173862e-06 | 0.00056231781 | 0.015178209 | 0.00034093656 |
| fixed\_2\_5us->fixed\_1\_25us | 6.1742988e-06 | 0.00052242843 | 0.010987465 | 0.00023586459 |

Candidate versus finest reference: position2.2303661e-05 m, velocity0.0015628478 m/s, spin0.052144196 rad/s, orientation0.0011607172 rad.

Full budgets:5 mm,0.05 m/s,0.1 rad/s and0.01 rad; reference edges use one quarter. These are declared error tolerances, not a demonstrated asymptotic convergence order. This result concerns one synthetic scene and horizon; it does not qualify random hulls, authentic rubber, general CCD, online error control or Vektor-native execution. The earlier6.11-times separate-endpoint result is preserved as historical evidence.

Audit: `python -m research.audit_shared_shake`. Exact source, plan, binary hash, all raw histories and individual times are retained in `results/`.
