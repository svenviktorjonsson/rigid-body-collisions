"""Derived checks from Cross 2002 Table I/II; not raw experimental trials."""
import json
import math
from pathlib import Path


def derive():
    mass, radius, alpha = 0.0464, 0.023, 0.40
    speed, angle, spin_before, spin_after = 2.69, math.radians(36), 0.0, 98.4
    horizontal_ratio, restitution = 0.59, 0.91
    inertia = alpha * mass * radius**2  # Authors' tabulated model; not measured tensor.
    incoming = [speed * math.cos(angle), -speed * math.sin(angle)]
    outgoing = [horizontal_ratio * incoming[0], -restitution * incoming[1]]
    translation_before = 0.5 * mass * sum(v*v for v in incoming)
    translation_after = 0.5 * mass * sum(v*v for v in outgoing)
    rotation_before = 0.5 * inertia * spin_before**2
    rotation_after = 0.5 * inertia * spin_after**2
    # Sign convention: positive spin reduces bottom-point horizontal velocity.
    return {
        'source': 'https://physics.usyd.edu.au/~cross/Gripslip.pdf',
        'source_tables': ['I', 'II'],
        'case': 'Superball on smooth finite-mass instrumented block',
        'diameter_m': 2 * radius, 'mass_kg': mass,
        'inertia_kg_m2': inertia, 'inertia_status': 'author-tabulated homogeneous-sphere model',
        'experimental_relative_errors_typical': [0.02, 0.03],
        'velocity_before_m_s': incoming, 'velocity_after_m_s': outgoing,
        'spin_before_rad_s': spin_before, 'spin_after_rad_s': spin_after,
        'translation_before_J': translation_before, 'translation_after_J': translation_after,
        'rotation_before_J': rotation_before, 'rotation_after_J': rotation_after,
        'ball_total_energy_change_J': translation_after + rotation_after - translation_before - rotation_before,
        'bottom_velocity_before_m_s': incoming[0] - radius * spin_before,
        'bottom_velocity_after_lab_m_s': outgoing[0] - radius * spin_after,
        'rigid_replay_qualified': False,
        'cautions': [
            'Velocity components reconstructed from rounded tabulated speed/angle/ratios.',
            'Support block moves; include its kinetic energy and velocity for system-energy/contact comparisons.',
            'Ball energy loss alone is not whole-system dissipation.',
            'No unique sliding friction coefficient follows from a gripping impact.',
            'This is an experimental observation, not an exact synthetic rigid-Coulomb oracle.'
        ]
    }

if __name__ == '__main__':
    output = Path(__file__).with_name('derived-energy.json')
    output.write_text(json.dumps(derive(), indent=2) + '\n')
