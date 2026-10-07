"""Generate a reproducible 3D matrix example, independent checks and figure.

Numerical inputs are in a separate table. The example evaluates the frozen
single-contact impulse algebra, not a native trajectory or experimental fit.
Output directories are exclusive so earlier evidence cannot be overwritten.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from matrix_tools import Body, contact_map

HERE = Path(__file__).resolve().parent


def calculate(inputs, ell, tangent_angle=0.0):
    mass = inputs["mass_kg"]
    half = np.asarray(inputs["half_extents_m"])
    attitude = np.asarray(inputs["attitude_numerator"]) / inputs["attitude_denominator"]
    assert np.allclose(attitude.T @ attitude, np.eye(3), atol=1e-14)
    assert np.isclose(np.linalg.det(attitude), 1)
    inertia_shape = np.diag(mass / 3 * np.array([
        half[1]**2 + half[2]**2, half[0]**2 + half[2]**2,
        half[0]**2 + half[1]**2]))
    inertia = attitude @ inertia_shape @ attitude.T
    normal = np.asarray(inputs["normal_world"])
    # Select the actual supporting vertex, rather than an arbitrary lever arm.
    lever_shape = -np.sign(attitude.T @ normal) * half
    lever = attitude @ lever_shape
    velocity = np.asarray(inputs["incoming_velocity_m_s"])
    spin = np.asarray(inputs["incoming_angular_velocity_rad_s"])
    t1 = np.asarray(inputs["tangent_1_world"])
    t2 = np.asarray(inputs["tangent_2_world"])
    c, s = np.cos(tangent_angle), np.sin(tangent_angle)
    basis = np.column_stack((normal, c*t1+s*t2, -s*t1+c*t2))
    assert np.allclose(basis.T @ basis, np.eye(3), atol=1e-14)
    body = Body(mass, inertia, ell)
    contact = contact_map(body, lever)
    state_before = np.r_[velocity, ell*spin]
    incoming = contact @ state_before
    mobility = np.column_stack([
        contact @ body.mobility(contact.T @ axis) for axis in np.eye(3)])
    contact_mobility = basis.T @ mobility @ basis
    coefficients = np.array([inputs["normal_restitution"],
                             inputs["tangential_restitution"],
                             inputs["tangential_restitution"]])
    target = -coefficients * (basis.T @ incoming)
    local_impulse = np.linalg.solve(contact_mobility, target - basis.T @ incoming)
    assert local_impulse[0] > 0
    cone_ratio = np.linalg.norm(local_impulse[1:]) / local_impulse[0]
    assert cone_ratio < inputs["pair_friction"]
    impulse = basis @ local_impulse
    state_after = state_before + body.mobility(contact.T @ impulse)
    after_velocity, after_spin = state_after[:3], state_after[3:] / ell
    outgoing = contact @ state_after
    # Independently evaluate the unscaled physical equations with cross products.
    direct_incoming = velocity + np.cross(spin, lever)
    direct_velocity = velocity + impulse / mass
    direct_spin = spin + np.linalg.solve(inertia, np.cross(lever, impulse))
    direct_outgoing = direct_velocity + np.cross(direct_spin, lever)
    before_energy = .5*mass*np.dot(velocity, velocity) + .5*spin @ inertia @ spin
    after_energy = .5*mass*np.dot(after_velocity, after_velocity) + .5*after_spin @ inertia @ after_spin
    impulse_work = impulse @ direct_incoming + .5*impulse @ mobility @ impulse
    checks = {
        "incoming_contact_error_m_s": float(np.max(abs(incoming-direct_incoming))),
        "linear_update_error_m_s": float(np.max(abs(after_velocity-direct_velocity))),
        "angular_update_error_rad_s": float(np.max(abs(after_spin-direct_spin))),
        "outgoing_contact_error_m_s": float(np.max(abs(outgoing-direct_outgoing))),
        "endpoint_target_error_m_s": float(np.max(abs(basis.T @ outgoing-target))),
        "linear_impulse_error_kg_m_s": float(np.max(abs(mass*(after_velocity-velocity)-impulse))),
        "angular_impulse_error_kg_m2_s": float(np.max(abs(inertia @ (after_spin-spin)-np.cross(lever, impulse)))),
        "energy_identity_error_J": float(abs(after_energy-before_energy-impulse_work)),
        "scaled_energy_error_J": float(abs(.5*state_before @ body.momentum(state_before)-before_energy)),
    }
    assert max(checks.values()) < 1e-11
    assert after_energy <= before_energy + 1e-12
    vertices = np.array([attitude @ (np.asarray(signs)*half)
                         for signs in itertools.product((-1, 1), repeat=3)])
    heights = (vertices-lever) @ normal
    assert np.min(heights) >= -1e-14
    assert np.count_nonzero(abs(heights) < 1e-12) == 1
    return {
        "attitude": attitude.tolist(), "inertia_shape_kg_m2": inertia_shape.tolist(),
        "inertia_world_kg_m2": inertia.tolist(), "contact_lever_m": lever.tolist(),
        "reference_length_m": ell, "signed_lever_operator": contact[:, 3:].tolist(),
        "contact_mobility_world_per_kg": mobility.tolist(),
        "contact_mobility_basis_per_kg": contact_mobility.tolist(),
        "contact_basis": basis.tolist(), "impulse_world_kg_m_s": impulse.tolist(),
        "impulse_basis_kg_m_s": local_impulse.tolist(),
        "incoming_contact_velocity_m_s": incoming.tolist(),
        "outgoing_contact_velocity_m_s": outgoing.tolist(),
        "outgoing_velocity_m_s": after_velocity.tolist(),
        "outgoing_angular_velocity_rad_s": after_spin.tolist(),
        "energy_before_J": float(before_energy), "energy_after_J": float(after_energy),
        "energy_change_J": float(after_energy-before_energy),
        "tangential_impulse_over_normal_impulse": float(cone_ratio),
        "checks": checks, "vertices_relative_com_m": vertices.tolist(),
    }


def matrix_tex(matrix, precision=6):
    return r"\begin{bmatrix}" + r"\\".join(
        " & ".join(f"{value:.{precision}f}" for value in row)
        for row in np.asarray(matrix)) + r"\end{bmatrix}"


def write_tex(inputs, result, output):
    fmt = lambda vector: "$(" + ",".join(f"{x:.6f}" for x in vector) + ")$"
    content = r"""% Generated by worked_3d.py from the separate input table.
\begin{center}\small
\begin{tabular}{ll}
\hline
Illustrative input & Value\\
\hline
Mass $m$ & $2\,{\rm kg}$\\
Half extents $(a,b,c)$ & $(.20,.15,.10)\,{\rm m}$\\
Reference length $\ell$ & $.20\,{\rm m}$ (coordinate choice)\\
Incoming $\mathbf v^-$ & $(1,-.6,-2)\,{\rm m/s}$\\
Incoming $\boldsymbol\omega^-$ & $(2,-1,3)\,{\rm rad/s}$\\
Normal $\mathbf n$; tangents $\mathbf t_1,\mathbf t_2$ & $\mathbf e_z;\mathbf e_x,\mathbf e_y$\\
Normal restitution $e_n$ & $.6$\\
Tangential restitution $e_t$ & $.3$\\
Pair friction $\mu$ & $.6$\\
Boundary & Stationary plane, $z=0$\\
\hline
\end{tabular}\end{center}
These are prescribed illustration inputs, not a published material profile or fitted experimental values. The matrix equations below retain symbolic material coefficients. Attitude and lever-arm signs are fixed by the actual box geometry.
\begin{equation}
\mathbf A=\frac19\begin{bmatrix}1&4&8\\4&7&-4\\-8&4&-1\end{bmatrix},\qquad
\mathbf r_{\rm shape}=\begin{bmatrix}a\\-b\\c\end{bmatrix},\qquad
\mathbf r=\mathbf A\mathbf r_{\rm shape}.
\end{equation}
The supporting vertex is unique: every other vertex lies above the plane when the centre of mass is placed at $-\mathbf r$. For a homogeneous cuboid,
\begin{equation}
\mathbf I_{\rm shape}=\frac m3\operatorname{diag}(b^2+c^2,a^2+c^2,a^2+b^2),\qquad
\mathbf I=\mathbf A\mathbf I_{\rm shape}\mathbf A^T.
\end{equation}
"""
    content += "\\begin{equation}\n\\mathbf r=" + matrix_tex(np.array(result["contact_lever_m"])[:, None]) + "\\,{\\rm m},\\qquad\n\\mathbf I=" + matrix_tex(result["inertia_world_kg_m2"]) + "\\,{\\rm kg\\,m^2}.\n\\end{equation}\n"
    content += r"""All three spin components are retained. The six-component state, mass block and three-by-six contact map are
\begin{equation}
\mathbf V^-=\begin{bmatrix}\mathbf v^-\\\ell\boldsymbol\omega^-\end{bmatrix},\qquad
\mathbf M=\begin{bmatrix}m\mathbf1_3&0\\0&\mathbf I/\ell^2\end{bmatrix},\qquad
\mathbf C=[\mathbf1_3\ \mathbf R].
\end{equation}
"""
    content += "\\begin{equation}\n\\mathbf R=" + matrix_tex(result["signed_lever_operator"]) + ",\\qquad\\mathbf u^-=\\mathbf C\\mathbf V^-.\n\\end{equation}\n"
    content += r"""The stationary plane contributes zero inverse mass. Use the full tensor, then rotate into the ordered contact basis $(n,t_1,t_2)$:
\begin{equation}
\mathbf W=\mathbf C\mathbf M^{-1}\mathbf C^T,\qquad
\widehat{\mathbf W}=\mathbf Q^T\mathbf W\mathbf Q,\qquad
\mathbf Q=[\mathbf e_z\ \mathbf e_x\ \mathbf e_y].
\end{equation}
"""
    content += "\\begin{equation}\n\\widehat{\\mathbf W}=" + matrix_tex(result["contact_mobility_basis_per_kg"]) + "\\,{\\rm kg^{-1}}.\n\\end{equation}\n"
    content += r"""Its off-diagonal entries couple normal and tangential impulses: solving only its diagonal would give a different collision. Solve the small symmetric system rather than form an explicit inverse:
\begin{align}
\widehat{\mathbf W}\widehat{\mathbf j}+(\mathbf1+\mathbf E)\mathbf Q^T\mathbf u^-&=0,\qquad
\mathbf E=\operatorname{diag}(e_n,e_t,e_t),\\
\mathbf j&=\mathbf Q\widehat{\mathbf j},\\
\mathbf V^+&=\mathbf V^-+\mathbf M^{-1}\mathbf C^T\mathbf j.
\end{align}
Check $j_n\ge0$, $\|\mathbf j_t\|\le\mu j_n$ and the actual kinetic-energy increment before accepting that endpoint solution. Here the unconstrained target is admissible, so no friction-capacity adjustment is required.
\begin{center}\small
\begin{tabular}{ll}
\hline
Computed result & Value\\
\hline
"""
    rows = [
        (r"$\mathbf u^-$, world $(x,y,z)$", fmt(result["incoming_contact_velocity_m_s"]) + r"\,{\rm m/s}"),
        (r"$\widehat{\mathbf j}$, contact $(n,t_1,t_2)$", fmt(result["impulse_basis_kg_m_s"]) + r"\,{\rm kg\,m/s}"),
        (r"$\mathbf v^+$", fmt(result["outgoing_velocity_m_s"]) + r"\,{\rm m/s}"),
        (r"$\boldsymbol\omega^+$", fmt(result["outgoing_angular_velocity_rad_s"]) + r"\,{\rm rad/s}"),
        (r"$\mathbf u^+$, world $(x,y,z)$", fmt(result["outgoing_contact_velocity_m_s"]) + r"\,{\rm m/s}"),
        (r"$\|\mathbf j_t\|/j_n$", f"${result['tangential_impulse_over_normal_impulse']:.6f}$"),
        (r"$T^-$; $T^+$", f"${result['energy_before_J']:.6f}$; ${result['energy_after_J']:.6f}\\,{{\\rm J}}$"),
    ]
    # Unit strings must be inside math mode, rather than following a closed $.
    for label, value in rows:
        value = value.replace(r")$\,", r")\,")
        if r"\,{\rm" in value and not value.endswith("$"):
            value += "$"
        content += label + " & " + value + r"\\" + "\n"
    content += r"""\hline
\end{tabular}\end{center}
The example changes translation and all three spin components while satisfying both restitution targets. Energy decreases; the stationary plane receives the opposing impulse. Linear and angular momentum are checked through their impulse balances, rather than assuming conservation for the cuboid alone.

The retained calculation compares scaled matrix operations with independent unscaled cross-product updates, checks the impulse/energy identities and the supporting vertex, and repeats the solve at four reference lengths and three rotated tangent bases. Physical impulses and outgoing states remain unchanged to the reported numerical precision. These are algebra and geometry checks, not an experimental accuracy claim or a native time-step replay.
\begin{figure}[htbp]
\centering
\includegraphics[width=\linewidth]{worked-3d-v1/geometry-motion.png}
\caption{The rotated cuboid contacts the plane at its unique lowest corner. The right panel compares incoming and outgoing translation and length-scaled spin in the same velocity units; arrows begin at a common origin for comparison. The impulse acts at the corner, while spin is defined about the centre of mass.}
\end{figure}
"""
    (output / "example.tex").write_text(content)


def plot(inputs, result, output):
    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(121, projection="3d")
    vertices = np.array(result["vertices_relative_com_m"]) - result["contact_lever_m"]
    signs = list(itertools.product((-1, 1), repeat=3))
    faces = []
    for axis in range(3):
        for side in (-1, 1):
            other = [i for i in range(3) if i != axis]
            indices = []
            for pair in [(-1,-1),(-1,1),(1,1),(1,-1)]:
                sign = [0,0,0]; sign[axis] = side
                for i, value in zip(other, pair): sign[i] = value
                indices.append(signs.index(tuple(sign)))
            faces.append(vertices[indices])
    ax.add_collection3d(Poly3DCollection(faces, facecolors="#8bc3d9", edgecolors="#35627a", alpha=.24))
    plane = np.array([[-.32,-.32,0],[.32,-.32,0],[.32,.32,0],[-.32,.32,0]])
    ax.add_collection3d(Poly3DCollection([plane], facecolors="#cccccc", alpha=.15))
    com = -np.asarray(result["contact_lever_m"])
    ax.scatter(*com, c="#344454", s=30)
    ax.text(*(com+np.array([0,0,.025])), "COM", fontsize=8)
    ax.scatter(0,0,0,c="#b64332",s=35)
    ax.text(0,0,-.04,"Contact",fontsize=8)
    ax.plot([com[0],0],[com[1],0],[com[2],0],color="#b64332",linewidth=2)
    ax.quiver(0,0,0,0,0,.14,color="#333333",arrow_length_ratio=.15)
    ax.text(0,0,.15,"n",fontsize=9)
    ax.set(xlim=(-.32,.32),ylim=(-.32,.32),zlim=(-.05,.58),
           xlabel="x (m)",ylabel="y (m)",zlabel="z (m)",title="3D contact geometry")
    ax.set_box_aspect((1,1,1))
    ax.set_xticks([-.2, 0, .2])
    ax.set_yticks([-.2, 0, .2])
    ax.set_zticks([0, .2, .4])
    ax.view_init(elev=21,azim=-55)
    ax = fig.add_subplot(122, projection="3d")
    for vector, color, label in [
        (inputs["incoming_velocity_m_s"],"#2876a3",r"$\mathbf{v}^-$"),
        (result["outgoing_velocity_m_s"],"#d77829",r"$\mathbf{v}^+$"),
        (inputs["reference_length_m"]*np.array(inputs["incoming_angular_velocity_rad_s"]),"#695ca5",r"$\ell\omega^-$"),
        (inputs["reference_length_m"]*np.array(result["outgoing_angular_velocity_rad_s"]),"#2d946c",r"$\ell\omega^+$"),
    ]:
        vector = np.asarray(vector)
        ax.quiver(0,0,0,*vector,color=color,arrow_length_ratio=.1,linewidth=2,label=label)
    ax.set(xlim=(-.3,1.2),ylim=(-.8,.7),zlim=(-2.1,1.5),
           xlabel="x (m/s)",ylabel="y (m/s)",zlabel="z (m/s)",title="Velocity and scaled spin")
    # Equal physical scale in all axes; the range in z is intentionally larger.
    ax.set_box_aspect((1.5,1.5,3.6))
    ax.set_xticks([0, .5, 1])
    ax.set_yticks([-.5, 0, .5])
    ax.set_zticks([-2, -1, 0, 1])
    ax.view_init(elev=15,azim=-55)
    ax.legend(loc="upper left",fontsize=9)
    fig.subplots_adjust(left=.01,right=.96,bottom=.05,top=.91,wspace=.08)
    for suffix in ("png","pdf"):
        fig.savefig(output / f"geometry-motion.{suffix}",dpi=180,bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "worked-3d-v1")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    input_path = HERE / "worked_3d_inputs.json"
    inputs = json.loads(input_path.read_text())
    result = calculate(inputs, inputs["reference_length_m"])
    scale_errors, basis_errors = [], []
    physical_keys = ["impulse_world_kg_m_s", "outgoing_velocity_m_s", "outgoing_angular_velocity_rad_s"]
    for ell in (1e-5, .02, .2, 1000.):
        trial = calculate(inputs, ell)
        scale_errors.append({"ell_m": ell, "max_physical_state_difference": float(max(
            np.max(abs(np.array(result[key])-trial[key])) for key in physical_keys)),
            "checks": trial["checks"]})
    for angle in (.35, .91, 1.7):
        trial = calculate(inputs, inputs["reference_length_m"], angle)
        basis_errors.append({"angle_rad": angle, "max_physical_state_difference": float(max(
            np.max(abs(np.array(result[key])-trial[key])) for key in physical_keys)),
            "checks": trial["checks"]})
    largest = max(row["max_physical_state_difference"] for row in scale_errors+basis_errors)
    assert largest < 1e-11
    receipt = {"input_table": input_path.name,
               "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
               "interpretation": inputs["purpose"], "native_replay_claimed": False,
               "experimental_validation_claimed": False, "main": result,
               "reference_length_controls": scale_errors, "tangent_basis_controls": basis_errors,
               "max_invariance_difference": largest}
    (args.output / "calculation.json").write_text(json.dumps(receipt,indent=2)+"\n")
    write_tex(inputs,result,args.output)
    plot(inputs,result,args.output)
    print(json.dumps({"output": str(args.output), "max_invariance_difference": largest,
                      "energy_change_J": result["energy_change_J"], "status": "PASS"}))


if __name__ == "__main__":
    main()
