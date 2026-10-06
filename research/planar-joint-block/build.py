"""Preserved baseline transform followed by an explicit isolated solver override."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from research.build_precision_backend import build
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=ROOT/'build/rigid_double_joint'
build(D,Path('/tmp/box2d-block.tar.gz'),1e-6)
manifest=D/'precision-source.json';shutil.copy2(manifest,D/'initial-precision-source.json');inventory=json.loads(manifest.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
folder=D/'source/box2d-2.4.1/src/dynamics';cpp=folder/'b2_contact_solver.cpp';text=cpp.read_text();marker='#include "b2_contact_solver.h"';assert text.count(marker)==1;text=text.replace(marker,marker+'\n#include "joint_contact.h"')
marker='\t\tb2Assert(pointCount == 1 || pointCount == 2);';assert text.count(marker)==1
text=text.replace(marker,marker+'''
        if(physicsJointContact(vc,vA,wA,vB,wB)) {
            m_velocities[indexA].v=vA;m_velocities[indexA].w=wA;
            m_velocities[indexB].v=vB;m_velocities[indexB].w=wB;
            continue;
        }
''');cpp.write_text(text);shutil.copy2(H/'joint_contact.h',folder/'joint_contact.h')
inventory['baseline_precision_inventory_sha256']=sha(D/'initial-precision-source.json');inventory['numerical_change']={'joint_normal_tangent_block_solver':True,'root_velocity_gate_m_s':1e-10,'max_faces_per_constraint':16,'max_unknowns':4,'linear_rank_relative_cutoff':1e-12,'fallback':'unchanged_original_contact_iteration_if_no_checked_root','selection':'smallest_warm_impulse_change','linear_slop_m':1e-6}
for p in [cpp,folder/'joint_contact.h']:inventory['transformed'][str(p.relative_to(D/'source'))]=sha(p)
inventory['prototype_sha256']=sha(H/'joint_contact.h');inventory['builder_sha256']=sha(Path(__file__));manifest.write_text(json.dumps(inventory,indent=2)+'\n')
subprocess.run(['cmake','--build',str(D),'-j','4'],check=True)
(H/'build-receipt.json').write_text(json.dumps({'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'manifest_sha256':sha(manifest),'binary_sha256':sha(D/'rigid_runner'),'builder_sha256':sha(Path(__file__)),'prototype_sha256':sha(H/'joint_contact.h'),'scope':'Experimental isolated solver override; original precision inventory and all prior builds preserved.'},indent=2)+'\n')
