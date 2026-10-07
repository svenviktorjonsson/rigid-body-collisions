"""Isolated full-island classical-friction adapter using frozen native searches."""
import hashlib,json,shutil,subprocess
from pathlib import Path
from importlib import import_module
build=import_module('research.planar-tight-slop.precision_build').build
H=Path(__file__).resolve().parent;ROOT=H.parents[1];D=ROOT/'build/rigid_double_tightslop_v1'
build(D,Path('/tmp/box2d-block.tar.gz'),1e-12);shutil.copy2(D/'rigid_runner',D/'original_tight_slop_runner');sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();manifest=D/'precision-source.json';shutil.copy2(manifest,D/'initial-precision-source.json');inventory=json.loads(manifest.read_text());folder=D/'source/box2d-2.4.1/src/dynamics'
p=folder/'b2_contact_solver.h';s=p.read_text();needle='\tb2TimeStep m_step;';assert s.count(needle)==1;s=s.replace(needle,'\tbool m_physicsGlobalSolved=false;\n'+needle);p.write_text(s)
p=folder/'b2_contact_solver.cpp';s=p.read_text();needle='#include "b2_contact_solver.h"';assert s.count(needle)==1;s=s.replace(needle,needle+'\n#include "global_contact.h"');needle='void b2ContactSolver::SolveVelocityConstraints()\n{';assert s.count(needle)==1;s=s.replace(needle,needle+'''
    if(m_physicsGlobalSolved)return;
    if(physicsGlobalContact(m_velocityConstraints,m_count,m_velocities)) {
        m_physicsGlobalSolved=true;return;
    }
''');s=s.replace('vc->pointCount = 1;', 'if(std::getenv("PHYSICS_DISABLE_GLOBAL")!=nullptr)vc->pointCount = 1;');p.write_text(s);shutil.copy2(H/'global_contact.h',folder/'global_contact.h')
p=D/'source/runner.cpp';s=p.read_text();s=s.replace('b2WorldId world = b2CreateWorld(&wd);','b2WorldId world = b2CreateWorld(&wd);\n    if(std::getenv("PHYSICS_DISABLE_GLOBAL")==nullptr)world->SetContinuousPhysics(false);');s=s.replace('int main() {','''extern "C" {
long long b2PhysicsGlobalCalls();long long b2PhysicsGlobalSolves();long long b2PhysicsGlobalDeclines();
double b2PhysicsGlobalResidual();double b2PhysicsGlobalBodyResidual();
}
int main() {''');needle='    std::cout << ",\\\"step_s\\\":" << stepSeconds';assert needle in s;s=s.replace(needle,'''    std::cout << ",\\\"global_contact_calls\\\":" << b2PhysicsGlobalCalls()
              << ",\\\"global_contact_solves\\\":" << b2PhysicsGlobalSolves()
              << ",\\\"global_contact_declines\\\":" << b2PhysicsGlobalDeclines()
              << ",\\\"global_native_residual_max_m_s\\\":" << b2PhysicsGlobalResidual()
              << ",\\\"global_actual_body_residual_max_m_s\\\":" << b2PhysicsGlobalBodyResidual();
'''+needle);p.write_text(s)
cache={}
for line in (ROOT/'build/spatial/CMakeCache.txt').read_text().splitlines():
 if '=' in line and ':' in line:cache[line.split(':')[0]]=line.split('=',1)[1]
bullet=Path(cache['BULLET_PHYSICS_SOURCE_DIR']);lib=ROOT/'build/spatial/_deps/bullet-build/src'
p=D/'source/CMakeLists.txt';s=p.read_text();s+='''
set_target_properties(box2d PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON)
target_compile_definitions(box2d PRIVATE BT_USE_DOUBLE_PRECISION SPATIAL_LAPACK_RECOVERY=1)
target_include_directories(box2d PRIVATE "'''+str(ROOT/'spatial_backend')+'" "'+str(bullet/'src')+'''")
target_link_libraries(box2d PUBLIC "'''+str(lib/'BulletDynamics/libBulletDynamics.a')+'" "'+str(lib/'BulletCollision/libBulletCollision.a')+'" "'+str(lib/'LinearMath/libLinearMath.a')+'''" -l:liblapack.so.3 -l:libblas.so.3)
''';p.write_text(s)
inventory['baseline_precision_inventory_sha256']=sha(D/'initial-precision-source.json');inventory['numerical_change']={'full_island_scalar_coulomb_gate_m_s':1e-10,'internal_search_gate_m_s':2.5e-11,'continuous_TOI_enabled':False,'prepared_manifold_points':'retain both; exact global matrix handles rank deficiency','virtual_tangent':'unit isolated mobility, zero rhs/impulse, no physical DOF','row_cap':4096,'native_search_iteration_budget':4096,'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'physical_passivity_scale':'1+preimpact_kinetic_energy+absolute_wall_work','linear_slop_m':1e-12,'outer_velocity_iteration_skip':'only after full native and actual-body gates pass'}
for p in [folder/'b2_contact_solver.h',folder/'b2_contact_solver.cpp',folder/'global_contact.h',D/'source/runner.cpp']:inventory['transformed'][str(p.relative_to(D/'source'))]=sha(p)
inventory['native_source_guards']={str(p):sha(p) for p in (ROOT/'spatial_backend').glob('*') if p.is_file()};inventory['prototype_sha256']=sha(H/'global_contact.h');inventory['builder_sha256']=sha(Path(__file__));manifest.write_text(json.dumps(inventory,indent=2)+'\n')
subprocess.run(['cmake','-S',str(D/'source'),'-B',str(D)],check=True);subprocess.run(['cmake','--build',str(D),'-j','4'],check=True)
(H/'build-receipt.json').write_text(json.dumps({'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'manifest_sha256':sha(manifest),'binary_sha256':sha(D/'rigid_runner'),'prototype_sha256':sha(H/'global_contact.h'),'builder_sha256':sha(Path(__file__))},indent=2)+'\n')
