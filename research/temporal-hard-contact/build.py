"""Isolated pinned Float64 hard-contact temporal diagnostic, not production."""
import hashlib,json,subprocess,tarfile
from pathlib import Path
from research.build_precision_backend import convert
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
plan=json.loads((H/'plan.json').read_text());D=ROOT/'build/rigid_temporal_double_hard_v2'
D.mkdir(exist_ok=False);source=D/'source';source.mkdir()
archive=ROOT/'build/rigid_backend/_deps/box2d-subbuild/box2d-populate-prefix/src/archive.tar'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(archive)==plan['archive_sha256']
with tarfile.open(archive) as t:t.extractall(source,filter='data')
box=next(p for p in source.iterdir() if p.is_dir());inputs={}
for p in sorted(box.rglob('*')):
 if p.is_file() and p.suffix in ('.c','.cpp','.h'):
  inputs[str(p.relative_to(source))]=sha(p);p.write_text(convert(p.read_text()))
def replace(p,a,b):
 s=p.read_text();assert s.count(a)==1,(p,a,s.count(a));p.write_text(s.replace(a,b))
replace(box/'src/constants.h','( 0.005 * b2_lengthUnitsPerMeter )','( 0.000001 * b2_lengthUnitsPerMeter )')
replace(box/'src/constraint_graph.c','#define B2_FORCE_OVERFLOW 0','#define B2_FORCE_OVERFLOW 1')
replace(box/'src/solver.h','.massScale = 0.0,','.massScale = 1.0,')
# Preserve the original authored core-only polygon mass/inertia convention.
p=box/'src/geometry.c';s=p.read_text();start=s.index('b2MassData b2ComputePolygonMass(');end=s.index('b2AABB b2ComputeCircleAABB(',start)
chunk=s[start:end];assert chunk.count('double radius = shape->radius;')==1
p.write_text(s[:start]+chunk.replace('double radius = shape->radius;','double radius = 0.0;')+s[end:])
p=box/'src/contact_solver.c';s=p.read_text();marker='void b2WarmStartOverflowContacts('
at=s.index(marker)
observer='''
static double physicsWork=0,physicsAbsWork=0,physicsFriction=0;
static long long physicsPoints=0;
static bool physicsEnabled=true;
void b2PhysicsObserverEnable(bool enabled){physicsEnabled=enabled;}
double b2PhysicsBoundaryWork(void){return physicsWork;}
double b2PhysicsAbsoluteBoundaryWork(void){return physicsAbsWork;}
double b2PhysicsFrictionImpulse(void){return physicsFriction;}
long long b2PhysicsBoundaryPoints(void){return physicsPoints;}
static void physicsObserve(double mA,double mB,b2Vec2 vA,double wA,b2Vec2 rA,b2Vec2 vB,double wB,b2Vec2 rB,b2Vec2 P,double tangentImpulse){
 if(!physicsEnabled)return;
 physicsFriction+=fabs(tangentImpulse);
 double work=0;
 if(mA==0 && mB>0)work=b2Dot(P,b2Add(vA,b2CrossSV(wA,rA)));
 else if(mB==0 && mA>0)work=-b2Dot(P,b2Add(vB,b2CrossSV(wB,rB)));
 else return;
 physicsWork+=work;physicsAbsWork+=fabs(work);physicsPoints++;
}
static void physicsObserveRolling(double mA,double mB,double wA,double wB,double impulse){
 if(!physicsEnabled)return;
 double work=0;
 if(mA==0 && mB>0)work=impulse*wA;
 else if(mB==0 && mA>0)work=-impulse*wB;
 else return;
 physicsWork+=work;physicsAbsWork+=fabs(work);
}
'''
s=s[:at]+observer+s[at:]
end=s.index('void b2StoreOverflowImpulses(');head=s[:end];tail=s[end:]
lines=head.splitlines();new=[];observed=0
for line in lines:
 new.append(line)
 if 'b2Vec2 P = ' in line:
  tangent='cp->tangentImpulse' if 'cp->normalImpulse, normal' in line else ('impulse' if 'impulse, tangent' in line else '0.0')
  new.append('\t\t\tphysicsObserve(mA,mB,vA,wA,rA,vB,wB,rB,P,'+tangent+');');observed+=1
 if 'wA -= iA * constraint->rollingImpulse;' in line:
  new.insert(len(new)-1,'\t\tphysicsObserveRolling(mA,mB,wA,wB,constraint->rollingImpulse);')
 if 'wA -= iA * deltaLambda;' in line:
  new.insert(len(new)-1,'\t\t\tphysicsObserveRolling(mA,mB,wA,wB,deltaLambda);')
assert observed==4,observed
p.write_text('\n'.join(new)+'\n'+tail)
runner=convert((ROOT/'rigid_backend/runner.cpp').read_text()).replace('std::setprecision(10)','std::setprecision(17)')
runner=runner.replace('int main() {','''extern "C" {
void b2PhysicsObserverEnable(bool);
double b2PhysicsBoundaryWork(void);
double b2PhysicsAbsoluteBoundaryWork(void);
double b2PhysicsFrictionImpulse(void);
long long b2PhysicsBoundaryPoints(void);
}
int main() {''')
runner=runner.replace('b2WorldId world = b2CreateWorld(&wd);','''wd.contactHertz=0;
    b2PhysicsObserverEnable(std::getenv("PHYSICS_DISABLE_OBSERVER")==nullptr);
    b2WorldId world = b2CreateWorld(&wd);''')
needle='    std::cout << ",\\\"step_s\\\":" << stepSeconds'
assert needle in runner
runner=runner.replace(needle,'''    std::cout << ",\\\"boundary_work_J\\\":" << b2PhysicsBoundaryWork()
              << ",\\\"absolute_boundary_work_J\\\":" << b2PhysicsAbsoluteBoundaryWork()
              << ",\\\"friction_impulse_abs_kg_m_s\\\":" << b2PhysicsFrictionImpulse()
              << ",\\\"boundary_impulse_points\\\":" << b2PhysicsBoundaryPoints();
'''+needle)
(source/'runner.cpp').write_text(runner)
(source/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.22)
project(temporal_hard_contact LANGUAGES C CXX)
set(BOX2D_SAMPLES OFF CACHE BOOL "" FORCE)
set(BOX2D_UNIT_TESTS OFF CACHE BOOL "" FORCE)
set(BOX2D_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(BOX2D_VALIDATE OFF CACHE BOOL "" FORCE)
set(BOX2D_DISABLE_SIMD ON CACHE BOOL "" FORCE)
add_subdirectory('''+box.name+''')
add_executable(rigid_runner runner.cpp)
target_compile_features(rigid_runner PRIVATE cxx_std_17)
target_compile_definitions(rigid_runner PRIVATE RIGID_DOUBLE_PRECISION)
target_compile_options(rigid_runner PRIVATE -ffp-contract=off)
target_compile_options(box2d PRIVATE -ffp-contract=off)
target_link_libraries(rigid_runner PRIVATE box2d)
''')
receipt={'linear_slop_m':1e-6,'polygon_mass_policy':'authored_core_only_skin_massless','archive_sha256':sha(archive),'source':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'inputs':inputs,'transformed':{str(p.relative_to(source)):sha(p) for p in sorted(source.rglob('*')) if p.is_file() and p.suffix in ('.h','.c','.cpp')},'plan_sha256':sha(H/'plan.json'),'builder_sha256':sha(Path(__file__))}
(D/'precision-source.json').write_text(json.dumps(receipt,indent=2)+'\n')
subprocess.run(['cmake','-S',str(source),'-B',str(D),'-G','Ninja','-DCMAKE_BUILD_TYPE=Release'],check=True)
subprocess.run(['cmake','--build',str(D),'-j','4'],check=True)
receipt['binary_sha256']=sha(D/'rigid_runner');(H/'build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
