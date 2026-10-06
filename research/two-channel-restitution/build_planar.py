"""Isolated two-channel restitution build; historical binaries remain frozen."""
import hashlib,json,shutil,subprocess
from pathlib import Path
H=Path(__file__).resolve().parent;ROOT=H.parents[1]
OLD=ROOT/'build/rigid_double_global_union_v1';D=ROOT/'build/rigid_double_restitution_v1'
D.mkdir(exist_ok=False);shutil.copytree(OLD/'source',D/'source')
p=D/'source/box2d-2.4.1/src/dynamics/global_contact.h';shutil.copy2(H/'global_contact.h',p)
p=D/'source/runner.cpp';s=p.read_text();s=s.replace('int main() {','''extern "C" void b2PhysicsSetRestitution(double,double);
int main() {
    const char* normalRestitution=std::getenv("PHYSICS_NORMAL_RESTITUTION");
    const char* tangentRestitution=std::getenv("PHYSICS_TANGENTIAL_RESTITUTION");
    if(normalRestitution||tangentRestitution){
        if(!normalRestitution||!tangentRestitution)throw std::runtime_error("Both restitution channels required");
        b2PhysicsSetRestitution(std::stod(normalRestitution),std::stod(tangentRestitution));
    }
''',1)
needle='    std::cout << ",\\\"global_contact_calls\\\":"'
insert='''    if(normalRestitution)std::cout << ",\\\"contact_restitution\\\":{\\\"normal\\\":" << std::stod(normalRestitution) << ",\\\"tangential\\\":" << std::stod(tangentRestitution) << "}";
'''
assert s.count(needle)==1;s=s.replace(needle,insert+needle);p.write_text(s)
commands=[['cmake','-S',str(D/'source'),'-B',str(D),'-DCMAKE_BUILD_TYPE=Release'],['cmake','--build',str(D),'-j','4']]
with (H/'planar-build.log').open('w') as log:
 for cmd in commands:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
receipt={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'binary':str(D/'rigid_runner'),'binary_sha256':sha(D/'rigid_runner'),'prototype_sha256':sha(H/'global_contact.h'),'source_guards':{str(p):sha(p) for p in (ROOT/'spatial_backend').glob('*') if p.is_file()}}
(D/'precision-source.json').write_text(json.dumps(receipt,indent=2)+'\n');(H/'planar-build-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
print('Isolated restitution-enabled planar build PASS',flush=True)
