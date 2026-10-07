// PREPARED, UNCOMPILED: link only against research candidate headers after clearance.
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "coulomb.h"
#include <iostream>
#include <stdexcept>

static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static bool equal(const btVectorXu& a,const btVectorXu& b){
 if(a.rows()!=b.rows())return false;
 for(int i=0;i<a.rows();i++)if(a[i]!=b[i])return false;
 return true;
}
static void immediate_success(){
 btMatrixXu A(3,3);btVectorXu b(3),x(3),lo(3),hi(3);
 btAlignedObjectArray<int> dep;dep.resize(3);
 for(int i=0;i<3;i++){b[i]=i==0?1:0;x[i]=0;lo[i]=i==0?0:-.4;hi[i]=i==0?1e30:.4;dep[i]=i==0?-1:0;
  for(int j=0;j<3;j++)A.setElem(i,j,i==j?1:0);}
 const auto initial=x;CoulombStats disabled,enabled;
 auto default_x=x;
 require(coulombSolve(A,b,default_x,lo,hi,dep,4096,1e-8,disabled),"Default small solve rejected");
 require(coulombSolve(A,b,x,lo,hi,dep,4096,1e-8,enabled,nullptr,true,true),"Opt-in small solve rejected");
 require(equal(default_x,x),"Opt-in changed an immediate-success endpoint");
 require(enabled.early_component_attempts==0,"Unneeded helper called after successful first phase");
 require(disabled.iteration_sweeps_total==enabled.iteration_sweeps_total,"Unneeded initial sweep work");
 // Aggregated earlier counters must not consume a subsequent call's work cap.
 x=initial;enabled.early_component_svd_calls=1000000;
 require(coulombSolve(A,b,x,lo,hi,dep,4096,1e-8,enabled,nullptr,true,true),"Aggregate counters blocked a new solve");
 require(enabled.early_component_svd_calls==1000000,"Receipt unexpectedly reset");
}
#ifdef SPATIAL_LAPACK_RECOVERY
static void rejected_component_and_invalid_input_never_write(){
 // 24 complete triples, all exactly coupled: after one supported release the
 // remaining 69-row component still exceeds the immutable per-component cap.
 const int n=72;btMatrixXu A(n,n);btVectorXu b(n),x(n),hi(n);
 btAlignedObjectArray<int> dep;dep.resize(n);
 for(int i=0;i<n;i++){
  bool normal=i%3==0;b[i]=normal?1:0;x[i]=normal?1:0;hi[i]=normal?1e30:.4;
  dep[i]=normal?-1:i-i%3;
  for(int j=0;j<n;j++)A.setElem(i,j,i==j?2:.001);
 }
 const auto initial=x;support_restart_v3::Stats first;
 require(!support_restart_v3::solve(A,b,x,hi,dep,1e-8,first),"Over-cap connected component accepted");
 require(equal(initial,x),"Declined structural trial wrote a rejected impulse");
 require(first.component_cap_rejections>0&&first.helper_calls==0&&first.svd_calls==0,
         "Structural rejection performed or hid forbidden numerical helper work");
 // Repeat with a fresh cap object, keeping a separate aggregate receipt.
 const int accumulated=first.component_cap_rejections;support_restart_v3::Stats second;
 require(!support_restart_v3::solve(A,b,x,hi,dep,1e-8,second),"Repeated cap rejection unexpectedly accepted");
 require(equal(initial,x)&&accumulated+second.component_cap_rejections>=2,
         "Independent caps/aggregate receipt contract broken");
 A.setElem(0,1,.002);support_restart_v3::Stats asymmetric;
 require(!support_restart_v3::solve(A,b,x,hi,dep,1e-8,asymmetric),"Asymmetric external matrix accepted");
 require(equal(initial,x)&&asymmetric.helper_calls==0&&asymmetric.svd_calls==0,
         "Invalid-matrix trial wrote x or spent numerical work");
}
#endif
int main(){try{
 static_assert(sizeof(btScalar)==8,"Require Float64 Bullet");
 immediate_success();
#ifdef SPATIAL_LAPACK_RECOVERY
 rejected_component_and_invalid_input_never_write();
#endif
 std::cout<<"Optional scheduling local checks passed; captured/trajectory checks remain separate.\n";
 return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 2;}}
