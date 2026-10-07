#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "coulomb.h"
#include "relation_recovery.h"
#include <iostream>
#include <stdexcept>

int main() {
 int checks=0;
 auto run=[&](int n,int invalid, bool reused=false) {
  btMatrixXu A(n,n);btVectorXu b(n),x(n),hi(n);
  btAlignedObjectArray<int> dep;dep.resize(n);
  for(int i=0;i<n;i++) {
   b[i]=1; x[i]=0;hi[i]=i%3==0?1e10:.4;dep[i]=i%3==0?-1:i-i%3;
   for(int j=0;j<n;j++)A.setElem(i,j,i==j?2:(n>64?.001:0));
  }
  double tolerance=1e-8;
  if(invalid==1)tolerance=-1;
  if(invalid==2)b[0]=std::numeric_limits<double>::quiet_NaN();
  if(invalid==3)A.setElem(0,1,.1);
  if(invalid==4)hi[1]=-1;
  if(invalid==5)dep[1]=n;
  relation_recovery::Stats stats;
  if(reused)stats.iteration_sweeps=100000;
  bool accepted=relation_recovery::solve(A,b,x,hi,dep,tolerance,stats);
  if(accepted)throw std::runtime_error("Unexpected control acceptance");
  for(int i=0;i<n;i++)if(x[i]!=0)throw std::runtime_error("Decline changed caller impulse");
  if(stats.guides>8||stats.components>4||stats.iteration_sweeps>(reused?116384:16384))throw std::runtime_error("Exceeded per-call budget");
  if(n>64&&stats.component_cap_rejections!=1)throw std::runtime_error("Missing component cap decline");
  checks++;
 };
 run(3,0);for(int invalid=1;invalid<=5;invalid++)run(3,invalid);
 run(66,0);run(3,0,true);
 std::cout<<checks<<" decline, invalid-input, cap and reused-stat controls pass\n";
}
