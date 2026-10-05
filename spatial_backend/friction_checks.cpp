// Independent analytic circular-friction checks, not scene integration tests.
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "coulomb.h"
#include <iostream>
int main(){
 auto check=[](double off,double pn,double pt,double ps,double wt,double ws,double mu){
  btMatrixXu A(3,3);A.setZero();A.setElem(0,0,1);A.setElem(1,1,3.5);A.setElem(2,2,3.5);A.setElem(0,1,off);A.setElem(1,0,off);
  btVectorXu b(3),x(3),lo(3),hi(3);double exact[]={pn,pt,ps},w[]={0,wt,ws};
  btAlignedObjectArray<int> dep;dep.resize(3);dep[0]=-1;dep[1]=dep[2]=0;
  for(int i=0;i<3;i++){b[i]=-w[i];for(int j=0;j<3;j++)b[i]+=A(i,j)*exact[j];x[i]=0;lo[i]=i==0?0:-mu;hi[i]=i==0?1e30:mu;}
  CoulombStats stats;
  if(!coulombSolve(A,b,x,lo,hi,dep,256,1e-10,stats))throw std::runtime_error("Coulomb did not converge");
  for(int i=0;i<3;i++)if(std::abs(x[i]-exact[i])>1e-9)throw std::runtime_error("Analytic circular friction mismatch");
  if(stats.passive_change_max>1e-10)throw std::runtime_error("Passive contact adds energy");
 };
 check(0,2,.1,.2,0,0,.4); // Sticking cancels both tangent velocities.
 for(double angle:{0.,.3,1.,2.5}){double c=std::cos(angle),s=std::sin(angle);check(0,2,.8*c,.8*s,-3*c,-3*s,.4);} // Isotropic sliding.
 check(.2,2,.8,0,-3,0,.4); // Normal/tangent cross coupling changes required pn.
 check(.2,2,.1,.2,0,0,.4);
 check(0,2,0,0,-3,2,0); // Zero friction still solves normals.
 std::cout<<"Circular Coulomb analytic checks PASS (8 cases)\n";
}
