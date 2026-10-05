// Analytic complementarity cases, including inactive and redundant normals.
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "normal_qp.h"
#include <iostream>
int main(){
 auto check=[](double diagonal,double off,double b0,double b1,double x0,double x1){
  btMatrixXu A(2,2);A.setElem(0,0,diagonal);A.setElem(1,1,diagonal);A.setElem(0,1,off);A.setElem(1,0,off);
  btVectorXu b(2),upper(2),x(2);b[0]=b0;b[1]=b1;upper[0]=upper[1]=1e10;x[0]=x[1]=0;
  if(!normalQP(A,b,upper,x)||std::abs(x[0]-x0)>1e-9||std::abs(x[1]-x1)>1e-9)throw std::runtime_error("Analytic normal QP mismatch");
 };
 check(1,.9,1,.1,1,0); // Inactive bound; unconstrained solution is infeasible.
 check(1,1,1,1,1,0); // Redundant pressure gauge: unique velocity, nonunique pressure.
 check(2,-1,1,0,2./3,1./3); // Coupled impulse propagation.
 check(1,0,-1,-2,0,0); // Separating contacts carry no normal impulse.
 std::cout<<"Normal QP analytic checks PASS\n";
}
