#include "coulomb_active.h"
#include <iostream>
#include <stdexcept>
static void require(bool ok,const char*message){if(!ok)throw std::runtime_error(message);}
int main(){
 {btMatrixXu A(2,2);A.setZero();A.setElem(0,0,1);A.setElem(1,1,1);btVectorXu b(2),u(2),seed(2),out(2);b[0]=1;b[1]=-1;u[0]=u[1]=1e30;seed[0]=seed[1]=1;out[0]=7;out[1]=8;normal_pressure::Stats s;require(normal_pressure::solve(A,b,u,seed,out,1e-8,s),"stale tensile row not released");require(out[0]==1&&out[1]==0,"wrong unilateral impulse");}
 {btMatrixXu A(2,2);A.setElem(0,0,1);A.setElem(1,1,1);A.setElem(0,1,-1);A.setElem(1,0,-1);btVectorXu b(2),u(2),seed(2),out(2);b[0]=b[1]=1;u[0]=u[1]=1e30;seed[0]=seed[1]=1;out[0]=7;out[1]=8;normal_pressure::Stats s;require(!normal_pressure::solve(A,b,u,seed,out,1e-8,s),"infeasible normal system falsely passed");require(out[0]==7&&out[1]==8&&s.attempts<=128,"failed numerical search changed output or exceeded bound");}
 {btMatrixXu A(2,2);for(int i=0;i<2;i++)for(int j=0;j<2;j++)A.setElem(i,j,1);btVectorXu b(2),u(2),seed(2),out(2);b[0]=b[1]=1;u[0]=u[1]=1e30;seed[0]=.9;seed[1]=.1;normal_pressure::Stats s;require(normal_pressure::solve(A,b,u,seed,out,1e-8,s),"redundant pressure face rejected");require(std::abs(out[0]+out[1]-1)<1e-12&&out[0]>=0&&out[1]>=0,"redundant pressure not physically complementary");}
 // A large contact set has only two physically compressing contacts. Contact
 // two starts omitted, so a subset-only success would violate its normal law.
 {const int contacts=150,n=contacts*3;btMatrixXu A(n,n);A.setZero();btVectorXu b(n),p(n),hi(n);btAlignedObjectArray<int>dep;dep.resize(n);
  for(int k=0;k<contacts;k++)for(int h=0;h<3;h++){int r=3*k+h;A.setElem(r,r,1);p[r]=0;b[r]=h?0:-1;hi[r]=h?.4:1e30;dep[r]=h?3*k:-1;}
  b[0]=1;p[0]=1;b[3]=.25;circular_active::Stats s;require(circular_active::solve(A,b,p,hi,dep,1e-8,s),"large sparse pressure support failed");require(std::abs(p[0]-1)<=1e-8&&std::abs(p[3]-.25)<=1e-8&&s.expanded_contacts==1&&s.passes==2,"omitted compressing contact escaped full-system gate");for(int r=6;r<n;r++)require(p[r]==0,"inactive large-system row changed");}
 {btMatrixXu A(4,4);A.setZero();btVectorXu b(4),p(4),hi(4);btAlignedObjectArray<int>d;d.resize(4);
  for(int i=0;i<4;i++){A.setElem(i,i,1);b[i]=0;p[i]=7;hi[i]=1e30;d[i]=i?0:-1;}
  circular_active::Stats s;require(!circular_active::solve(A,b,p,hi,d,1e-8,s),"orphan contact row falsely accepted");for(int i=0;i<4;i++)require(p[i]==7,"unsupported contact structure changed output");}
 std::cout<<"PASS pressure release/redundancy/infeasibility/failed-output; full450-row inactive-contact expansion\n";
}
