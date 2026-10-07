#include "support_restart.h"
#include <iostream>
#include <stdexcept>
void require(bool pass,const char* message){if(!pass)throw std::runtime_error(message);}
int main(){
 const int n=66;btMatrixXu A(n,n);A.setZero();btVectorXu b(n),p(n),upper(n);b.setZero();p.setZero();btAlignedObjectArray<int>dep;dep.resize(n);
 for(int i=0;i<n;i++){A.setElem(i,i,1);if(i%3==0){b[i]=p[i]=1;upper[i]=1e10;dep[i]=-1;}else{upper[i]=.4;dep[i]=i-i%3;}}
 support_restart_v3::Stats independent;require(support_restart_v3::solve(A,b,p,upper,dep,1e-8,independent),"large support exact independent components");require(independent.helper_calls==0&&independent.skipped_components==22&&independent.largest_reduced_rows==3,"contact triples and valid component skip");
 A.setElem(0,3,1e-20);A.setElem(3,0,1e-20);support_restart_v3::Stats tiny;require(support_restart_v3::solve(A,b,p,upper,dep,1e-8,tiny),"tiny coupling valid seed");require(tiny.largest_reduced_rows==6,"never drop tiny nonzero coupling");
 for(int i=0;i<n;i+=3)for(int j=0;j<i;j+=3){A.setElem(i,j,1e-20);A.setElem(j,i,1e-20);}
 const auto saved=p;support_restart_v3::Stats cap;require(!support_restart_v3::solve(A,b,p,upper,dep,1e-8,cap),"component cap decline");require(cap.component_cap_rejections==1&&cap.svd_calls==0,"cap decline before search");for(int i=0;i<n;i++)require(p[i]==saved[i],"cap failure unchanged caller");
 dep[1]=n+1;support_restart_v3::Stats bad;require(!support_restart_v3::solve(A,b,p,upper,dep,1e-8,bad),"dependency index rejection");for(int i=0;i<n;i++)require(p[i]==saved[i],"invalid dependency unchanged caller");dep[1]=0;
 b[0]=std::numeric_limits<double>::quiet_NaN();require(!support_restart_v3::solve(A,b,p,upper,dep,1e-8,bad),"nonfinite rejection");for(int i=0;i<n;i++)require(p[i]==saved[i],"nonfinite unchanged caller");
 {
  btMatrixXu extreme(3,3);extreme.setZero();extreme.setElem(0,0,1);extreme.setElem(1,1,1e308);extreme.setElem(2,2,1e308);
  btVectorXu rhs(3),x(3),hi(3);rhs.setZero();rhs[0]=rhs[1]=1;x.setZero();x[0]=1;hi[0]=1e10;hi[1]=hi[2]=.5;
  btAlignedObjectArray<int> d;d.resize(3);d[0]=-1;d[1]=d[2]=0;
  support_restart_v3::Stats overflow;require(!support_restart_v3::solve(extreme,rhs,x,hi,d,1e-8,overflow),"overflow metric declines");
  require(x[0]==1&&x[1]==0&&x[2]==0&&overflow.helper_calls==0,"overflow failure unchanged and no helper");
 }
 {
  btMatrixXu invalid(3,3);invalid.setZero();for(int i=0;i<3;i++)invalid.setElem(i,i,1);
  btVectorXu rhs(3),x(3),hi(3);rhs.setZero();rhs[0]=1;x=rhs;hi[0]=1e10;hi[1]=hi[2]=.5;
  btAlignedObjectArray<int> d;d.resize(3);d[0]=-1;d[1]=d[2]=0;
  invalid.setElem(2,1,1);support_restart_v3::Stats asym;
  require(!support_restart_v3::solve(invalid,rhs,x,hi,d,1e-8,asym),"asymmetric matrix declines");
  invalid.setElem(0,0,1e100);rhs[0]=1e100;support_restart_v3::Stats scaled_asym;
  require(!support_restart_v3::solve(invalid,rhs,x,hi,d,1e-8,scaled_asym),"large unrelated diagonal cannot hide asymmetry");
  invalid.setElem(0,0,1);rhs[0]=1;
  invalid.setElem(2,1,0);invalid.setElem(2,2,-1);support_restart_v3::Stats negative;
  require(!support_restart_v3::solve(invalid,rhs,x,hi,d,1e-8,negative),"negative tangent diagonal declines");
  invalid.setElem(2,2,1);invalid.setElem(1,2,2);invalid.setElem(2,1,2);support_restart_v3::Stats indefinite;
  require(!support_restart_v3::solve(invalid,rhs,x,hi,d,1e-8,indefinite),"indefinite tangent block declines");
  require(x[0]==1&&x[1]==0&&x[2]==0,"invalid mobility leaves caller unchanged");
 }
 std::cout<<"PASS exact component triples, independent support66, untouched tiny couplings, no-work cap rejection, invalid dependency/nonfinite and failure-output preservation\n";
}
