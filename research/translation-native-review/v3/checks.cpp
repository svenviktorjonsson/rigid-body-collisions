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
 std::cout<<"PASS exact component triples, independent support66, untouched tiny couplings, no-work cap rejection, invalid dependency/nonfinite and failure-output preservation\n";
}
