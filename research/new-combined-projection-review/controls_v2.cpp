#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include <nlohmann/json.hpp>
#include "projection_more_v2.h"
#include <iostream>
#include <limits>
using J=nlohmann::json;
int main(){J cases=J::array();bool all=true;
 auto test=[&](std::string name,int n,int mode,int budget,bool expected){btMatrixXu A(n,n);A.setZero();btVectorXu b(n),x(n),hi(n);b.setZero();x.setZero();btAlignedObjectArray<int>dep;dep.resize(n);for(int i=0;i<n;i++){A.setElem(i,i,1);hi[i]=i%3==0?1e10:.4;dep[i]=i%3==0?-1:i-i%3;}b[0]=1;if(n>1)b[1]=.2;
 if(mode==1)b[0]=-1;if(mode==2)hi[1]=-1;if(mode==3)A.setElem(0,0,std::numeric_limits<double>::quiet_NaN());if(mode==4)dep[1]=n;if(mode==5){A.setElem(0,3,-1);A.setElem(3,0,-1);b[3]=1;for(int i:{1,2,4,5})hi[i]=0;b[1]=0;}
 auto initial=x;projection_recovery_v2::Stats s;bool ok=projection_recovery_v2::solve(A,b,x,hi,dep,1e-8,s,budget,budget);bool unchanged=true;for(int i=0;i<n;i++)unchanged&=x[i]==initial[i];bool pass=ok==expected&&(ok||unchanged)&&s.svd_calls<=std::max(0,std::min(budget,2048));if(ok&&mode==0)pass&=std::abs(x[0]-1)<1e-8&&std::abs(x[1]-.2)<1e-8;if(ok&&mode==1)pass&=unchanged;all&=pass;cases.push_back({{"name",name},{"passed",pass},{"accepted",ok},{"unchanged_on_decline",ok||unchanged},{"svd_calls",s.svd_calls}});};
 test("diagonal-stick",3,0,1024,true);test("separating",3,1,1024,true);test("zero-budget",3,0,0,false);test("negative-budget",3,0,-1,false);test("cap-66",66,0,1024,false);test("negative-friction",3,2,1024,false);test("nan-matrix",3,3,1024,false);test("invalid-dependency",3,4,1024,false);test("inconsistent-PSD-normal-targets",6,5,64,false);
 std::cout<<J({{"passed",all},{"controls",cases}}).dump()<<'\n';return all?0:2;}
