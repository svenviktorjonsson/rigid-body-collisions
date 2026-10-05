#include "normal_null.h"
#include "translation_split.h"
#include <iostream>
#include <stdexcept>
void check(bool value,const char*message){if(!value)throw std::runtime_error(message);}
int main(){
 btMatrixXu A(2,2);btVectorXu b(2),hi(2),seed(2),out(2);for(int i=0;i<2;i++){hi[i]=1e30;seed[i]=out[i]=.5;for(int j=0;j<2;j++)A.setElem(i,j,1);}
 b[0]=1;b[1]=2;normal_null::Stats inconsistent;check(normal_null::solve(A,b,hi,seed,out,1e-8,inconsistent),"redundant unequal targets need pressure release");check(out[0]==0&&std::abs(out[1]-2)<1e-12&&inconsistent.null_steps==1,"wrong redundant pressure face");
 b[0]=b[1]=1;seed[0]=seed[1]=.2;normal_null::Stats compatible;check(normal_null::solve(A,b,hi,seed,out,1e-8,compatible),"compatible redundant target");check(std::abs(out[0]+out[1]-1)<1e-12,"wrong compatible total pressure");
 A.setElem(0,1,-1);A.setElem(1,0,-1);seed[0]=seed[1]=0;out[0]=4;out[1]=7;normal_null::Stats infeasible;check(!normal_null::solve(A,b,hi,seed,out,1e-8,infeasible),"opposing contradictory normals should reject");check(out[0]==4&&out[1]==7,"failed search modified caller");
 A.setElem(0,1,0);A.setElem(1,0,0);b[0]=2;b[1]=-1;normal_null::Stats independent;check(normal_null::solve(A,b,hi,seed,out,1e-8,independent),"independent compression/separation");check(std::abs(out[0]-2)<1e-12&&out[1]==0,"wrong independent result");
 out[0]=4;out[1]=7;hi[0]=1;normal_null::Stats bounded;check(!normal_null::solve(A,b,hi,seed,out,1e-8,bounded),"upper bound violation must reject");check(out[0]==4&&out[1]==7,"bounds rejection modified caller");
 hi[0]=1e30;normal_null::Stats budget;check(!normal_null::solve(A,b,hi,seed,out,1e-8,budget,1),"exhausted budget must reject");check(out[0]==4&&out[1]==7&&budget.budget_rejections==1,"budget rejection mutated caller or missing counter");
 b[0]=std::numeric_limits<double>::quiet_NaN();normal_null::Stats invalid;check(!normal_null::solve(A,b,hi,seed,out,1e-8,invalid),"nonfinite input");check(out[0]==4&&out[1]==7,"invalid input modified caller");
 {btMatrixXu M(3,3);const double values[9]={1,0,1,0,1,1,1,1,2};btVectorXu rhs(3),upper(3),answer(3);for(int i=0;i<3;i++){rhs[i]=i==2?5:3;upper[i]=1e30;answer[i]=0;for(int j=0;j<3;j++)M.setElem(i,j,values[i*3+j]);}
  normal_null::Stats enabled;check(translationSplitSolve(M,rhs,upper,answer,1e-8,64,nullptr,&enabled),"translation null-pressure repair failed");check(enabled.solves==1&&enabled.null_steps>0,"translation helper was not exercised");
  normal_null::Stats disabled;check(translationSplitSolve(M,rhs,upper,answer,1e-8,64,nullptr,&disabled,false),"disabled recovery changed ordinary projected solve");check(disabled.states==0&&disabled.svd_calls==0,"disabled recovery performed hidden SVD work");}
 std::cout<<"9 normal-null controls passed (including translation recovery enable/disable)\n";
}
