// Project extension: mechanically gated normal-only QP; no diagonal regularization.
// Bullet's unmodified frictional MLCP remains the fallback for unsupported cases.
#pragma once
#include <BulletDynamics/MLCPSolvers/btMLCPSolver.h>
#include <algorithm>
#include <cmath>
#include <vector>

inline bool normalQP(const btMatrixXu& A,const btVectorXu& b,const btVectorXu& upper,btVectorXu& x){
 const int n=b.rows();double scale=1;
 for(int i=0;i<n;i++)scale=std::max(scale,std::abs(static_cast<double>(b[i])));
 const double tolerance=1e-10*scale;
 bool separating=true;for(int i=0;i<n;i++)separating &= b[i]<=tolerance;
 if(separating){for(int i=0;i<n;i++)x[i]=0;return true;}
 auto solve=[&](const std::vector<int>& free,std::vector<double>& answer){
  const int k=static_cast<int>(free.size());std::vector<double>L(k*k,0),y(k,0);answer.assign(k,0);
  for(int i=0;i<k;i++)for(int j=0;j<=i;j++){
   double value=A(free[i],free[j]);for(int z=0;z<j;z++)value-=L[i*k+z]*L[j*k+z];
   if(i==j){if(value<=1e-14*std::abs(static_cast<double>(A(free[i],free[i]))))return false;L[i*k+j]=std::sqrt(value);}
   else L[i*k+j]=value/L[j*k+j];
  }
  for(int i=0;i<k;i++){double value=b[free[i]];for(int j=0;j<i;j++)value-=L[i*k+j]*y[j];y[i]=value/L[i*k+i];}
  for(int i=k-1;i>=0;i--){double value=y[i];for(int j=i+1;j<k;j++)value-=L[j*k+i]*answer[j];answer[i]=value/L[i*k+i];}
  return true;
 };
 auto accepted=[&](const std::vector<double>& solution){
  double impulse_scale=1;for(double value:solution)impulse_scale=std::max(impulse_scale,std::abs(value));
  for(int i=0;i<n;i++){
   if(!std::isfinite(solution[i])||solution[i]<0||solution[i]>upper[i])return false;
   double residual=-b[i];for(int j=0;j<n;j++)residual+=A(i,j)*solution[j];
   if(!std::isfinite(residual)||residual<-tolerance||std::abs(residual*solution[i])>tolerance*impulse_scale)return false;
  }
  for(int i=0;i<n;i++)x[i]=solution[i];
  return true;
 };
 std::vector<double> solution(n,0),answer;
 if(accepted(solution))return true;
 // Frequent compressive SPD case: one factorization solves the entire island.
 std::vector<int> all(n);for(int i=0;i<n;i++)all[i]=i;
 if(solve(all,answer)){
  for(double& value:answer)if(value<0&&value>=-1e-12)value=0;
  if(accepted(answer))return true;
 }
 // Bound active set also admits redundant inactive contact rows (pressure gauge).
 std::vector<bool> passive(n,false);std::vector<int> free;
 for(int outer=0;outer<4*n+4;outer++){
  int entering=-1;double worst=-tolerance;
  for(int i=0;i<n;i++)if(!passive[i]){double w=-b[i];for(int j=0;j<n;j++)w+=A(i,j)*solution[j];if(w<worst){worst=w;entering=i;}}
  if(entering<0)return accepted(solution);
  passive[entering]=true;
  for(int inner=0;inner<4*n+4;inner++){
   free.clear();for(int i=0;i<n;i++)if(passive[i])free.push_back(i);
   if(!solve(free,answer))return false; // Singular active face: disclose fallback.
   double alpha=1;bool feasible=true;
   for(size_t j=0;j<free.size();j++)if(answer[j]<=0){feasible=false;int i=free[j];double denominator=solution[i]-answer[j];if(denominator>0)alpha=std::min(alpha,solution[i]/denominator);}
   if(feasible){std::fill(solution.begin(),solution.end(),0);for(size_t j=0;j<free.size();j++)solution[free[j]]=answer[j];break;}
   for(size_t j=0;j<free.size();j++){int i=free[j];solution[i]+=alpha*(answer[j]-solution[i]);if(solution[i]<=1e-12){solution[i]=0;passive[i]=false;}}
   if(inner==4*n+3)return false;
  }
 }
 return false;
}

class NormalFirstDantzig : public btDantzigSolver {
public:
 int normal_solves=0,normal_rejections=0;
 NormalFirstDantzig(){m_acceptableUpperLimitSolution=btScalar(1e30);}
 bool solveMLCP(const btMatrixXu& A,const btVectorXu& b,btVectorXu& x,const btVectorXu& lo,const btVectorXu& hi,const btAlignedObjectArray<int>& dependency,int iterations,bool sparse=true) override {
  int n=b.rows();std::vector<int>keep,map(n,-1);
  for(int i=0;i<n;i++)if(!(lo[i]==0&&hi[i]==0)){map[i]=static_cast<int>(keep.size());keep.push_back(i);}
  if(static_cast<int>(keep.size())==n){
   bool normal_only=true;for(int i=0;i<n;i++)normal_only &= dependency[i]<0&&lo[i]==0;
   if(normal_only){if(normalQP(A,b,hi,x)){normal_solves++;return true;}normal_rejections++;}
   return btDantzigSolver::solveMLCP(A,b,x,lo,hi,dependency,iterations,sparse);
  }
  int k=static_cast<int>(keep.size());btMatrixXu reduced(k,k);btVectorXu rhs(k),solution(k),lower(k),upper(k);btAlignedObjectArray<int>dep;dep.resize(k);bool normal_only=true;
  for(int i=0;i<k;i++){
   int row=keep[i];rhs[i]=b[row];solution[i]=x[row];lower[i]=lo[row];upper[i]=hi[row];
   if(dependency[row]>=0&&map[dependency[row]]<0)return btDantzigSolver::solveMLCP(A,b,x,lo,hi,dependency,iterations,sparse);
   dep[i]=dependency[row]<0?-1:map[dependency[row]];
   normal_only &= dep[i]<0&&lower[i]==0;
   for(int j=0;j<k;j++)reduced.setElem(i,j,A(row,keep[j]));
  }
  bool result=false;
  if(normal_only){result=normalQP(reduced,rhs,upper,solution);if(result)normal_solves++;else normal_rejections++;}
  if(!result)result=btDantzigSolver::solveMLCP(reduced,rhs,solution,lower,upper,dep,iterations,sparse);
  if(result)for(int i=0;i<n;i++)x[i]=map[i]<0?0:solution[map[i]];
  return result;
 }
};

// Record the mobility matrix's scalar payload, not process RSS or all work buffers.
class RecordedMLCP : public btMLCPSolver {
protected:
 void createMLCPFast(const btContactSolverInfo& info) override {btMLCPSolver::createMLCPFast(info);rows_max=std::max(rows_max,m_A.rows());}
 void createMLCP(const btContactSolverInfo& info) override {btMLCPSolver::createMLCP(info);rows_max=std::max(rows_max,m_A.rows());}
public:
 int rows_max=0;
 explicit RecordedMLCP(btMLCPSolverInterface* solver):btMLCPSolver(solver){}
};

// Remove exactly zero tangent impulses before assembling the dense mobility.
class NormalMLCP : public RecordedMLCP {
 void stripZeroRows(){
  for(int i=0;i<m_allConstraintPtrArray.size();i++){
   const auto& c=*m_allConstraintPtrArray[i];
   if(!(c.m_lowerLimit==0&&c.m_upperLimit==0)&&m_limitDependencies[i]>=0)return;
  }
  int old=m_allConstraintPtrArray.size(),count=0;
  for(int i=0;i<old;i++){
   auto* c=m_allConstraintPtrArray[i];
   if(c->m_lowerLimit==0&&c->m_upperLimit==0)continue;
   m_allConstraintPtrArray[count]=c;m_limitDependencies[count++]=-1;
  }
  m_allConstraintPtrArray.resize(count);m_limitDependencies.resize(count);
  removed_rows_max=std::max(removed_rows_max,old-count);
 }
protected:
 void createMLCPFast(const btContactSolverInfo& info) override {stripZeroRows();RecordedMLCP::createMLCPFast(info);}
 void createMLCP(const btContactSolverInfo& info) override {stripZeroRows();RecordedMLCP::createMLCP(info);}
public:
 int removed_rows_max=0;
 explicit NormalMLCP(btMLCPSolverInterface* solver):RecordedMLCP(solver){}
};
