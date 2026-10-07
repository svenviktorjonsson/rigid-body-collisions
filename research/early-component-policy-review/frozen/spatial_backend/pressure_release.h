// Prototype outside frozen native glob. Original normal LCP/energy gate only.
#pragma once
#include "newton_linear.h"
#include <LinearMath/btMatrixX.h>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
namespace normal_pressure {
struct Stats {int attempts=0,svd_calls=0,released=0;double residual=std::numeric_limits<double>::infinity();};
inline bool solve(const btMatrixXu& A,const btVectorXu& b,const btVectorXu& upper,
 const btVectorXu& seed,btVectorXu& out,double tolerance,Stats& stats,int attempt_limit=128){
 const int n=b.rows();if(attempt_limit<=0||n<=0||n>128||A.rows()!=n||A.cols()!=n||upper.rows()!=n||seed.rows()!=n||out.rows()!=n||!(tolerance>0)||!std::isfinite(tolerance))return false;
 std::vector<int>active;std::vector<double>w(n);for(int i=0;i<n;i++){
  if(!(A(i,i)>0)||!std::isfinite(seed[i])||!std::isfinite(b[i])||!(upper[i]>=0))return false;
  for(int j=0;j<n;j++)if(!std::isfinite(A(i,j)))return false;
  w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*std::max(0.,static_cast<double>(seed[j]));
  if(seed[i]>0)active.push_back(i);
 }
 std::vector<int>release=active;
 std::stable_sort(release.begin(),release.end(),[&](int i,int j){
  const double a=std::max(0.,w[i])*seed[i],bb=std::max(0.,w[j])*seed[j];
  return a!=bb?a>bb:std::abs(w[i])>std::abs(w[j]);
 });
 if(release.size()>16)release.resize(16);
 int attempts=0;
 auto candidate=[&](const std::vector<int>&removed){
  if(attempts>=std::min(128,attempt_limit))return false;
  attempts++;stats.attempts++;
  std::vector<int>keep;for(int i:active)if(std::find(removed.begin(),removed.end(),i)==removed.end())keep.push_back(i);
  const int m=static_cast<int>(keep.size());std::vector<double>p(n,0);
  if(m){
   std::vector<double>M(m*m),rhs(m);for(int i=0;i<m;i++){rhs[i]=b[keep[i]];for(int j=0;j<m;j++)M[i*m+j]=A(keep[i],keep[j]);}
   stats.svd_calls++;auto linear=minimumNormNewton(M,rhs,m,1e-13);if(!linear.converged)return false;
   for(int i=0;i<m;i++)p[keep[i]]=linear.step[i];
  }
  for(int i=0;i<n;i++){
   if(!std::isfinite(p[i])||p[i]>upper[i]||p[i]<-tolerance/A(i,i))return false;
   if(p[i]<0)p[i]=0;
  }
  double error=0,change=0,scale=1;
  for(int i=0;i<n;i++){
   double v=-b[i];for(int j=0;j<n;j++)v+=A(i,j)*p[j];
   error=std::max(error,std::abs(p[i]-std::max(0.,p[i]-v/A(i,i)))*A(i,i));
   change+=.5*p[i]*(v-b[i]);scale+=std::abs(p[i]*b[i]);
  }
  stats.residual=std::min(stats.residual,error);
  if(!std::isfinite(error)||!std::isfinite(change)||!std::isfinite(scale)||error>tolerance||change>tolerance*scale)return false;
  for(int i=0;i<n;i++)out[i]=p[i];
  stats.residual=error;stats.released=removed.size();return true;
 };
 if(candidate({}))return true;
 for(int i:release)if(candidate({i}))return true;
 for(size_t i=0;i<release.size();i++)for(size_t j=i+1;j<release.size();j++)if(candidate({release[i],release[j]}))return true;
 for(size_t i=0;i<release.size();i++)for(size_t j=i+1;j<release.size();j++)for(size_t k=j+1;k<release.size();k++)if(candidate({release[i],release[j],release[k]}))return true;
 return false;
}
}
