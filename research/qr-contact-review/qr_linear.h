// Research-only numerical Newton direction. No physical mobility changes.
#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>
namespace trial_qr {
struct Stats {int calls=0,accepted=0,rejected=0,budget_rejections=0;};
inline Stats stats;
struct Solution {std::vector<double>step;int rank=0;bool converged=false;double model_residual_square=0;};
inline Solution direction(const std::vector<double>& matrix,const std::vector<double>& rhs,int n,double relative_cutoff=1e-12){
 Solution out;out.step.assign(n,0);
 if(stats.calls>=1024){stats.budget_rejections++;return out;}stats.calls++;
 if(n<=0||n>384||matrix.size()!=static_cast<size_t>(n*n)||rhs.size()!=static_cast<size_t>(n))return out;
 for(double x:matrix)if(!std::isfinite(x))return out;
 for(double x:rhs)if(!std::isfinite(x))return out;
 auto R=matrix;auto y=rhs;std::vector<int> permutation(n);std::iota(permutation.begin(),permutation.end(),0);
 double scale=0;for(int j=0;j<n;j++){double norm=0;for(int i=0;i<n;i++)norm=std::hypot(norm,R[i*n+j]);scale=std::max(scale,norm);}
 if(!(scale>0))return out;
 for(int k=0;k<n;k++){
  int pivot=k;double largest=0;
  // Recompute norms rather than downdating across almost dependent columns.
  for(int j=k;j<n;j++){double norm=0;for(int i=k;i<n;i++)norm=std::hypot(norm,R[i*n+j]);if(norm>largest){largest=norm;pivot=j;}}
  if(largest<=relative_cutoff*scale)break;
  if(pivot!=k){for(int i=0;i<n;i++)std::swap(R[i*n+k],R[i*n+pivot]);std::swap(permutation[k],permutation[pivot]);}
  const double alpha=-std::copysign(largest,R[k*n+k]);std::vector<double>v(n-k);
  for(int i=k;i<n;i++)v[i-k]=R[i*n+k];v[0]-=alpha;
  double norm=0;for(double x:v)norm=std::hypot(norm,x);if(!(norm>0&&std::isfinite(norm)))return out;for(double& x:v)x/=norm;
  for(int j=k;j<n;j++){double product=0;for(int i=k;i<n;i++)product+=v[i-k]*R[i*n+j];for(int i=k;i<n;i++)R[i*n+j]-=2*v[i-k]*product;}
  double product=0;for(int i=k;i<n;i++)product+=v[i-k]*y[i];for(int i=k;i<n;i++)y[i]-=2*v[i-k]*product;
  R[k*n+k]=alpha;for(int i=k+1;i<n;i++)R[i*n+k]=0;out.rank++;
 }
 std::vector<double>z(n,0);for(int i=out.rank-1;i>=0;i--){double value=y[i];for(int j=i+1;j<out.rank;j++)value-=R[i*n+j]*z[j];z[i]=value/R[i*n+i];}
 for(int i=0;i<n;i++)out.step[permutation[i]]=z[i];
 double old_square=0;for(double x:rhs)old_square+=x*x;
 for(int i=0;i<n;i++){double residual=-rhs[i];for(int j=0;j<n;j++)residual+=matrix[i*n+j]*out.step[j];out.model_residual_square+=residual*residual;}
 out.converged=out.rank>0&&std::isfinite(out.model_residual_square)&&out.model_residual_square<=old_square*(1+1e-10)+1e-30;
 for(double x:out.step)out.converged&=std::isfinite(x);
 return out;
}
}
