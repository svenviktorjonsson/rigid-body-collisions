// Research-only complete minimum-norm QR Newton direction; physical A untouched.
#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>
namespace trial_qr {
struct Stats {int calls=0,accepted=0,rejected=0,budget_rejections=0,gram_calls=0,gram_rejections=0,orthogonality_rejections=0,model_rejections=0;};
inline Stats stats;
struct Solution {std::vector<double>step;int rank=0;bool converged=false;double model_residual_square=0;};
inline bool positiveSolve(std::vector<double>H,std::vector<double>& rhs,int n){
 double scale=0;for(int i=0;i<n;i++)scale=std::max(scale,H[i*n+i]);if(!(scale>0&&std::isfinite(scale)))return false;
 for(int i=0;i<n;i++)for(int j=0;j<=i;j++){
  double value=H[i*n+j];for(int k=0;k<j;k++)value-=H[i*n+k]*H[j*n+k];
  if(i==j){if(!(value>1e-14*scale&&std::isfinite(value)))return false;H[i*n+j]=std::sqrt(value);}
  else H[i*n+j]=value/H[j*n+j];
 }
 for(int i=0;i<n;i++){for(int j=0;j<i;j++)rhs[i]-=H[i*n+j]*rhs[j];rhs[i]/=H[i*n+i];}
 for(int i=n-1;i>=0;i--){for(int j=i+1;j<n;j++)rhs[i]-=H[j*n+i]*rhs[j];rhs[i]/=H[i*n+i];}
 for(double x:rhs)if(!std::isfinite(x))return false;return true;
}
inline Solution direction(const std::vector<double>& matrix,const std::vector<double>& rhs,int n,double relative_cutoff=1e-12){
 Solution out;out.step.assign(n,0);
 if(stats.calls>=1024){stats.budget_rejections++;return out;}stats.calls++;
 if(n<=0||n>384||matrix.size()!=static_cast<size_t>(n*n)||rhs.size()!=static_cast<size_t>(n))return out;
 double entry_scale=0;for(double x:matrix){if(!std::isfinite(x))return out;entry_scale=std::max(entry_scale,std::abs(x));}
 for(double x:rhs)if(!std::isfinite(x))return out;if(!(entry_scale>0))return out;
 auto R=matrix;auto y=rhs;for(double& x:R)x/=entry_scale;for(double& x:y){x/=entry_scale;if(!std::isfinite(x))return out;}
 std::vector<int> permutation(n);std::iota(permutation.begin(),permutation.end(),0);double scale=0;
 for(int j=0;j<n;j++){double square=0;for(int i=0;i<n;i++)square+=R[i*n+j]*R[i*n+j];scale=std::max(scale,std::sqrt(square));}
 for(int k=0;k<n;k++){
  int pivot=k;double largest_square=0;
  for(int j=k;j<n;j++){double square=0;for(int i=k;i<n;i++)square+=R[i*n+j]*R[i*n+j];if(square>largest_square){largest_square=square;pivot=j;}}
  const double largest=std::sqrt(largest_square);if(largest<=relative_cutoff*scale)break;
  if(pivot!=k){for(int i=0;i<n;i++)std::swap(R[i*n+k],R[i*n+pivot]);std::swap(permutation[k],permutation[pivot]);}
  const double alpha=-std::copysign(largest,R[k*n+k]);std::vector<double>v(n-k);for(int i=k;i<n;i++)v[i-k]=R[i*n+k];v[0]-=alpha;
  double square=0;for(double x:v)square+=x*x;double norm=std::sqrt(square);if(!(norm>0&&std::isfinite(norm)))return out;for(double& x:v)x/=norm;
  for(int j=k;j<n;j++){double product=0;for(int i=k;i<n;i++)product+=v[i-k]*R[i*n+j];for(int i=k;i<n;i++)R[i*n+j]-=2*v[i-k]*product;}
  double product=0;for(int i=k;i<n;i++)product+=v[i-k]*y[i];for(int i=k;i<n;i++)y[i]-=2*v[i-k]*product;
  R[k*n+k]=alpha;for(int i=k+1;i<n;i++)R[i*n+k]=0;out.rank++;
 }
 const int r=out.rank,q=n-r;if(r==0)return out;std::vector<double>a(r),z(n,0);
 for(int i=r-1;i>=0;i--){double value=y[i];for(int j=i+1;j<r;j++)value-=R[i*n+j]*a[j];a[i]=value/R[i*n+i];}
 std::vector<double>T(r*q,0);
 for(int f=0;f<q;f++)for(int i=r-1;i>=0;i--){double value=R[i*n+r+f];for(int j=i+1;j<r;j++)value-=R[i*n+j]*T[j*q+f];T[i*q+f]=value/R[i*n+i];}
 for(double x:T)if(!std::isfinite(x))return out;
 if(q==0)std::copy(a.begin(),a.end(),z.begin());
 else{
  stats.gram_calls++;
  if(q<=r){
   std::vector<double>G(q*q,0),free(q,0);
   for(int f=0;f<q;f++){for(int i=0;i<r;i++)free[f]+=T[i*q+f]*a[i];for(int h=0;h<q;h++){double value=f==h?1.:0.;for(int i=0;i<r;i++)value+=T[i*q+f]*T[i*q+h];G[f*q+h]=value;}}
   if(!positiveSolve(G,free,q)){stats.gram_rejections++;return out;}
   for(int i=0;i<r;i++){z[i]=a[i];for(int f=0;f<q;f++)z[i]-=T[i*q+f]*free[f];}
   for(int f=0;f<q;f++)z[r+f]=free[f];
  }else{
   std::vector<double>G(r*r,0),retained=a;
   for(int i=0;i<r;i++)for(int j=0;j<r;j++){double value=i==j?1.:0.;for(int f=0;f<q;f++)value+=T[i*q+f]*T[j*q+f];G[i*r+j]=value;}
   if(!positiveSolve(G,retained,r)){stats.gram_rejections++;return out;}
   std::copy(retained.begin(),retained.end(),z.begin());for(int f=0;f<q;f++)for(int i=0;i<r;i++)z[r+f]+=T[i*q+f]*retained[i];
  }
  // Orthogonality to the discarded coordinate nullspace [-T; identity]
  // certifies the minimum-norm completion, independently of its linear fit.
  for(int f=0;f<q;f++){
   double defect=z[r+f],bound=std::abs(z[r+f]);for(int i=0;i<r;i++){double term=T[i*q+f]*z[i];defect-=term;bound+=std::abs(term);}
   if(!std::isfinite(defect)||std::abs(defect)>1e-10*std::max(bound,std::numeric_limits<double>::min())){stats.orthogonality_rejections++;return out;}
  }
 }
 for(int i=0;i<n;i++)out.step[permutation[i]]=z[i];double old_square=0;for(double x:rhs)old_square+=x*x;
 for(int i=0;i<n;i++){double residual=-rhs[i];for(int j=0;j<n;j++)residual+=matrix[i*n+j]*out.step[j];out.model_residual_square+=residual*residual;}
 out.converged=std::isfinite(out.model_residual_square)&&std::isfinite(old_square)&&old_square>0&&out.model_residual_square<=old_square*(1-1e-3);
 for(double x:out.step)out.converged&=std::isfinite(x);if(!out.converged)stats.model_rejections++;
 return out;
}
}
