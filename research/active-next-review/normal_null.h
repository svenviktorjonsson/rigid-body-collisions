// Bounded PSD normal active-face search. No physical mobility regularization.
#pragma once
#include "../../spatial_backend/newton_linear.h"
#include <LinearMath/btMatrixX.h>
#include <limits>
namespace normal_null {
struct Stats {int states=0,svd_calls=0,null_steps=0,range_steps=0,released=0,entered=0,svd_rejections=0,budget_rejections=0;double residual=std::numeric_limits<double>::infinity(),maximum_null_velocity_change=0;};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,const btVectorXu&upper,const btVectorXu&seed,btVectorXu&out,double tolerance,Stats&stats,int limit=128){
 const int n=b.rows();if(n<=0||n>384||A.rows()!=n||A.cols()!=n||upper.rows()!=n||seed.rows()!=n||out.rows()!=n||limit<=0||limit>512||!(tolerance>0)||!std::isfinite(tolerance))return false;
 std::vector<double>q(n),w(n);std::vector<bool>active(n);
 for(int i=0;i<n;i++){if(!(A(i,i)>0)||!std::isfinite(b[i])||!std::isfinite(seed[i])||!std::isfinite(upper[i])||upper[i]<0)return false;for(int j=0;j<n;j++)if(!std::isfinite(A(i,j)))return false;q[i]=std::max(0.,static_cast<double>(seed[i]));active[i]=q[i]>0;}
 auto gate=[&](){double residual=0,energy=0,scale=1;bool bounds=true;for(int i=0;i<n;i++){w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*q[j];if(!std::isfinite(w[i])||!std::isfinite(q[i]))return false;bounds&=q[i]>=0&&q[i]<=upper[i];residual=std::max(residual,std::abs(q[i]-std::max(0.,q[i]-w[i]/A(i,i)))*A(i,i));energy+=.5*q[i]*(w[i]-b[i]);scale+=std::abs(q[i]*b[i]);}stats.residual=residual;return bounds&&std::isfinite(residual)&&std::isfinite(energy)&&std::isfinite(scale)&&residual<=tolerance&&energy<=tolerance*scale;};
 for(int state=0;state<limit;state++){
  stats.states++;if(gate()){for(int i=0;i<n;i++)out[i]=q[i];return true;}
  double activeError=0;int entering=0;bool any=false;for(int i=0;i<n;i++){if(w[i]<w[entering])entering=i;if(active[i]){any=true;activeError=std::max(activeError,std::abs(w[i]));}}
  if(!any||activeError<tolerance*.1){if(w[entering]>=-tolerance)return false;active[entering]=true;stats.entered++;}
  std::vector<int>free;for(int i=0;i<n;i++)if(active[i])free.push_back(i);const int m=free.size();std::vector<double>M(m*m),rhs(m);for(int i=0;i<m;i++){rhs[i]=-w[free[i]];for(int j=0;j<m;j++)M[i*m+j]=A(free[i],free[j]);}
  stats.svd_calls++;auto linear=minimumNormNewton(M,rhs,m,1e-13);if(!linear.converged){stats.svd_rejections++;return false;}
  std::vector<double>delta(m,0);double magnitude=0;
  for(auto&basis:linear.nullspace){double dot=0;for(int i=0;i<m;i++)dot+=basis[i]*rhs[i];for(int i=0;i<m;i++)delta[i]+=basis[i]*dot;}
  for(double v:delta)magnitude=std::max(magnitude,std::abs(v));const bool nullStep=magnitude>1e-12;
  if(nullStep){for(double&v:delta)v/=magnitude;stats.null_steps++;}else{delta=linear.step;stats.range_steps++;}
  double alpha=nullStep?std::numeric_limits<double>::infinity():1.;for(int i=0;i<m;i++)if(delta[i]<0)alpha=std::min(alpha,q[free[i]]/-delta[i]);if(!std::isfinite(alpha)||alpha<0)return false;
  if(nullStep){double response=0;for(int i=0;i<n;i++){double value=0;for(int j=0;j<m;j++)value+=A(i,free[j])*delta[j]*alpha;response=std::max(response,std::abs(value));}stats.maximum_null_velocity_change=std::max(stats.maximum_null_velocity_change,response);}
  bool removed=false;for(int i=0;i<m;i++){int k=free[i];q[k]+=alpha*delta[i];if(delta[i]<0&&q[k]<=1e-12){q[k]=0;active[k]=false;stats.released++;removed=true;}}
  if(alpha==0&&!removed)return false;
 }
 stats.budget_rejections++;return false;
}
}
