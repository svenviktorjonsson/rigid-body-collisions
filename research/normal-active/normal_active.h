// Numerical nonnegative convex-QP active-face search; original LCP gate only.
#pragma once
#include "newton_linear.h"
#include <LinearMath/btMatrixX.h>
#include <vector>
#include <cmath>
#include <limits>
namespace normal_active {
struct Stats {int svds=0,entries=0,releases=0;double residual=std::numeric_limits<double>::infinity();};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,const btVectorXu&upper,btVectorXu&out,double tol,Stats&stats){
 const int n=b.rows();if(n<=0||n>128)return false;std::vector<double>p(n,0),w(n);std::vector<bool>active(n,false);
 auto update=[&](){double error=0;for(int i=0;i<n;i++){w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*p[j];error=std::max(error,std::abs(p[i]-std::max(0.,p[i]-w[i]/A(i,i)))*A(i,i));}stats.residual=error;return error;};
 for(int iteration=0;iteration<512&&stats.svds<128;iteration++){
  const double error=update();if(error<=tol){double E=0,scale=1;for(int i=0;i<n;i++){if(!std::isfinite(p[i])||p[i]<0||p[i]>upper[i])return false;E+=.5*p[i]*(w[i]-b[i]);scale+=std::abs(p[i]*b[i]);}if(!std::isfinite(E)||!std::isfinite(scale)||E>tol*scale)return false;for(int i=0;i<n;i++)out[i]=p[i];return true;}
  int entering=-1;double worst=-tol;for(int i=0;i<n;i++)if(!active[i]&&w[i]<worst){worst=w[i];entering=i;}
  if(entering>=0){active[entering]=true;stats.entries++;}
  else{int released=-1;double score=0;for(int i=0;i<n;i++)if(active[i]&&w[i]>tol&&w[i]*p[i]>score){score=w[i]*p[i];released=i;}if(released<0)return false;active[released]=false;p[released]=0;stats.releases++;continue;}
  for(int inner=0;inner<128&&stats.svds<128;inner++){
   std::vector<int>free;for(int i=0;i<n;i++)if(active[i])free.push_back(i);int m=free.size();if(!m)break;
   std::vector<double>M(m*m),rhs(m);for(int i=0;i<m;i++){rhs[i]=b[free[i]];for(int j=0;j<m;j++)M[i*m+j]=A(free[i],free[j]);}stats.svds++;auto root=minimumNormNewton(M,rhs,m,1e-13);if(!root.converged)return false;
   double alpha=1;bool positive=true;for(int i=0;i<m;i++)if(root.step[i]<=0){positive=false;const double denominator=p[free[i]]-root.step[i];if(denominator>0)alpha=std::min(alpha,p[free[i]]/denominator);}
   if(positive){std::fill(p.begin(),p.end(),0);for(int i=0;i<m;i++)p[free[i]]=root.step[i];break;}
   for(int i=0;i<m;i++){int row=free[i];p[row]+=alpha*(root.step[i]-p[row]);if(p[row]<=1e-14){p[row]=0;active[row]=false;stats.releases++;}}
  }
 }
 update();return false;
}
}
