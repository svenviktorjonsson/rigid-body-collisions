// Isolated numerical regularization continuation. NEVER a physical mobility change.
#pragma once
#include "coulomb.h"
#include "projection_more.h"
namespace mobility_continuation {
struct Stats {int components=0,largest_rows=0,stage_attempts=0,stage_accepts=0,iteration_steps=0,svd_calls=0;};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&p,const btVectorXu&lo,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats){
 const int n=b.rows();if(n<=0||n>4096)return false;std::vector<bool>visited(n,false);btVectorXu candidate=p;
 for(int first=0;first<n;first++)if(!visited[first]){
  std::vector<int>ids{first};visited[first]=true;
  for(size_t at=0;at<ids.size();at++)for(int j=0;j<n;j++)if(!visited[j]&&(A(ids[at],j)!=0||A(j,ids[at])!=0||dep[j]==ids[at]||dep[ids[at]]==j)){visited[j]=true;ids.push_back(j);}
  int m=ids.size();stats.components++;stats.largest_rows=std::max(stats.largest_rows,m);if(m>192)return false;
  std::vector<int>inverse(n,-1);for(int i=0;i<m;i++)inverse[ids[i]]=i;
  btMatrixXu M(m,m);btVectorXu rhs(m),q(m),lower(m),upper(m);btAlignedObjectArray<int>d;d.resize(m);
  for(int i=0;i<m;i++){rhs[i]=b[ids[i]];q[i]=candidate[ids[i]];lower[i]=lo[ids[i]];upper[i]=hi[ids[i]];d[i]=dep[ids[i]]<0?-1:inverse[dep[ids[i]]];for(int j=0;j<m;j++)M.setElem(i,j,A(ids[i],ids[j]));}
  CoulombStats initial;btVectorXu checked=q;if(coulombIterate(M,rhs,checked,lower,upper,d,0,tol,initial)){for(int i=0;i<m;i++)candidate[ids[i]]=checked[i];continue;}
  for(double alpha:{.1,.03,.01,.003,.001,.0003,.0001,.00003,.00001,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,0.}){
   btMatrixXu search=M;for(int i=0;i<m;i++)search.setElem(i,i,M(i,i)*(1+alpha));
   projection_recovery_v2::Stats s;stats.stage_attempts++;bool found=projection_recovery_v2::solve(search,rhs,q,upper,d,tol,s,2048,2048,true,true,true,192);stats.stage_accepts+=found;stats.iteration_steps+=s.iteration_steps;stats.svd_calls+=s.svd_calls;
   if(!found&&s.failed_candidate.size()==static_cast<size_t>(m))for(int i=0;i<m;i++)q[i]=s.failed_candidate[i];
   // Numerical feasible seed only; actual unchanged law recomputed at the end.
   for(int k=0;k<m;k++)if(d[k]<0){q[k]=std::max(0.,static_cast<double>(q[k]));std::vector<int>ts;for(int j=0;j<m;j++)if(d[j]==k)ts.push_back(j);if(ts.size()!=2)return false;double length=std::hypot(q[ts[0]],q[ts[1]]),cap=upper[ts[0]]*q[k];if(length>cap){q[ts[0]]*=cap/length;q[ts[1]]*=cap/length;}}
  }
  CoulombStats original;if(!coulombIterate(M,rhs,q,lower,upper,d,0,tol,original))return false;
  for(int i=0;i<m;i++)candidate[ids[i]]=q[i];
 }
 CoulombStats final;if(!coulombIterate(A,b,candidate,lo,hi,dep,0,tol,final))return false;p=candidate;return true;
}
}
