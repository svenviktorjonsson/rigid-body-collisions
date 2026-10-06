#pragma once
#include "coulomb.h"
namespace component_polish_retry {
struct Stats {int components=0,largest_rows=0,iteration_steps=0,svd_calls=0,polish_steps=0;};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&p,const btVectorXu&lo,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats){
 const int n=b.rows();if(n<=0||n>4096)return false;std::vector<bool>visited(n,false);btVectorXu candidate=p;
 for(int first=0;first<n;first++)if(!visited[first]){
  std::vector<int>ids{first};visited[first]=true;
  for(size_t at=0;at<ids.size();at++)for(int j=0;j<n;j++)if(!visited[j]&&(A(ids[at],j)!=0||A(j,ids[at])!=0||dep[j]==ids[at]||dep[ids[at]]==j)){visited[j]=true;ids.push_back(j);}
  int m=ids.size();stats.components++;stats.largest_rows=std::max(stats.largest_rows,m);std::vector<int>inverse(n,-1);for(int i=0;i<m;i++)inverse[ids[i]]=i;
  btMatrixXu M(m,m);btVectorXu rhs(m),q(m),lower(m),upper(m);btAlignedObjectArray<int>d;d.resize(m);
  for(int i=0;i<m;i++){rhs[i]=b[ids[i]];q[i]=candidate[ids[i]];lower[i]=lo[ids[i]];upper[i]=hi[ids[i]];d[i]=dep[ids[i]]<0?-1:inverse[dep[ids[i]]];for(int j=0;j<m;j++)M.setElem(i,j,A(ids[i],ids[j]));}
  CoulombStats gate;btVectorXu checked=q;if(coulombIterate(M,rhs,checked,lower,upper,d,0,tol,gate)){for(int i=0;i<m;i++)candidate[ids[i]]=checked[i];continue;}
  if(m>192)return false;CoulombStats iteration;std::vector<double>rejected;bool found=coulombIterate(M,rhs,q,lower,upper,d,256,tol,iteration,&rejected);stats.iteration_steps+=iteration.iteration_sweeps_total;
  if(!found){if(rejected.size()!=static_cast<size_t>(m))return false;for(int i=0;i<m;i++)q[i]=rejected[i];CoulombStats polish;found=circular_polish::solve(M,rhs,q,upper,d,tol,polish);stats.svd_calls+=polish.polish_svd_calls;stats.polish_steps+=polish.polish_steps;}
  CoulombStats final;if(!found||!coulombIterate(M,rhs,q,lower,upper,d,0,tol,final))return false;for(int i=0;i<m;i++)candidate[ids[i]]=q[i];
 }
 CoulombStats final;if(!coulombIterate(A,b,candidate,lo,hi,dep,0,tol,final))return false;p=candidate;return true;
}
}
