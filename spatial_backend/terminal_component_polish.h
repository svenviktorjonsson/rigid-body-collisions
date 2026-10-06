#pragma once
#include "normal_qp.h"
namespace terminal_component_polish {
struct Stats {int components=0,largest_rows=0,iteration_steps=0,svd_calls=0,polish_steps=0;};
#ifdef SPATIAL_LAPACK_RECOVERY
template<class Gate,class Iterate,class Polish>
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&p,const btVectorXu&lo,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats,Gate original_gate,Iterate original_iteration,Polish original_polish){
 const int n=b.rows();if(n<=0||n>4096)return false;std::vector<bool>visited(n,false);btVectorXu candidate=p;
 for(int first=0;first<n;first++)if(!visited[first]){
  std::vector<int>ids{first};visited[first]=true;
  for(size_t at=0;at<ids.size();at++)for(int j=0;j<n;j++)if(!visited[j]&&(A(ids[at],j)!=0||A(j,ids[at])!=0||dep[j]==ids[at]||dep[ids[at]]==j)){visited[j]=true;ids.push_back(j);}
  int m=ids.size();stats.components++;stats.largest_rows=std::max(stats.largest_rows,m);std::vector<int>inverse(n,-1);for(int i=0;i<m;i++)inverse[ids[i]]=i;
  btMatrixXu M(m,m);btVectorXu rhs(m),q(m),lower(m),upper(m);btAlignedObjectArray<int>d;d.resize(m);
  for(int i=0;i<m;i++){rhs[i]=b[ids[i]];q[i]=candidate[ids[i]];lower[i]=lo[ids[i]];upper[i]=hi[ids[i]];d[i]=dep[ids[i]]<0?-1:inverse[dep[ids[i]]];for(int j=0;j<m;j++)M.setElem(i,j,A(ids[i],ids[j]));}
  btVectorXu checked=q;if(original_gate(M,rhs,checked,lower,upper,d,tol)){for(int i=0;i<m;i++)candidate[ids[i]]=checked[i];continue;}
  if(m>192)return false;std::vector<double>rejected;int sweeps=0;bool found=original_iteration(M,rhs,q,lower,upper,d,tol,rejected,sweeps);stats.iteration_steps+=sweeps;
  if(!found){if(rejected.size()!=static_cast<size_t>(m))return false;for(int i=0;i<m;i++)q[i]=rejected[i];int svds=0,steps=0;found=original_polish(M,rhs,q,upper,d,tol,svds,steps);stats.svd_calls+=svds;stats.polish_steps+=steps;}
  if(!found||!original_gate(M,rhs,q,lower,upper,d,tol))return false;for(int i=0;i<m;i++)candidate[ids[i]]=q[i];
 }
 if(!original_gate(A,b,candidate,lo,hi,dep,tol))return false;p=candidate;return true;
}
#endif
}
