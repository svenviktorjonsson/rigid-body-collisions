// Bounded numerical mobility-null seeds. Original equations and final gates unchanged.
#pragma once
#include "projection192.h"
#include "newton_linear.h"
namespace null_seed192 {
using Stats=null_traction_seed::Stats;
inline bool solve(const btMatrixXu&A,const btVectorXu&b,const btVectorXu&seed,const btVectorXu&hi,
 const btAlignedObjectArray<int>&dep,btVectorXu&out,double tolerance,Stats&stats){
 const int n=b.rows();if(n<=0||n>4096||seed.rows()!=n||out.rows()!=n||A.rows()!=n||A.cols()!=n||hi.rows()!=n||dep.size()!=n)return false;
 std::vector<bool>visited(n,false);btVectorXu candidate=seed;
 for(int first=0;first<n;first++)if(!visited[first]){
  std::vector<int>ids{first};visited[first]=true;
  for(size_t at=0;at<ids.size();at++)for(int j=0;j<n;j++)if(!visited[j]&&(A(ids[at],j)!=0||A(j,ids[at])!=0||dep[j]==ids[at]||dep[ids[at]]==j)){visited[j]=true;ids.push_back(j);}
  const int m=ids.size();stats.components++;stats.largest_rows=std::max(stats.largest_rows,m);
  if(m>192){stats.cap_rejections++;return false;}
  std::vector<int>inverse(n,-1);for(int i=0;i<m;i++)inverse[ids[i]]=i;
  btMatrixXu M(m,m);btVectorXu rhs(m),p(m),upper(m);btAlignedObjectArray<int>d;d.resize(m);
  for(int i=0;i<m;i++){rhs[i]=b[ids[i]];p[i]=seed[ids[i]];upper[i]=hi[ids[i]];d[i]=dep[ids[i]]<0?-1:inverse[dep[ids[i]]];for(int j=0;j<m;j++)M.setElem(i,j,A(ids[i],ids[j]));}
  if(!restart_validation::valid(M,rhs,p,upper,d,tolerance,192))return false;
  auto search=[&](btVectorXu&q){projection192::Stats s;const bool found=projection192::solve(M,rhs,q,upper,d,tolerance,s,2048,2048,true,true,true);stats.iteration_steps+=s.iteration_steps;stats.svd_calls+=s.svd_calls;stats.newton_steps+=s.newton_steps;return found;};
  bool found=search(p);
  if(!found){
   std::vector<double>matrix(m*m),zero(m,0),warm(m);for(int i=0;i<m;i++){warm[i]=seed[ids[i]];for(int j=0;j<m;j++)matrix[i*m+j]=M(i,j);}
   stats.null_svd_calls++;auto gauge=minimumNormNewton(matrix,zero,m,1e-12);if(!gauge.converged||gauge.nullspace.empty())return false;
   struct Target{int k,t,s;double error;};std::vector<Target>targets;
   for(int k=0;k<m;k++)if(d[k]<0){
    std::vector<int>ts;for(int j=0;j<m;j++)if(d[j]==k)ts.push_back(j);const int t=ts[0],s=ts[1];const int rows[3]={k,t,s};double w[3]={-rhs[k],-rhs[t],-rhs[s]};for(int z=0;z<3;z++)for(int j=0;j<m;j++)w[z]+=M(rows[z],j)*warm[j];
    const double eig=.5*(M(t,t)+M(s,s)+std::hypot(M(t,t)-M(s,s),2*M(t,s))),zn=warm[k]-w[0]/M(k,k),zt=warm[t]-w[1]/eig,zs=warm[s]-w[2]/eig,cap=upper[t]*std::max(0.,warm[k]),length=std::hypot(zt,zs),factor=length>cap?cap/length:1.;
    const double error=std::max(std::abs(warm[k]-std::max(0.,zn))*M(k,k),std::hypot(warm[t]-zt*factor,warm[s]-zs*factor)*eig);targets.push_back({k,t,s,error});
   }
   std::stable_sort(targets.begin(),targets.end(),[](const Target&a,const Target&b){return a.error>b.error;});
   int tries=0;for(auto c:targets){
    if(found||tries++>=6)break;stats.seed_attempts++;const int rows[3]={c.k,c.t,c.s};std::vector<double>gram(9,0),target(3);
    for(int i=0;i<3;i++){target[i]=-warm[rows[i]];for(int j=0;j<3;j++)for(const auto&v:gauge.nullspace)gram[i*3+j]+=v[rows[i]]*v[rows[j]];}
    stats.seed_svd_calls++;auto move=minimumNormNewton(gram,target,3,1e-12);if(!move.converged)continue;
    btVectorXu trial(m);for(int i=0;i<m;i++){double delta=0;for(const auto&v:gauge.nullspace){double coef=0;for(int j=0;j<3;j++)coef+=v[rows[j]]*move.step[j];delta+=v[i]*coef;}trial[i]=warm[i]+delta;}
    for(int k:rows)trial[k]=0;
    for(auto contact:targets){const int k=contact.k,t=contact.t,s=contact.s;trial[k]=std::max(0.,static_cast<double>(trial[k]));double length=std::hypot(trial[t],trial[s]),cap=upper[t]*trial[k];if(length>cap){trial[t]*=cap/length;trial[s]*=cap/length;}}
    for(int i=0;i<m;i++){double change=0;for(int j=0;j<m;j++)change+=M(i,j)*(trial[j]-warm[j]);stats.seed_response_change_max=std::max(stats.seed_response_change_max,std::abs(change));}
    found=search(trial);if(found)p=trial;
   }
  }
  if(!found)return false;for(int i=0;i<m;i++)candidate[ids[i]]=p[i];
 }
 out=candidate;return true;
}
}
