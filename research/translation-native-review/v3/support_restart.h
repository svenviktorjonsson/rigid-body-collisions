// Exact principal-support numerical search; ALL original equations are the gate.
#pragma once
#include "coulomb_restart.h"
namespace support_restart_v3 {
struct Stats {int helper_calls=0,skipped_components=0,component_cap_rejections=0,passes=0,largest_reduced_rows=0,expanded_contacts=0,svd_calls=0,iteration_steps=0,pressure_svd_calls=0,pressure_attempts=0,pivot_attempts=0;double residual=0;std::vector<int>row_counts,release_contacts,component_sizes;};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&x,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats){
 const int n=b.rows();if(!restart_validation::valid(A,b,x,hi,dep,tol,4096))return false;
 std::vector<int>normals;for(int i=0;i<n;i++)if(dep[i]<0)normals.push_back(i);
 std::vector<double>seed(n),candidate(n,0),w(n),warmw(n);std::vector<bool>active(n,false);
 for(int i=0;i<n;i++)seed[i]=x[i];
 for(int k:normals){double free=-b[k];for(int j=0;j<n;j++)free+=A(k,j)*seed[j];warmw[k]=free;active[k]=seed[k]>1e-9||free< -tol;}
 const auto original=seed;const auto initial_active=active;std::vector<int>release;
 for(int k:normals)if(active[k]&&seed[k]>1e-9&&warmw[k]>tol)release.push_back(k);
 std::stable_sort(release.begin(),release.end(),[&](int a,int c){return warmw[a]>warmw[c];});int released=0;bool base_tried=false;
 if(!release.empty()){active[release[released++]]=false;stats.release_contacts.push_back(release[0]);}else base_tried=true;
 auto fullGate=[&](){
  double error=0,energy=0,scale=1;
  for(int i=0;i<n;i++){w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*candidate[j];if(!std::isfinite(candidate[i])||!std::isfinite(w[i]))return false;energy+=.5*candidate[i]*(w[i]-b[i]);scale+=std::abs(candidate[i]*b[i]);}
  for(int k:normals){
   if(candidate[k]<0||candidate[k]>hi[k])return false;
   error=std::max(error,std::abs(candidate[k]-std::max(0.,candidate[k]-w[k]/A(k,k)))*A(k,k));
   std::vector<int>t;for(int i=0;i<n;i++)if(dep[i]==k)t.push_back(i);int u=t[0],v=t[1];double eig=.5*(A(u,u)+A(v,v)+std::hypot(A(u,u)-A(v,v),2*A(u,v)));if(!(eig>0))return false;
   double z0=candidate[u]-w[u]/eig,z1=candidate[v]-w[v]/eig,r=std::hypot(z0,z1),cap=hi[u]*candidate[k],factor=r>cap&&r>0?cap/r:1;
   error=std::max(error,std::hypot(candidate[u]-factor*z0,candidate[v]-factor*z1)*eig);
  }
  stats.residual=error;return std::isfinite(error)&&error<=tol&&std::isfinite(energy)&&std::isfinite(scale)&&energy<=tol*scale;
 };
 for(int pass=0;pass<8;pass++){
  std::vector<int>rows,map(n,-1);for(int k:normals)if(active[k]){rows.push_back(k);for(int i=0;i<n;i++)if(dep[i]==k)rows.push_back(i);}
  const int m=rows.size();stats.row_counts.push_back(m);stats.passes++;
  std::fill(candidate.begin(),candidate.end(),0);
  if(m>0){
   std::vector<int>parent(m);for(int i=0;i<m;i++)parent[i]=i;
   auto find=[&](int i){while(parent[i]!=i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
   auto unite=[&](int i,int j){int a=find(i),c=find(j);if(a!=c)parent[a]=c;};
   for(int i=0;i<m;i+=3){unite(i,i+1);unite(i,i+2);}
   for(int i=0;i<m;i++)for(int j=0;j<i;j++)if(A(rows[i],rows[j])!=0||A(rows[j],rows[i])!=0)unite(i,j);
   std::vector<std::vector<int>>groups(m);for(int i=0;i<m;i++)groups[find(i)].push_back(rows[i]);
   for(const auto& component:groups)if(!component.empty()){
    const int size=component.size();stats.component_sizes.push_back(size);stats.largest_reduced_rows=std::max(stats.largest_reduced_rows,size);
    if(size>64){stats.component_cap_rejections++;return false;}
    std::fill(map.begin(),map.end(),-1);for(int i=0;i<size;i++)map[component[i]]=i;
    btMatrixXu reduced(size,size);btVectorXu rhs(size),p(size),upper(size);btAlignedObjectArray<int>d;d.resize(size);
    for(int i=0;i<size;i++){int r=component[i];rhs[i]=b[r];p[i]=seed[r];upper[i]=hi[r];d[i]=dep[r]<0?-1:map[dep[r]];for(int j=0;j<size;j++)reduced.setElem(i,j,A(r,component[j]));}
    auto localGate=[&](){
     std::vector<double>velocity(size);double error=0,energy=0,scale=1;
     for(int i=0;i<size;i++){velocity[i]=-rhs[i];for(int j=0;j<size;j++)velocity[i]+=reduced(i,j)*p[j];if(!std::isfinite(p[i])||!std::isfinite(velocity[i]))return false;energy+=.5*p[i]*(velocity[i]-rhs[i]);scale+=std::abs(p[i]*rhs[i]);}
     for(int k=0;k<size;k++)if(d[k]<0){
      if(p[k]<0||p[k]>upper[k])return false;
      error=std::max(error,std::abs(p[k]-std::max(0.,static_cast<double>(p[k])-velocity[k]/reduced(k,k)))*reduced(k,k));
      std::vector<int>t;for(int i=0;i<size;i++)if(d[i]==k)t.push_back(i);int u=t[0],v=t[1];double eig=.5*(reduced(u,u)+reduced(v,v)+std::hypot(reduced(u,u)-reduced(v,v),2*reduced(u,v)));if(!(eig>0))return false;
      double z0=p[u]-velocity[u]/eig,z1=p[v]-velocity[v]/eig,r=std::hypot(z0,z1),cap=upper[u]*p[k],factor=r>cap&&r>0?cap/r:1;error=std::max(error,std::hypot(p[u]-factor*z0,p[v]-factor*z1)*eig);
     }
     return std::isfinite(error)&&error<=tol&&std::isfinite(energy)&&std::isfinite(scale)&&energy<=tol*scale;
    };
    if(localGate())stats.skipped_components++;
    else{
     if(stats.helper_calls>=8||1024-stats.svd_calls<256)return false;
     circular_restart::Stats inner;stats.helper_calls++;circular_restart::solve(reduced,rhs,p,upper,d,tol,inner,std::min(512,1024-stats.svd_calls));
     stats.svd_calls+=inner.svd_calls;stats.iteration_steps+=inner.iteration_steps;stats.pressure_svd_calls+=inner.neutral.pressure_svd_calls;stats.pressure_attempts+=inner.neutral.pressure_attempts;stats.pivot_attempts+=inner.neutral.normal_pivot_attempts;
    }
    for(int i=0;i<size;i++)candidate[component[i]]=p[i];
   }
  }
  if(fullGate()){for(int i=0;i<n;i++)x[i]=candidate[i];return true;}
  bool expanded=false;for(int k:normals)if(!active[k]&&w[k]< -tol){active[k]=true;stats.expanded_contacts++;expanded=true;}
  if(expanded){seed=candidate;continue;}
  active=initial_active;seed=original;
  if(released<std::min(2,static_cast<int>(release.size()))){active[release[released]]=false;stats.release_contacts.push_back(release[released++]);}
  else if(!base_tried)base_tried=true;
  else return false;
 }
 return false;
}
}
