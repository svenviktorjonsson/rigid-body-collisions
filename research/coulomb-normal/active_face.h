// Numerical reduced contact search; ONLY the full original contact gate accepts.
#pragma once
#include "active_trust.h"
namespace active_face {
struct Stats {int passes=0,active_contacts=0,expanded_contacts=0;double residual=std::numeric_limits<double>::infinity();active_trust::Stats search;};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&x,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats){
 const int n=b.rows();if(n<=0||n>384)return false;
 std::vector<int>normals;std::vector<bool>active(n,false);
 for(int k=0;k<n;k++)if(dep[k]<0){normals.push_back(k);active[k]=x[k]>0;}
 active_trust::Budget budget;
 std::vector<double>candidate(n,0),w(n,0);
 auto fullGate=[&](){
  double error=0,energy=0,scale=1;
  for(int i=0;i<n;i++){w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*candidate[j];if(!std::isfinite(candidate[i])||!std::isfinite(w[i]))return false;energy+=.5*candidate[i]*(w[i]-b[i]);scale+=std::abs(candidate[i]*b[i]);}
  for(int k:normals){
   if(candidate[k]<0||candidate[k]>hi[k]||!(A(k,k)>0))return false;
   error=std::max(error,std::abs(candidate[k]-std::max(0.,candidate[k]-w[k]/A(k,k)))*A(k,k));
   std::vector<int>ts;for(int j=0;j<n;j++)if(dep[j]==k)ts.push_back(j);if(ts.size()!=2)return false;
   const int u=ts[0],v=ts[1];double eig=.5*(A(u,u)+A(v,v)+std::hypot(A(u,u)-A(v,v),2*A(u,v)));if(!(eig>0)||hi[u]!=hi[v]||hi[u]<0)return false;
   double z0=candidate[u]-w[u]/eig,z1=candidate[v]-w[v]/eig,length=std::hypot(z0,z1),cap=hi[u]*candidate[k],factor=length>cap&&length>0?cap/length:1;
   error=std::max(error,std::hypot(candidate[u]-factor*z0,candidate[v]-factor*z1)*eig);
  }
  stats.residual=error;return std::isfinite(error)&&std::isfinite(energy)&&std::isfinite(scale)&&error<=tol&&energy<=tol*scale;
 };
 for(int pass=0;pass<8&&budget.steps>0;pass++){
  std::vector<int>rows,map(n,-1);for(int k:normals)if(active[k]){rows.push_back(k);for(int j=0;j<n;j++)if(dep[j]==k)rows.push_back(j);}
  if(rows.empty()){if(fullGate()){for(int i=0;i<n;i++)x[i]=candidate[i];return true;}for(int k:normals)if(w[k]<-tol)active[k]=true;continue;}
  const int m=rows.size();for(int i=0;i<m;i++)map[rows[i]]=i;
  btMatrixXu reduced(m,m);btVectorXu rhs(m),p(m),upper(m);btAlignedObjectArray<int>dependencies;dependencies.resize(m);
  for(int i=0;i<m;i++){int r=rows[i];rhs[i]=b[r];upper[i]=hi[r];p[i]=pass?candidate[r]:x[r];dependencies[i]=dep[r]<0?-1:map[dep[r]];for(int j=0;j<m;j++)reduced.setElem(i,j,A(r,rows[j]));}
  stats.passes++;stats.active_contacts=m/3;
  active_trust::solve(reduced,rhs,p,upper,dependencies,tol,stats.search,budget);
  std::fill(candidate.begin(),candidate.end(),0);for(int i=0;i<m;i++)candidate[rows[i]]=p[i];
  if(fullGate()){for(int i=0;i<n;i++)x[i]=candidate[i];return true;}
  bool expanded=false;for(int k:normals)if(!active[k]&&w[k]<-tol){active[k]=true;stats.expanded_contacts++;expanded=true;}
  if(!expanded)return false;
 }
 return false;
}
}
