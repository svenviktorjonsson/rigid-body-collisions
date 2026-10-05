// Numerical reduced contact search; ONLY the full original contact gate accepts.
#pragma once
#include "active_trust_v3.h"
namespace active_face_v4 {
struct Stats {int passes=0,active_contacts=0,expanded_contacts=0,mode_guesses=0;double residual=std::numeric_limits<double>::infinity();active_trust_v3::Stats search;};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&x,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats){
 const int n=b.rows();if(n<=0||n>384)return false;
 std::vector<int>normals;std::vector<bool>active(n,false);
 for(int k=0;k<n;k++)if(dep[k]<0){normals.push_back(k);active[k]=x[k]>0;}
 std::vector<double>seed(n);for(int i=0;i<n;i++)seed[i]=x[i];
 // When normal complementarity already passes but a nominally sticking
 // disk does not, choose a numerical sliding-boundary starting point. This
 // guess never changes the physical A,b,mu or the final full-system gate.
 double normalError=0;int boundary=-1;double highestRatio=-1;
 for(int k:normals){double wn=-b[k];for(int j=0;j<n;j++)wn+=A(k,j)*seed[j];normalError=std::max(normalError,std::abs(seed[k]-std::max(0.,seed[k]-wn/A(k,k)))*A(k,k));
  if(seed[k]<=0)continue;std::vector<int>ts;for(int j=0;j<n;j++)if(dep[j]==k)ts.push_back(j);if(ts.size()!=2)return false;
  int u=ts[0],v=ts[1];double wu=-b[u],wv=-b[v];for(int j=0;j<n;j++){wu+=A(u,j)*seed[j];wv+=A(v,j)*seed[j];}
  const double cap=hi[u]*seed[k],length=std::hypot(seed[u],seed[v]),ratio=cap>0?length/cap:1;
  if(ratio<1-1e-8&&std::hypot(wu,wv)>tol&&ratio>highestRatio){boundary=k;highestRatio=ratio;}
 }
 if(normalError<=tol&&boundary>=0){std::vector<int>ts;for(int j=0;j<n;j++)if(dep[j]==boundary)ts.push_back(j);double wu=-b[ts[0]],wv=-b[ts[1]];for(int j=0;j<n;j++){wu+=A(ts[0],j)*seed[j];wv+=A(ts[1],j)*seed[j];}double length=std::hypot(wu,wv),cap=hi[ts[0]]*seed[boundary];seed[ts[0]]=-cap*wu/length;seed[ts[1]]=-cap*wv/length;stats.mode_guesses++;}
 auto base=active;std::vector<int>release;for(int k:normals)if(active[k])release.push_back(k);
 std::stable_sort(release.begin(),release.end(),[&](int u,int v){double wu=-b[u],wv=-b[v];for(int j=0;j<n;j++){wu+=A(u,j)*x[j];wv+=A(v,j)*x[j];}return std::max(0.,wu)*x[u]>std::max(0.,wv)*x[v];});
 int released=0;active_trust_v3::Budget budget;
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
  for(int i=0;i<m;i++){int r=rows[i];rhs[i]=b[r];upper[i]=hi[r];p[i]=seed[r];dependencies[i]=dep[r]<0?-1:map[dep[r]];for(int j=0;j<m;j++)reduced.setElem(i,j,A(r,rows[j]));}
  stats.passes++;stats.active_contacts=m/3;
  active_trust_v3::solve(reduced,rhs,p,upper,dependencies,tol,stats.search,budget);
  std::fill(candidate.begin(),candidate.end(),0);for(int i=0;i<m;i++)candidate[rows[i]]=p[i];
  if(fullGate()){for(int i=0;i<n;i++)x[i]=candidate[i];return true;}
  bool expanded=false;for(int k:normals)if(!active[k]&&w[k]<-tol){active[k]=true;stats.expanded_contacts++;expanded=true;}
  if(!expanded){if(released>=static_cast<int>(release.size()))return false;active=base;active[release[released++]]=false;}
 }
 return false;
}
}
