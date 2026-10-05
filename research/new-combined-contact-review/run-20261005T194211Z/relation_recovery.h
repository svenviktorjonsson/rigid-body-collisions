// Research-only numerical guide. Physical equations and final gate stay original.
#pragma once
#include "restart_validate.h"
#include <numeric>
namespace relation_recovery {
struct Stats {
 int attempts=0,solves=0,declines=0,components=0,component_cap_rejections=0;
 int guides=0,iteration_sweeps=0,newton_steps=0,largest_rows=0;
 double residual=std::numeric_limits<double>::infinity(),passivity=0.;
};
inline bool solve(const btMatrixXu& A,const btVectorXu& b,btVectorXu& x,
 const btVectorXu& hi,const btAlignedObjectArray<int>& dep,double tolerance,Stats& stats){
 stats.attempts++;
 const int n=b.rows();
 if(!restart_validation::valid(A,b,x,hi,dep,tolerance,4096)){stats.declines++;return false;}
 for(int i=0;i<n;i++)for(int j=0;j<i;j++)if(A(i,j)!=A(j,i)){stats.declines++;return false;}
 struct Contact {int k,t,s;double eig,mu;};std::vector<Contact> contacts;
 for(int k=0;k<n;k++)if(dep[k]<0){
  std::vector<int> ts;for(int j=0;j<n;j++)if(dep[j]==k)ts.push_back(j);
  int t=ts[0],s=ts[1];double eig=.5*(A(t,t)+A(s,s)+std::hypot(A(t,t)-A(s,s),2*A(t,s)));
  if(!(eig>0&&std::isfinite(eig)&&A(t,t)>0&&A(s,s)>0&&A(t,t)*A(s,s)-A(t,s)*A(t,s)>0)){stats.declines++;return false;}
  contacts.push_back({k,t,s,eig,static_cast<double>(hi[t])});
 }
 auto trial=x;
 auto velocities=[&](const btVectorXu& p){std::vector<double>w(n);for(int i=0;i<n;i++){w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*p[j];}return w;};
 auto contactError=[&](const Contact& c,const btVectorXu& p,const std::vector<double>& w){
  if(!std::isfinite(p[c.k])||!std::isfinite(p[c.t])||!std::isfinite(p[c.s])||!std::isfinite(w[c.k])||!std::isfinite(w[c.t])||!std::isfinite(w[c.s])||p[c.k]<0||p[c.k]>hi[c.k])return std::numeric_limits<double>::infinity();
  double normal=std::abs(p[c.k]-std::max(0.,static_cast<double>(p[c.k])-w[c.k]/A(c.k,c.k)))*A(c.k,c.k);
  double z0=p[c.t]-w[c.t]/c.eig,z1=p[c.s]-w[c.s]/c.eig,length=std::hypot(z0,z1),cap=c.mu*p[c.k],factor=length>cap&&length>0?cap/length:1.;
  return std::max(normal,std::hypot(p[c.t]-factor*z0,p[c.s]-factor*z1)*c.eig);
 };
 auto gate=[&](){
  auto w=velocities(trial);double residual=0,energy=0,scale=1;
  for(int i=0;i<n;i++){if(!std::isfinite(trial[i])||!std::isfinite(w[i]))return false;energy+=.5*trial[i]*(w[i]-b[i]);scale+=std::abs(trial[i]*b[i]);}
  for(auto c:contacts)residual=std::max(residual,contactError(c,trial,w));
  stats.residual=residual;stats.passivity=energy;
  return std::isfinite(residual)&&std::isfinite(energy)&&std::isfinite(scale)&&residual<=tolerance&&energy<=tolerance*scale;
 };
 if(gate()){x=trial;stats.solves++;return true;}
 // Exact graph plus constitutive dependency edges: never discard small couplings.
 std::vector<int> parent(n);std::iota(parent.begin(),parent.end(),0);
 auto find=[&](int i){while(parent[i]!=i){parent[i]=parent[parent[i]];i=parent[i];}return i;};
 auto join=[&](int i,int j){i=find(i);j=find(j);if(i!=j)parent[j]=i;};
 for(int i=0;i<n;i++){if(dep[i]>=0)join(i,dep[i]);for(int j=0;j<i;j++)if(A(i,j)!=0)join(i,j);}
 std::vector<std::vector<int>> components(n);for(int i=0;i<n;i++)components[find(i)].push_back(i);
 // All caps below are fresh local counters; Stats aggregates only receipts.
 int components_used=0,guides_used=0,sweeps_used=0;
 for(const auto& rows:components){
  if(rows.empty())continue;auto w=velocities(trial);bool violated=false;
  for(auto c:contacts)if(find(c.k)==find(rows[0])&&contactError(c,trial,w)>tolerance)violated=true;
  if(!violated)continue;
  if(rows.size()>64){stats.component_cap_rejections++;stats.declines++;return false;}
  if(components_used>=4){stats.declines++;return false;}components_used++;stats.components++;stats.largest_rows=std::max(stats.largest_rows,static_cast<int>(rows.size()));
  const int m=static_cast<int>(rows.size());std::vector<int> map(n,-1);for(int i=0;i<m;i++)map[rows[i]]=i;
  btMatrixXu reduced(m,m);btVectorXu rhs(m),upper(m),lower(m),seed(m);btAlignedObjectArray<int> dependencies;dependencies.resize(m);
  for(int i=0;i<m;i++){int r=rows[i];rhs[i]=b[r];upper[i]=hi[r];lower[i]=dep[r]<0?0:-hi[r];seed[i]=trial[r];dependencies[i]=dep[r]<0?-1:map[dep[r]];for(int j=0;j<m;j++)reduced.setElem(i,j,A(r,rows[j]));}
  bool repaired=false;
  // Roundoff-related rows suggest a sliding face only; full exact equations
  // still decide acceptance. Detection never changes A,b or the tolerance.
  for(auto active:contacts){if(repaired)break;if(map[active.k]<0||!(trial[active.k]>0))continue;
   for(auto sliding:contacts){if(repaired)break;if(map[sliding.k]<0||!(trial[sliding.k]>0))continue;
    for(int t:{sliding.t,sliding.s}){if(repaired)break;int other=t==sliding.t?sliding.s:sliding.t;
     if(w[other]==0||!std::isfinite(w[other]))continue;
     for(int sign:{1,-1}){if(repaired)break;double scale=std::max({1.,std::abs(static_cast<double>(b[active.k])),std::abs(static_cast<double>(b[t]))}),difference=std::abs(b[active.k]-sign*b[t]);
      for(int j=0;j<n;j++){scale=std::max({scale,std::abs(static_cast<double>(A(active.k,j))),std::abs(static_cast<double>(A(t,j)))});difference=std::max(difference,std::abs(A(active.k,j)-sign*A(t,j)));}
      if(difference>64*std::numeric_limits<double>::epsilon()*scale)continue;
      if(guides_used>=8||sweeps_used>=16384){stats.declines++;return false;}
      auto candidate=seed;candidate[map[t]]=0;candidate[map[other]]=-std::copysign(sliding.mu*std::max(0.,static_cast<double>(trial[sliding.k])),w[other]);
      guides_used++;stats.guides++;CoulombStats local;std::vector<double> rejected;
      const int cap=std::min(4096,16384-sweeps_used);
      const bool accepted=coulombIterate(reduced,rhs,candidate,lower,upper,dependencies,cap,tolerance,local,&rejected);
      sweeps_used+=static_cast<int>(local.iteration_sweeps_total);stats.iteration_sweeps+=static_cast<int>(local.iteration_sweeps_total);stats.newton_steps+=local.newton_steps;
      if(accepted){for(int i=0;i<m;i++)trial[rows[i]]=candidate[i];repaired=true;}
     }
    }
   }
  }
  if(!repaired){stats.declines++;return false;}
 }
 if(gate()){x=trial;stats.solves++;return true;}
 stats.declines++;return false;
}
}
