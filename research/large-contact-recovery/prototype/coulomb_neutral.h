// Search-only mobility-neutral opposing-traction guesses. Original law is the gate.
#pragma once
#include "coulomb_homotopy_guide.h"
#include "coulomb_more.h"
namespace restart_neutral {
struct Stats {
 int homotopy_attempts=0,homotopy_stages=0,iteration_steps=0,svd_calls=0,newton_steps=0,pressure_svd_calls=0,pressure_attempts=0,normal_pivot_attempts=0,normal_pivot_guides=0,normal_qp_guides=0,damped_steps=0,projector_calls=0,restarts=0,neutral_rank=0,neutral_dimension=0;
 double residual=0,maximum_neutral_velocity=0;std::vector<int> restart_contacts;std::vector<double> restart_fractions;
};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&x,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats,int svd_limit=1024){
 const int n=b.rows();
 if(n<=0||n>64||svd_limit<=0||A.rows()!=n||A.cols()!=n||x.rows()!=n||hi.rows()!=n||dep.size()!=n||!(tol>0)||!std::isfinite(tol))return false;
 for(int i=0;i<n;i++){
  if(!std::isfinite(b[i])||!std::isfinite(x[i])||!std::isfinite(hi[i])||hi[i]<0||dep[i]<-1||dep[i]>=n)return false;
  if(dep[i]>=0&&dep[dep[i]]!=-1)return false;
  for(int j=0;j<n;j++)if(!std::isfinite(A(i,j)))return false;
 }
 for(int k=0;k<n;k++)if(dep[k]==-1){int count=0;double mu=-1;for(int j=0;j<n;j++)if(dep[j]==k){count++;if(mu<0)mu=hi[j];else if(mu!=hi[j])return false;}if(count!=2||!(A(k,k)>0))return false;}
 auto candidate=x;restart_guide::Stats hs;
 bool ok=restart_guide::solve(A,b,candidate,hi,dep,tol,hs,svd_limit);
 stats.iteration_steps+=hs.iteration_steps;stats.svd_calls+=hs.svd_calls;stats.newton_steps+=hs.newton_steps;stats.damped_steps+=hs.damped_steps;stats.residual=hs.residual;stats.homotopy_attempts=hs.attempts;stats.homotopy_stages=hs.stages;stats.pressure_svd_calls=hs.pressure_svd_calls;stats.pressure_attempts=hs.pressure_attempts;stats.normal_pivot_attempts=hs.normal_pivot_attempts;stats.normal_pivot_guides=hs.normal_pivot_guides;stats.normal_qp_guides=hs.normal_qp_guides;
 if(ok){x=candidate;return true;}if(hs.failed_candidate.size()!=static_cast<size_t>(n))return false;
 const auto p=hs.failed_candidate;std::vector<double>w(n,0);for(int i=0;i<n;i++){w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*p[j];}
 struct Contact{int k,t,s;double mu,ratio;};std::vector<Contact>active,modes;std::vector<int>rows;
 for(int k=0;k<n;k++)if(dep[k]<0){std::vector<int>t;for(int i=0;i<n;i++)if(dep[i]==k)t.push_back(i);if(t.size()!=2)return false;
  if(p[k]>1e-9||std::abs(w[k])<10*tol){Contact c{k,t[0],t[1],static_cast<double>(hi[t[0]]),0};active.push_back(c);rows.push_back(k);rows.push_back(t[0]);rows.push_back(t[1]);
   const double cap=c.mu*std::max(0.,p[k]);const double pt=std::hypot(p[c.t],p[c.s]);const double speed=std::hypot(w[c.t],w[c.s]);
   if(cap>0&&pt<cap&&speed>tol){c.ratio=pt/cap;modes.push_back(c);}
  }
 }
 std::stable_sort(modes.begin(),modes.end(),[](auto a,auto c){return a.ratio>c.ratio;});
 if(stats.svd_calls>=svd_limit)return false;
 const int m=rows.size();if(m==0||m>n||modes.empty())return false;
 std::vector<double>C(n*n,0),zero(n,0);for(int j:rows)for(int i=0;i<n;i++)C[i*n+j]=A(i,j);
 auto decomposition=minimumNormNewton(C,zero,n,1e-13);stats.svd_calls++;stats.projector_calls++;if(!decomposition.converged)return false;
 stats.neutral_rank=decomposition.rank;stats.neutral_dimension=m-decomposition.rank;
 for(int contact=0;contact<std::min(2,static_cast<int>(modes.size()));contact++){
  const auto c=modes[contact];const double speed=std::hypot(w[c.t],w[c.s]);auto target=p;const double cap=c.mu*std::max(0.,p[c.k]);target[c.t]=-cap*w[c.t]/speed;target[c.s]=-cap*w[c.s]/speed;
  std::vector<double>delta(n,0);for(const auto& direction:decomposition.nullspace){double dot=0;for(int j:rows)dot+=direction[j]*(target[j]-p[j]);for(int j:rows)delta[j]+=direction[j]*dot;}
  for(double fraction:{.25,.5,1.,2.}){
   const int remaining=svd_limit-stats.svd_calls;if(remaining<=0)return false;
   auto trial=x;for(int i=0;i<n;i++)trial[i]=p[i]+fraction*delta[i];double change=0;for(int i=0;i<n;i++){double d=0;for(int j=0;j<n;j++)d+=A(i,j)*(trial[j]-p[j]);change=std::max(change,std::abs(d));}stats.maximum_neutral_velocity=std::max(stats.maximum_neutral_velocity,change);
   const double trialcap=c.mu*std::max(0.,static_cast<double>(trial[c.k]));trial[c.t]=-trialcap*w[c.t]/speed;trial[c.s]=-trialcap*w[c.s]/speed;
   stats.restarts++;stats.restart_contacts.push_back(c.k);stats.restart_fractions.push_back(fraction);restart_more::Stats ms;
   ok=restart_more::solve(A,b,trial,hi,dep,tol,ms,std::min(300,remaining),std::min(300,remaining));stats.iteration_steps+=ms.iteration_steps;stats.svd_calls+=ms.svd_calls;stats.newton_steps+=ms.newton_steps;stats.damped_steps+=ms.damped_steps;stats.residual=ms.residual;
   if(ok){x=trial;return true;}
  }
 }
 return false;
}
}
