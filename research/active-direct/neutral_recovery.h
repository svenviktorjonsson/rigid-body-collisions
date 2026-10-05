// Search-only mobility-neutral opposing-traction guesses. Original law is the gate.
#pragma once
#include "homotopy_candidate.h"
#include "more_direct.h"
namespace neutral_recovery {
struct Stats {
 int svd_calls=0,newton_steps=0,damped_steps=0,projector_calls=0,restarts=0,neutral_rank=0,neutral_dimension=0;
 double residual=0,maximum_neutral_velocity=0;std::vector<int> restart_contacts;std::vector<double> restart_fractions;
};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&x,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats,int svd_limit=1024){
 const int n=b.rows();if(n<=0||n>64)return false;
 auto candidate=x;homotopy_candidate::Stats hs;
 bool ok=homotopy_candidate::solve(A,b,candidate,hi,dep,tol,hs);
 stats.svd_calls+=hs.svd_calls;stats.newton_steps+=hs.newton_steps;stats.damped_steps+=hs.damped_steps;stats.residual=hs.residual;
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
 const int m=rows.size();if(m==0||m>n||modes.empty())return false;
 std::vector<double>C(n*m),s(m),U(n*m),VT(m*m);for(int j=0;j<m;j++)for(int i=0;i<n;i++)C[i+j*n]=A(i,rows[j]);
 char job='S';int lwork=-1,info=0;double query=0;std::vector<int>iwork(8*m);
 dgesdd_(&job,&n,&m,C.data(),&n,s.data(),U.data(),&n,VT.data(),&m,&query,&lwork,iwork.data(),&info);if(info||!std::isfinite(query))return false;
 lwork=static_cast<int>(query);std::vector<double>workspace(lwork);dgesdd_(&job,&n,&m,C.data(),&n,s.data(),U.data(),&n,VT.data(),&m,workspace.data(),&lwork,iwork.data(),&info);stats.svd_calls++;stats.projector_calls++;if(info)return false;
 int rank=0;while(rank<m&&s[rank]>s[0]*1e-13)rank++;stats.neutral_rank=rank;stats.neutral_dimension=m-rank;
 for(int contact=0;contact<std::min(2,static_cast<int>(modes.size()));contact++){
  const auto c=modes[contact];const double speed=std::hypot(w[c.t],w[c.s]);auto target=p;const double cap=c.mu*std::max(0.,p[c.k]);target[c.t]=-cap*w[c.t]/speed;target[c.s]=-cap*w[c.s]/speed;
  std::vector<double>delta(n,0);for(int null=rank;null<m;null++){double dot=0;for(int j=0;j<m;j++)dot+=VT[null+j*m]*(target[rows[j]]-p[rows[j]]);for(int j=0;j<m;j++)delta[rows[j]]+=VT[null+j*m]*dot;}
  for(double fraction:{.25,.5,1.,2.}){
   const int remaining=svd_limit-stats.svd_calls;if(remaining<=0)return false;
   auto trial=x;for(int i=0;i<n;i++)trial[i]=p[i]+fraction*delta[i];double change=0;for(int i=0;i<n;i++){double d=0;for(int j=0;j<n;j++)d+=A(i,j)*(trial[j]-p[j]);change=std::max(change,std::abs(d));}stats.maximum_neutral_velocity=std::max(stats.maximum_neutral_velocity,change);
   const double trialcap=c.mu*std::max(0.,static_cast<double>(trial[c.k]));trial[c.t]=-trialcap*w[c.t]/speed;trial[c.s]=-trialcap*w[c.s]/speed;
   stats.restarts++;stats.restart_contacts.push_back(c.k);stats.restart_fractions.push_back(fraction);more_direct::Stats ms;
   ok=more_direct::solve(A,b,trial,hi,dep,tol,ms,std::min(300,remaining),std::min(300,remaining));stats.svd_calls+=ms.svd_calls;stats.newton_steps+=ms.newton_steps;stats.damped_steps+=ms.damped_steps;stats.residual=ms.residual;
   if(ok){x=trial;return true;}
  }
 }
 return false;
}
}
