// Research-only original-law projection-merit trust search.
// Warm alpha=1 only: no physical friction continuation or contact change.
// Numerical column scaling and damped trust subproblems never alter A,b,mu.
// Only original all-row projection/bounds/finite-passivity gates accept x.
// Stats must be fresh per invocation; aggregate externally after return.
#pragma once
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "restart_validate.h"
#include "spectral_more.h"
#include <limits>
#include <cmath>
namespace projection_recovery_v2 {
struct Budget {int steps=2048,svds=2048;};
struct Stats {int iteration_steps=0,newton_steps=0,svd_calls=0,budget_rejections=0;std::vector<double>failed_candidate;double residual=0;};
struct Contact {int k,t,s;double mu,rn,rt;};
inline bool solve(const btMatrixXu& A,const btVectorXu& b,btVectorXu& x,const btVectorXu& hi,
 const btAlignedObjectArray<int>& dep,double tolerance,Stats& stats,int step_limit=2048,int svd_limit=2048,bool scale_columns=true,bool projection_merit=true,bool right_cone_derivative=false){
 Budget budget;budget.steps=std::min(step_limit,2048);budget.svds=std::min(svd_limit,2048);
 const int n=b.rows();if(stats.svd_calls||stats.iteration_steps||stats.newton_steps||stats.budget_rejections||!stats.failed_candidate.empty())return false;if(!restart_validation::valid(A,b,x,hi,dep,tolerance,64)||svd_limit<=0||step_limit<=0)return false;
 std::vector<Contact> contacts;
 for(int k=0;k<n;k++)if(dep[k]<0){
  std::vector<int> t;for(int j=0;j<n;j++)if(dep[j]==k)t.push_back(j);
  if(t.size()!=2||!(A(k,k)>0)||hi[t[0]]!=hi[t[1]]||hi[t[0]]<0)return false;
  const double a=A(t[0],t[0]),d=A(t[1],t[1]),off=A(t[0],t[1]);
  const double eigen=.5*(a+d+std::hypot(a-d,2*off));if(!(eigen>0))return false;
  contacts.push_back({k,t[0],t[1],static_cast<double>(hi[t[0]]),1/A(k,k),1/eigen});
 }
 if(contacts.size()*3!=static_cast<size_t>(n))return false;
 int& remaining_steps=budget.steps;int& remaining_svd=budget.svds;
 auto equations=[&](const std::vector<double>& p,double alpha,std::vector<double>& F,std::vector<double>* J,bool projection=false){
  std::vector<double>w(n);for(int i=0;i<n;i++){w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*p[j];}
  F.assign(n,0);if(J)J->assign(n*n,0);double maximum=0;
  for(auto c:contacts){
   if(projection){
    const double zn=p[c.k]-c.rn*w[c.k];F[c.k]=(p[c.k]-std::max(0.,zn))/c.rn;
    if(J)for(int j=0;j<n;j++)(*J)[c.k*n+j]=zn>0?A(c.k,j):(j==c.k?1/c.rn:0);
   }else{
    // FB is only a numerical merit representation of pn>=0,w>=0,pn*w=0.
    // Final acceptance explicitly recomputes the original projection map.
    const double u=p[c.k]/c.rn,v=w[c.k],length=std::hypot(u,v);
    F[c.k]=u+v-length;
    const double du=length>0?1-u/length:1,dv=length>0?1-v/length:1;
    if(J)for(int j=0;j<n;j++)(*J)[c.k*n+j]=dv*A(c.k,j)+(j==c.k?du/c.rn:0);
   }
   const int rows[2]={c.t,c.s};const double mu=alpha*c.mu,cap=mu*std::max(0.,p[c.k]);
   const double z[2]={p[c.t]-c.rt*w[c.t],p[c.s]-c.rt*w[c.s]},length=std::hypot(z[0],z[1]);
   if(length<=cap&&cap>0){for(int r:rows){F[r]=w[r];if(J)for(int j=0;j<n;j++)(*J)[r*n+j]=A(r,j);}}
   else{
    const double direction[2]={length>0?z[0]/length:0,length>0?z[1]/length:0};
    for(int u=0;u<2;u++){const int r=rows[u];F[r]=(p[r]-cap*direction[u])/c.rt;
     if(J)for(int j=0;j<n;j++){
      double value=j==r?1.:0.;for(int h=0;h<2;h++){
       const double D=length>0?cap/length*((u==h?1.:0.)-direction[u]*direction[h]):0;
       value-=D*((j==rows[h]?1.:0.)-c.rt*A(rows[h],j));
      }
      if(j==c.k&&(p[c.k]>0||(right_cone_derivative&&p[c.k]==0)))value-=mu*direction[u];
      (*J)[r*n+j]=value/c.rt;
     }
    }
   }
   maximum=std::max({maximum,std::abs(F[c.k]),std::hypot(F[c.t],F[c.s])});
  }
  for(double f:F)if(!std::isfinite(f))return std::numeric_limits<double>::infinity();
  return maximum;
 };
 auto merit=[](const std::vector<double>& F){double m=0;for(double v:F)m+=v*v;return m;};
 auto finalGate=[&](std::vector<double>& p){
  auto trial=p;for(auto c:contacts){if(trial[c.k]<0){if(trial[c.k]<-tolerance*c.rn)return false;trial[c.k]=0;}if(trial[c.k]>hi[c.k])return false;}
  std::vector<double>F;const double error=equations(trial,1,F,nullptr,true);if(!std::isfinite(error)||error>tolerance)return false;
  double energy=0,scale=1;for(int i=0;i<n;i++){double w=-b[i];for(int j=0;j<n;j++)w+=A(i,j)*trial[j];energy+=.5*trial[i]*(w-b[i]);scale+=std::abs(trial[i]*b[i]);}
  if(!std::isfinite(energy)||!std::isfinite(scale)||energy>tolerance*scale)return false;
  p=std::move(trial);return true;
 };
 auto stage=[&](std::vector<double>&p,double alpha,int limit=64){
  double alphaParameter=-1;double radius=0;std::vector<double> diagonal(n,0);bool initialized=false;
  for(int iteration=0;iteration<limit&&remaining_steps>0&&remaining_svd>0;iteration++){
   if(alpha==1&&finalGate(p))return true;
   remaining_steps--;stats.iteration_steps++;std::vector<double>F,J;equations(p,alpha,F,&J,projection_merit);double oldMerit=merit(F);if(!std::isfinite(oldMerit))return false;
   // Scaling changes only numerical impulse coordinates, never A/contact law.
   for(int j=0;j<n;j++){double column=0;for(int i=0;i<n;i++)column=std::hypot(column,J[i*n+j]);if(!std::isfinite(column))return false;diagonal[j]=scale_columns?std::max(diagonal[j],column):1.;if(diagonal[j]==0)diagonal[j]=1.;}
   if(!initialized){for(int j=0;j<n;j++)radius=std::hypot(radius,diagonal[j]*p[j]);if(radius==0)radius=1;if(!std::isfinite(radius))return false;initialized=true;}
   auto scaled=J;for(int i=0;i<n;i++)for(int j=0;j<n;j++)scaled[i*n+j]/=diagonal[j];
   auto rhs=F;for(double&v:rhs)v=-v;remaining_svd--;stats.svd_calls++;auto linear=restart_spectral::spectralMoreStep(scaled,rhs,n,radius,alphaParameter);if(!linear.converged)return false;
   auto step=linear.step;for(int j=0;j<n;j++)step[j]/=diagonal[j];
   auto trial=p;for(int i=0;i<n;i++)trial[i]+=step[i];std::vector<double>next;equations(trial,alpha,next,nullptr,projection_merit);double newMerit=merit(next);double predicted=0;for(int i=0;i<n;i++){double value=F[i];for(int j=0;j<n;j++)value+=J[i*n+j]*step[j];predicted+=value*value;}predicted=oldMerit-predicted;double ratio=predicted>0?(oldMerit-newMerit)/predicted:0;
   double oldRadius=radius;alphaParameter=linear.lambda;
   if(ratio<.25)radius=std::max(1e-16,.25*linear.norm);else if(ratio>.75&&linear.norm>=.95*radius)radius*=2;
   if(!(radius>0&&std::isfinite(radius)))return false;alphaParameter*=oldRadius/radius;if(!std::isfinite(alphaParameter))return false;
   if(std::isfinite(newMerit)&&newMerit<oldMerit){p=std::move(trial);stats.newton_steps++;}
  }
  return alpha==1&&finalGate(p);
 };
 std::vector<double>warm(n);for(int i=0;i<n;i++)warm[i]=x[i];
 auto accept=[&](std::vector<double>&candidate){if(!finalGate(candidate))return false;std::vector<double>F;stats.residual=equations(candidate,1,F,nullptr,true);for(int i=0;i<n;i++)x[i]=candidate[i];return true;};
 stage(warm,1,step_limit);if(accept(warm))return true;
 if(remaining_steps<=0||remaining_svd<=0)stats.budget_rejections++;
 stats.failed_candidate=warm;std::vector<double>F;stats.residual=equations(warm,1,F,nullptr,true);return false;
}
}
