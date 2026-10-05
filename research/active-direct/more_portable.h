// Bounded numerical continuation/trust search for the ORIGINAL circular law.
// Trial friction parameters and damped Newton subproblems are never physical
// output: only the unmodified A,b,mu complementarity/passivity gate accepts x.
#pragma once
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "newton_linear.h"
#include "normal_qp.h"
#include "pressure_release.h"
#include "spectral_more_jacobi.h"
#include <limits>
#include <cmath>
namespace more_portable {
struct Budget {int steps=512,svds=256,damping=512,attempts=96,pressure_attempts=128;};
struct Stats {int attempts=0,stages=0,newton_steps=0,svd_calls=0,damped_steps=0,budget_rejections=0,pressure_guides=0,pressure_attempts=0,pressure_svd_calls=0,normal_qp_guides=0,normal_pivot_attempts=0,normal_pivot_guides=0;std::vector<double>failed_candidate;double residual=0,guide_residual=0;};
struct Contact {int k,t,s;double mu,rn,rt;};
inline bool solve(const btMatrixXu& A,const btVectorXu& b,btVectorXu& x,const btVectorXu& hi,
 const btAlignedObjectArray<int>& dep,double tolerance,Stats& stats,int step_limit=256,int svd_limit=256){
 Budget budget;
 const int n=b.rows();if(n<=0||n>384)return false;
 budget.steps=step_limit;budget.svds=svd_limit;
 std::vector<Contact> contacts;
 for(int k=0;k<n;k++)if(dep[k]<0){
  std::vector<int> t;for(int j=0;j<n;j++)if(dep[j]==k)t.push_back(j);
  if(t.size()!=2||!(A(k,k)>0)||hi[t[0]]!=hi[t[1]]||hi[t[0]]<0)return false;
  const double a=A(t[0],t[0]),d=A(t[1],t[1]),off=A(t[0],t[1]);
  const double eigen=.5*(a+d+std::hypot(a-d,2*off));if(!(eigen>0))return false;
  contacts.push_back({k,t[0],t[1],static_cast<double>(hi[t[0]]),1/A(k,k),1/eigen});
 }
 if(contacts.size()*3!=static_cast<size_t>(n))return false;
 int& remaining_steps=budget.steps;int& remaining_svd=budget.svds;int& remaining_damped=budget.damping;
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
      if(j==c.k&&p[c.k]>0)value-=mu*direction[u];
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
  double alphaParameter=-1;double radius=0;for(double v:p)radius+=v*v;radius=std::sqrt(radius);if(radius==0)radius=1;
  for(int iteration=0;iteration<limit&&remaining_steps>0&&remaining_svd>0;iteration++){
   if(alpha==1&&finalGate(p))return true;remaining_steps--;std::vector<double>F,J;equations(p,alpha,F,&J);double oldMerit=merit(F);if(!std::isfinite(oldMerit))return false;
   auto rhs=F;for(double&v:rhs)v=-v;remaining_svd--;stats.svd_calls++;auto linear=spectralMoreStep(J,rhs,n,radius,alphaParameter);if(!linear.converged)return false;
   auto trial=p;for(int i=0;i<n;i++)trial[i]+=linear.step[i];std::vector<double>next;equations(trial,alpha,next,nullptr);double newMerit=merit(next);double predicted=0;for(int i=0;i<n;i++){double value=F[i];for(int j=0;j<n;j++)value+=J[i*n+j]*linear.step[j];predicted+=value*value;}predicted=oldMerit-predicted;double ratio=predicted>0?(oldMerit-newMerit)/predicted:0;
   double oldRadius=radius;alphaParameter=linear.lambda;
   if(ratio<.25)radius=std::max(1e-16,.25*linear.norm);else if(ratio>.75&&linear.norm>=.95*radius)radius*=2;
   alphaParameter*=oldRadius/radius;
   if(std::isfinite(newMerit)&&newMerit<oldMerit){p=std::move(trial);stats.newton_steps++;}
  }
  return alpha==1&&finalGate(p);
 };
 std::vector<double>warm(n);for(int i=0;i<n;i++)warm[i]=x[i];
 auto accept=[&](std::vector<double>&candidate){if(!finalGate(candidate))return false;std::vector<double>F;stats.residual=equations(candidate,1,F,nullptr,true);for(int i=0;i<n;i++)x[i]=candidate[i];return true;};
 stage(warm,1,step_limit);if(accept(warm))return true;
 stats.failed_candidate=warm;std::vector<double>F;stats.residual=equations(warm,1,F,nullptr,true);return false;
}
}
