// Bounded numerical continuation/trust search for the ORIGINAL circular law.
// Trial friction parameters and damped Newton subproblems are never physical
// output: only the unmodified A,b,mu complementarity/passivity gate accepts x.
#pragma once
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "newton_linear.h"
#include "normal_qp.h"
#include "pressure_release.h"
#include <limits>
#include <cmath>
namespace direct_fb {
struct Budget {int steps=512,svds=256,damping=512,attempts=96,pressure_attempts=128;};
struct Stats {int attempts=0,stages=0,newton_steps=0,svd_calls=0,damped_steps=0,budget_rejections=0,pressure_guides=0,pressure_attempts=0,pressure_svd_calls=0,normal_qp_guides=0,normal_pivot_attempts=0,normal_pivot_guides=0;double residual=0,guide_residual=0;};
struct Contact {int k,t,s;double mu,rn,rt;};
inline bool solve(const btMatrixXu& A,const btVectorXu& b,btVectorXu& x,const btVectorXu& hi,
 const btAlignedObjectArray<int>& dep,double tolerance,Stats& stats){
 Budget budget;
 const int n=b.rows();if(n<=0||n>384)return false;
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
 auto stage=[&](std::vector<double>& p,double alpha,int limit=48){
  double damping=1e-4;
  for(int iteration=0;iteration<limit&&remaining_steps>0;iteration++){
   remaining_steps--;if(alpha==1&&finalGate(p))return true;std::vector<double>F,J;double error=equations(p,alpha,F,&J);
   if(error<=std::min(1e-10,tolerance*.1))return true;
   if(!std::isfinite(error))return false;
   const double oldMerit=merit(F);bool accepted=false;
   auto trialStep=[&](const std::vector<double>& step){
    for(int line=0;line<24;line++){
     const double fraction=std::ldexp(1.,-line);std::vector<double>trial=p;
     for(int i=0;i<n;i++)trial[i]+=fraction*step[i];
     std::vector<double>next;equations(trial,alpha,next,nullptr);
     const double newMerit=merit(next);
     if(std::isfinite(newMerit)&&newMerit<=(1-1e-4*fraction)*oldMerit){p=std::move(trial);return true;}
    }
    return false;
   };
   if(remaining_svd>0){
    remaining_svd--;stats.svd_calls++;auto rhs=F;for(double& v:rhs)v=-v;
    auto linear=minimumNormNewton(J,rhs,n,1e-12);
    if(linear.converged&&trialStep(linear.step)){accepted=true;stats.newton_steps++;}
   }
   if(accepted)continue;
   // Levenberg damping affects only the numerical search Hessian J^T J.
   // The physical mobility A and every evaluated contact equation stay exact.
   std::vector<double>H(n*n,0),gradient(n,0);double scale=0;
   for(int i=0;i<n;i++)for(int j=0;j<n;j++){
    gradient[j]-=J[i*n+j]*F[i];
    for(int k=0;k<=j;k++)H[j*n+k]+=J[i*n+j]*J[i*n+k];
   }
   for(int j=0;j<n;j++){scale=std::max(scale,H[j*n+j]);for(int k=0;k<j;k++)H[k*n+j]=H[j*n+k];}
   if(!(scale>0&&std::isfinite(scale)))return false;
   for(int attempt=0;attempt<12&&remaining_damped>0;attempt++){
    remaining_damped--;auto L=H;for(int j=0;j<n;j++)L[j*n+j]+=damping*scale;
    bool factor=true;for(int i=0;i<n&&factor;i++)for(int j=0;j<=i;j++){
     double v=L[i*n+j];for(int k=0;k<j;k++)v-=L[i*n+k]*L[j*n+k];
     if(i==j){if(!(v>0&&std::isfinite(v))){factor=false;break;}L[i*n+j]=std::sqrt(v);}
     else L[i*n+j]=v/L[j*n+j];
    }
    if(factor){
     std::vector<double>step=gradient;for(int i=0;i<n;i++){for(int j=0;j<i;j++)step[i]-=L[i*n+j]*step[j];step[i]/=L[i*n+i];}
     for(int i=n-1;i>=0;i--){for(int j=i+1;j<n;j++)step[i]-=L[j*n+i]*step[j];step[i]/=L[i*n+i];}
     if(trialStep(step)){accepted=true;stats.damped_steps++;damping=std::max(1e-16,damping*.2);break;}
    }
    damping=std::min(1e12,damping*10);
   }
   if(!accepted)return false;
  }
  if(alpha==1&&finalGate(p))return true;
  std::vector<double>F;return equations(p,alpha,F,nullptr)<=std::min(1e-10,tolerance*.1);
 };
 std::vector<double>warm(n);for(int i=0;i<n;i++)warm[i]=x[i];
 auto accept=[&](std::vector<double>&candidate){if(!finalGate(candidate))return false;std::vector<double>F;stats.residual=equations(candidate,1,F,nullptr,true);for(int i=0;i<n;i++)x[i]=candidate[i];return true;};
 stage(warm,1,64);if(accept(warm))return true;
 std::vector<double>cold(n,0);stage(cold,1,64);if(accept(cold))return true;
 for(auto base:{warm,cold}){
  std::vector<std::pair<double,int>>mode;
  std::vector<double>w(n);for(int i=0;i<n;i++){w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*base[j];}
  for(int ci=0;ci<static_cast<int>(contacts.size());ci++){auto c=contacts[ci];double cap=c.mu*std::max(0.,base[c.k]);double slip=std::hypot(w[c.t],w[c.s]);if(cap>0&&slip>tolerance)mode.push_back({slip,ci});}
  std::stable_sort(mode.begin(),mode.end(),[](auto a,auto b){return a.first>b.first;});
  for(int m=0;m<std::min(2,static_cast<int>(mode.size()))&&remaining_steps>0;m++){
   auto c=contacts[mode[m].second];auto trial=base;double slip=std::hypot(w[c.t],w[c.s]),cap=c.mu*std::max(0.,base[c.k]);trial[c.t]=-cap*w[c.t]/slip;trial[c.s]=-cap*w[c.s]/slip;stage(trial,1,64);if(accept(trial))return true;
  }
 }
 std::vector<double>F;stats.residual=equations(warm,1,F,nullptr,true);return false;
}
}
