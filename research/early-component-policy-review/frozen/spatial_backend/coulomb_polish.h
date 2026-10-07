// Rare recovery of the SAME non-associated circular-contact equations.
// Numerical SVD, pressure gauge exploration and cold restart add no compliance.
#pragma once
#include "newton_linear.h"
namespace circular_polish {
struct Contact {int k,t,s;double mu,rn,rt;};
inline bool solve(const btMatrixXu& A,const btVectorXu& b,btVectorXu& x,const btVectorXu& hi,const btAlignedObjectArray<int>& dep,double tolerance,CoulombStats& stats){
 const int n=b.rows();if(n>384)return false;
 int remaining_svd_calls=256;
 std::vector<Contact> contacts;
 for(int k=0;k<n;k++)if(dep[k]<0){std::vector<int> ts;for(int j=0;j<n;j++)if(dep[j]==k)ts.push_back(j);int t=ts[0],s=ts[1];double a=A(t,t),d=A(s,s),off=A(t,s);contacts.push_back({k,t,s,static_cast<double>(hi[t]),1/A(k,k),2/(a+d+std::hypot(a-d,2*off))});}
 auto equations=[&](const std::vector<double>& p,std::vector<double>& F,std::vector<double>* J){
  F.assign(n,0);std::vector<double>w(n);for(int i=0;i<n;i++){w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*p[j];}
  if(J)J->assign(n*n,0);
  double maximum=0;
  for(auto c:contacts){
   double zn=p[c.k]-c.rn*w[c.k];F[c.k]=(p[c.k]-std::max(0.,zn))/c.rn;
   if(J)for(int j=0;j<n;j++)(*J)[c.k*n+j]=zn>0?A(c.k,j):(j==c.k?1/c.rn:0);
   int rows[2]={c.t,c.s};double z[2]={p[c.t]-c.rt*w[c.t],p[c.s]-c.rt*w[c.s]};double length=std::hypot(z[0],z[1]),cap=c.mu*std::max(0.,p[c.k]);
   if(length<=cap&&cap>0){for(int r:rows){F[r]=w[r];if(J)for(int j=0;j<n;j++)(*J)[r*n+j]=A(r,j);}}
   else{
    double direction[2]={length>0?z[0]/length:0,length>0?z[1]/length:0};
    for(int u=0;u<2;u++){int r=rows[u];F[r]=(p[r]-cap*direction[u])/c.rt;if(J)for(int j=0;j<n;j++){
     double value=j==r?1.:0.;for(int h=0;h<2;h++){double D=length>0?cap/length*((u==h?1.:0.)-direction[u]*direction[h]):0;value-=D*((j==rows[h]?1.:0.)-c.rt*A(rows[h],j));}
     if(j==c.k&&p[c.k]>0)value-=c.mu*direction[u];
     (*J)[r*n+j]=value/c.rt;
    }}
   }
   maximum=std::max({maximum,std::abs(F[c.k]),std::hypot(F[c.t],F[c.s])});
  }
  for(double value:F)if(!std::isfinite(value))return std::numeric_limits<double>::infinity();
  return maximum;
 };
 auto gate=[&](std::vector<double>& p){
  for(auto c:contacts){if(!std::isfinite(p[c.k])||p[c.k]>hi[c.k])return false;if(p[c.k]<0){if(p[c.k]<-tolerance*c.rn)return false;p[c.k]=0;}}
  std::vector<double>F;double error=equations(p,F,nullptr);if(error>tolerance)return false;
  double change=0,scale=1;for(int i=0;i<n;i++){double w=-b[i];for(int j=0;j<n;j++)w+=A(i,j)*p[j];change+=.5*p[i]*(w-b[i]);scale+=std::abs(p[i]*b[i]);}
  if(!std::isfinite(change)||!std::isfinite(scale)||change>tolerance*scale)return false;
  for(int i=0;i<n;i++)x[i]=p[i];
  stats.last_residual=error;stats.residual_max=std::max(stats.residual_max,error);stats.passive_change_max=std::max(stats.passive_change_max,change);stats.solves++;stats.polish_solves++;return true;
 };
 auto newton=[&](std::vector<double>& p,double rank_cutoff=1e-12){
  for(int iteration=0;iteration<64;iteration++){
   if(gate(p))return true;
   std::vector<double>F,J;equations(p,F,&J);std::vector<double>rhs=F;double merit=0;for(double& value:rhs){merit+=value*value;value=-value;}
   if(remaining_svd_calls<=0)return false;
   remaining_svd_calls--;stats.polish_svd_calls++;
   auto linear=minimumNormNewton(J,rhs,n,rank_cutoff);if(!linear.converged){stats.polish_svd_rejections++;return false;}bool accepted=false;
   for(int line=0;line<30;line++){
    double alpha=std::ldexp(1.,-line);std::vector<double>trial=p;for(int i=0;i<n;i++)trial[i]+=alpha*linear.step[i];std::vector<double>next;equations(trial,next,nullptr);double value=0;for(double f:next)value+=f*f;
    if(std::isfinite(value)&&value<=(1-1e-4*alpha)*merit){p=std::move(trial);stats.polish_steps++;accepted=true;break;}
   }
   if(!accepted)break;
  }
  return gate(p);
 };
 auto gauges=[&](const std::vector<double>& p){
  std::vector<std::vector<double>> out;std::vector<double>F,J;equations(p,F,&J);std::vector<double>zero(n,0);
  if(remaining_svd_calls<=0)return out;
  remaining_svd_calls--;stats.polish_svd_calls++;
  auto linear=minimumNormNewton(J,zero,n);if(!linear.converged){stats.polish_svd_rejections++;return out;}
  double mobility_scale=1;for(int i=0;i<n;i++){double row=0;for(int j=0;j<n;j++)row+=std::abs(A(i,j));mobility_scale=std::max(mobility_scale,row);}
  int tried=0;
  for(auto direction:linear.nullspace){
   if(tried++>=4)break;
   int pivot=0;for(int i=0;i<n;i++)if(std::abs(direction[i])>std::abs(direction[pivot]))pivot=i;if(direction[pivot]<0)for(double& v:direction)v=-v;
   double response=0;for(int i=0;i<n;i++){double value=0;for(int j=0;j<n;j++)value+=A(i,j)*direction[j];response=std::max(response,std::abs(value));}if(response>1e-12*mobility_scale)continue;
   for(double sign:{1.,-1.}){
    auto d=direction;for(double& value:d)value*=sign;double limit=std::numeric_limits<double>::infinity();bool blocked=false;
    for(auto c:contacts){
     if(std::hypot(d[c.k],std::hypot(d[c.t],d[c.s]))<1e-10)continue;
     if(d[c.k]<0){double bound=-p[c.k]/d[c.k];if(bound<=1e-10){blocked=true;break;}limit=std::min(limit,bound);}
     double a=d[c.t]*d[c.t]+d[c.s]*d[c.s]-c.mu*c.mu*d[c.k]*d[c.k];
     double bb=2*(p[c.t]*d[c.t]+p[c.s]*d[c.s]-c.mu*c.mu*p[c.k]*d[c.k]);
     double cc=p[c.t]*p[c.t]+p[c.s]*p[c.s]-c.mu*c.mu*p[c.k]*p[c.k];
     if(std::abs(cc)<1e-12*std::max(1.,p[c.t]*p[c.t]+p[c.s]*p[c.s]))cc=0;
     if(cc==0&&bb>1e-12){blocked=true;break;}
     auto root=[&](double value){if(std::isfinite(value)&&value>1e-10&&2*a*value+bb>=-1e-12)limit=std::min(limit,value);};
     if(std::abs(a)>1e-20){double disc=bb*bb-4*a*cc;if(disc>=0){double q=-.5*(bb+std::copysign(std::sqrt(disc),bb));if(q!=0){root(q/a);root(cc/q);}else root(-bb/(2*a));}}
     else if(std::abs(bb)>1e-20)root(-cc/bb);
    }
    if(blocked||!std::isfinite(limit))continue;
    auto trial=p;for(int i=0;i<n;i++)trial[i]+=limit*d[i];bool feasible=true;
    for(auto c:contacts)feasible&=trial[c.k]>=-1e-10&&std::hypot(trial[c.t],trial[c.s])<=c.mu*std::max(0.,trial[c.k])+1e-10;
    double change=0;for(int i=0;i<n;i++){double value=0;for(int j=0;j<n;j++)value+=A(i,j)*(trial[j]-p[j]);change=std::max(change,std::abs(value));}
    if(feasible&&change<=.01*tolerance)out.push_back(std::move(trial));
   }
  }
  return out;
 };
 std::vector<double>original(n);for(int i=0;i<n;i++)original[i]=x[i];auto p=original;
 if(newton(p))return true;
 auto stagnated=p;
 // A weak numerical Jacobian mode can demand an enormous Newton increment
 // from a tiny residual. One truncated-J retry changes only the search step,
 // never physical mobility or the exact final equations/capacity/energy gate.
 auto truncated=original;stats.rank_restarts++;
 if(newton(truncated,1e-10))return true;
 // A feasible opposing-slip guess may cross a merit basin that neutral pressure
 // relocation cannot escape. This trial is never applied to bodies. The SAME
 // final contact/energy gate alone accepts its converged result.
 for(const auto& start:{original,stagnated}){
  int attempts=0;
  for(auto c:contacts){
   if(attempts>=8||remaining_svd_calls<=0)break;
   if(c.mu*start[c.k]<=0)continue;
   double wt=-b[c.t],ws=-b[c.s];for(int j=0;j<n;j++){wt+=A(c.t,j)*start[j];ws+=A(c.s,j)*start[j];}
   double slip=std::hypot(wt,ws);if(slip<=tolerance*.01)continue;
   auto trial=start;double cap=c.mu*std::max(0.,start[c.k]);trial[c.t]=-cap*wt/slip;trial[c.s]=-cap*ws/slip;
   attempts++;stats.opposing_restarts++;if(newton(trial))return true;
  }
 }
 for(const auto& start:{original,p})for(auto trial:gauges(start)){stats.gauge_restarts++;if(newton(trial))return true;}
 stats.cold_restarts++;p.assign(n,0);if(newton(p))return true;
 if(remaining_svd_calls<=0)stats.polish_budget_rejections++;
 return false;
}
}
