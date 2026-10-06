// Project adapter: circular, non-associated Coulomb friction. Bullet is unmodified.
#pragma once
#include "normal_qp.h"
#include "mobility_continuation.h"
#include "reduced_mobility_continuation.h"
#include "terminal_component_polish.h"
#include <stdexcept>
#include <sstream>
#include <functional>
#include <limits>
#include "coulomb_trust.h"
#include "translation_split.h"
#include "translation_combined.h"
#ifdef SPATIAL_LAPACK_RECOVERY
#include "projection_more.h"
#include "null_traction_seed.h"
#endif
#include "pressure_release.h"
#include "normal_null.h"
#include "coulomb_active.h"
#ifdef SPATIAL_LAPACK_RECOVERY
#include "support_restart.h"
#endif

inline bool coulombLapackRecoveryEnabled(){
#ifdef SPATIAL_LAPACK_RECOVERY
 return true;
#else
 return false;
#endif
}

// Upstream friction RHS omits this angular free-velocity increment, although
// normal RHS and final body writeback include it. Preserve the signed B row.
inline double consistentTangentRHS(double rhs,const btSolverConstraint& c,
 const btSolverBody& a,const btSolverBody& b){
 return rhs-c.m_relpos1CrossNormal.dot(a.m_externalTorqueImpulse)
           -c.m_relpos2CrossNormal.dot(b.m_externalTorqueImpulse);
}

struct CoulombStats {
 int terminal_polish_attempts=0,terminal_polish_solves=0,terminal_polish_declines=0;terminal_component_polish::Stats terminal_polish;
 int reduced_mobility_attempts=0,reduced_mobility_solves=0,reduced_mobility_declines=0;reduced_mobility_continuation::Stats reduced_mobility;
 int mobility_attempts=0,mobility_solves=0,mobility_declines=0;mobility_continuation::Stats mobility;
 int projection_attempts=0,projection_solves=0,projection_declines=0;
 int projection_svd_calls=0,projection_iteration_steps=0,projection_newton_steps=0;
 // Optional schedule receipts; these aggregate across calls, never set caps.
 int early_component_attempts=0,early_component_solves=0,early_component_declines=0;
 int early_component_helper_calls=0,early_component_skipped_components=0;
 int early_component_cap_rejections=0,early_component_passes=0,early_component_largest_rows=0;
 int early_component_expanded_contacts=0,early_component_svd_calls=0;
 int early_component_iteration_steps=0,early_component_pressure_svd_calls=0;
 int early_component_pressure_attempts=0,early_component_pivot_calls=0;
#ifdef SPATIAL_LAPACK_RECOVERY
 int null_seed_attempts=0,null_seed_solves=0,null_seed_declines=0;null_traction_seed::Stats null_seed;
#endif
 int solves=0,sweeps_max=0,fast_solves=0,newton_steps=0,polish_solves=0,polish_steps=0,gauge_restarts=0,cold_restarts=0,polish_svd_calls=0,polish_budget_rejections=0,polish_svd_rejections=0,rank_restarts=0,opposing_restarts=0;
 double residual_max=0,passive_change_max=0,last_residual=0;
 normal_pressure::Stats pressure;int pressure_solves=0;
 normal_null::Stats null_pressure;
 int supplemental_solves=0,supplemental_svd_calls=0,supplemental_iteration_steps=0,supplemental_damped_steps=0,supplemental_pressure_svd_calls=0,supplemental_pressure_attempts=0,supplemental_pivot_calls=0,supplemental_projector_calls=0,supplemental_restarts=0;
 double supplemental_null_response_max=0;
 int support_solves=0,support_helper_calls=0,support_skipped_components=0,support_component_cap_rejections=0,support_passes=0,support_largest_rows=0,support_expanded_contacts=0,support_svd_calls=0,support_iteration_steps=0,support_pressure_svd_calls=0,support_pivot_calls=0;
 circular_active::Stats active;int active_solves=0;
 int continuation_solves=0;unsigned long long iteration_sweeps_total=0;
 circular_trust::Stats continuation;
};

#include "coulomb_polish.h"

// The same scalar rho is used for both tangent coordinates, preserving rotations
// of their basis. Normal complementarity remains independent of the slip cone.
inline bool coulombIterate(const btMatrixXu& A,const btVectorXu& b,btVectorXu& x,
 const btVectorXu& lo,const btVectorXu& hi,const btAlignedObjectArray<int>& dep,
 int budget,double tolerance,CoulombStats& stats,std::vector<double>* rejected_impulses=nullptr){
 const int n=b.rows();
 struct Contact {int normal;std::vector<int> tangent;double mu=0;};
 std::vector<Contact> contacts;std::vector<int> map(n,-1);
 for(int i=0;i<n;i++)if(dep[i]<0){
  if(lo[i]!=0||hi[i]<1e9||!(A(i,i)>0))throw std::runtime_error("Unsupported Coulomb normal row");
  map[i]=static_cast<int>(contacts.size());contacts.push_back({i,{},0});
 }
 for(int i=0;i<n;i++)if(dep[i]>=0){
  if(dep[i]>=n||map[dep[i]]<0||lo[i]!=-hi[i]||hi[i]<0)throw std::runtime_error("Unsupported Coulomb tangent row");
  auto& c=contacts[map[dep[i]]];
  if(!c.tangent.empty()&&hi[i]!=c.mu)throw std::runtime_error("Anisotropic coefficients are not supported");
  c.mu=hi[i];c.tangent.push_back(i);
 }
 for(auto& c:contacts)if(c.tangent.size()!=2)throw std::runtime_error("Two tangents per contact required");
 // Exact zeros only: retain every normal/tangent and inter-contact coupling.
 // Dense Bullet assembly is still used; iterative propagation uses sparse columns.
 std::vector<std::vector<std::pair<int,double>>> columns(n);
 for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(A(i,j)!=0)columns[j].push_back({i,A(i,j)});
 std::vector<double> p(n),w(n);
 for(auto& c:contacts){
  p[c.normal]=std::max(0.,static_cast<double>(x[c.normal]));
  int t=c.tangent[0],s=c.tangent[1];p[t]=x[t];p[s]=x[s];
  double cap=c.mu*p[c.normal],length=std::hypot(p[t],p[s]);
  if(length>cap){p[t]*=cap/length;p[s]*=cap/length;}
 }
 auto recompute=[&](){for(int i=0;i<n;i++)w[i]=-b[i];for(int j=0;j<n;j++)for(auto e:columns[j])w[e.first]+=e.second*p[j];};
 auto update=[&](int j,double value){double delta=value-p[j];p[j]=value;for(auto e:columns[j])w[e.first]+=e.second*delta;};
 auto residual=[&](){
  double maximum=0;
  for(int i=0;i<n;i++)if(!std::isfinite(p[i])||!std::isfinite(w[i]))return std::numeric_limits<double>::infinity();
  for(auto& c:contacts){
   int k=c.normal,t=c.tangent[0],s=c.tangent[1];double a=A(t,t),d=A(s,s),off=A(t,s);
   double eigen=.5*(a+d+std::hypot(a-d,2*off));
   if(!(eigen>0))throw std::runtime_error("Degenerate Coulomb tangent mobility");
   maximum=std::max(maximum,std::abs(p[k]-std::max(0.,p[k]-w[k]/A(k,k)))*A(k,k));
   double zt=p[t]-w[t]/eigen,zs=p[s]-w[s]/eigen;
   double length=std::hypot(zt,zs),cap=c.mu*p[k],factor=length>cap?cap/length:1.;
   maximum=std::max(maximum,std::hypot(p[t]-zt*factor,p[s]-zs*factor)*eigen);
  }
  return maximum;
 };
 // Semismooth Newton acceleration uses the same circular-contact equations.
 // Free impulse coordinates are set to zero at rank-deficient faces; no compliance is added.
 auto newton=[&](){
  if(n>512)return false;
  std::vector<double> F(n,0),J(n*n,0);
  for(auto& c:contacts){
   int k=c.normal,t=c.tangent[0],s=c.tangent[1];double rho=1/A(k,k),zn=p[k]-rho*w[k];
   F[k]=(p[k]-std::max(0.,zn))/rho;
   for(int j=0;j<n;j++)J[k*n+j]=zn>0?A(k,j):(j==k?1/rho:0);
   double a=A(t,t),d=A(s,s),off=A(t,s);rho=2/(a+d+std::hypot(a-d,2*off));
   double z[2]={p[t]-rho*w[t],p[s]-rho*w[s]},length=std::hypot(z[0],z[1]),cap=c.mu*std::max(0.,p[k]);int rows[2]={t,s};
   if(length<=cap&&cap>0){
    for(int r:rows){F[r]=w[r];for(int j=0;j<n;j++)J[r*n+j]=A(r,j);}
   }else{
    double direction[2]={length>0?z[0]/length:0,length>0?z[1]/length:0};
    for(int u=0;u<2;u++){
     int r=rows[u];F[r]=(p[r]-cap*direction[u])/rho;
     for(int j=0;j<n;j++){
      double v=j==r?1.:0.;
      for(int h=0;h<2;h++){
       double dp=length>0?cap/length*((u==h?1.:0.)-direction[u]*direction[h]):0;
       v-=dp*((j==rows[h]?1.:0.)-rho*A(rows[h],j));
      }
      if(j==k&&p[k]>0)v-=c.mu*direction[u];
      J[r*n+j]=v/rho;
     }
    }
   }
  }
  double merit=0,scale=0;for(double f:F)merit+=f*f;for(double v:J)scale=std::max(scale,std::abs(v));
  auto rhs=F;for(int i=0;i<n;i++){rhs[i]=-F[i];for(int j=0;j<n;j++)rhs[i]+=J[i*n+j]*p[j];}
  std::vector<int> pivots;int rank=0;
  for(int col=0;col<n;col++){
   int row=rank;for(int r=rank;r<n;r++)if(std::abs(J[r*n+col])>std::abs(J[row*n+col]))row=r;
   if(std::abs(J[row*n+col])<=1e-12*scale)continue;
   for(int j=col;j<n;j++)std::swap(J[rank*n+j],J[row*n+j]);
   std::swap(rhs[rank],rhs[row]);
   for(int r=rank+1;r<n;r++){
    double factor=J[r*n+col]/J[rank*n+col];J[r*n+col]=0;
    for(int j=col+1;j<n;j++)J[r*n+j]-=factor*J[rank*n+j];
    rhs[r]-=factor*rhs[rank];
   }
   pivots.push_back(col);if(++rank==n)break;
  }
  std::vector<double> step(n,0),old=p;
  for(int i=rank-1;i>=0;i--){int col=pivots[i];double value=rhs[i];for(int j=col+1;j<n;j++)value-=J[i*n+j]*step[j];step[col]=value/J[i*n+col];}
  for(int i=0;i<n;i++)step[i]-=old[i];
  for(int line=0;line<24;line++){
   double alpha=std::ldexp(1.,-line);for(int i=0;i<n;i++)p[i]=old[i]+alpha*step[i];
   for(auto& c:contacts)p[c.normal]=std::max(0.,p[c.normal]);
   recompute();double trial=0;
   for(auto& c:contacts){
    int k=c.normal,t=c.tangent[0],ss=c.tangent[1];double a=A(t,t),d=A(ss,ss),off=A(t,ss),eigen=.5*(a+d+std::hypot(a-d,2*off));
    double fn=(p[k]-std::max(0.,p[k]-w[k]/A(k,k)))*A(k,k);trial+=fn*fn;
    double zt=p[t]-w[t]/eigen,zs=p[ss]-w[ss]/eigen,length=std::hypot(zt,zs),cap=c.mu*p[k],factor=length>cap?cap/length:1.;
    double ft=(p[t]-zt*factor)*eigen,fs=(p[ss]-zs*factor)*eigen;trial+=ft*ft+fs*fs;
   }
   if(std::isfinite(trial)&&trial<(1-1e-4*alpha)*merit){stats.newton_steps++;return true;}
  }
  p=old;recompute();return false;
 };
 recompute();
 for(int sweep=0;sweep<=budget;sweep++){
  if(sweep%8==0||sweep==budget){
   recompute();double error=residual();stats.last_residual=error;
   if(error<=tolerance){
    double change=0,scale=1;
    for(int i=0;i<n;i++){change+=.5*p[i]*(w[i]-b[i]);scale+=std::abs(p[i]*b[i]);}
    // With e=0 and separate position correction this is an upper bound on
    // E_after-E_before-W_wall. Positive-gap targets add a conservative term.
    if(!std::isfinite(change)||!std::isfinite(scale)||change>tolerance*scale)throw std::runtime_error("Coulomb passivity gate failed");
    for(auto& c:contacts)if(p[c.normal]>hi[c.normal]){if(rejected_impulses)*rejected_impulses=p;return false;}
    for(int i=0;i<n;i++)x[i]=p[i];
    stats.solves++;stats.sweeps_max=std::max(stats.sweeps_max,sweep);
    stats.fast_solves+=sweep<=8;stats.residual_max=std::max(stats.residual_max,error);
    stats.passive_change_max=std::max(stats.passive_change_max,change);return true;
   }
   if(sweep==budget){if(rejected_impulses)*rejected_impulses=p;return false;}
  }
  stats.iteration_sweeps_total++;
  if(sweep>=32&&sweep%16==0)newton();
  // Coupled block Gauss-Seidel: solve a normal, then its circular tangent block.
  for(auto& c:contacts){
   int k=c.normal,t=c.tangent[0],s=c.tangent[1];
   update(k,std::max(0.,p[k]-w[k]/A(k,k)));
   double cap=c.mu*p[k];
   if(cap==0){update(t,0);update(s,0);continue;}
   double a=A(t,t),d=A(s,s),off=A(t,s);
   double qt=w[t]-a*p[t]-off*p[s],qs=w[s]-off*p[t]-d*p[s];
   auto solve=[&](double lambda){double aa=a+lambda,dd=d+lambda,det=aa*dd-off*off;
    if(!(det>0))throw std::runtime_error("Singular tangent block");
    return std::pair<double,double>{(-dd*qt+off*qs)/det,(off*qt-aa*qs)/det};};
   auto answer=solve(0);
   if(std::hypot(answer.first,answer.second)>cap){
    double low=0,high=std::hypot(qt,qs)/cap;
    // high bounds the Lagrange multiplier for the circular trust region.
    for(int j=0;j<48;j++){double mid=.5*(low+high);auto trial=solve(mid);if(std::hypot(trial.first,trial.second)>cap)low=mid;else high=mid;}
    answer=solve(high);
   }
   update(t,answer.first);update(s,answer.second);
  }
 }
 return false;
}

inline double coulombResidual(const btMatrixXu& A,const btVectorXu& b,const btVectorXu& x,
 const btVectorXu& hi,const btAlignedObjectArray<int>& dep){
 const int n=b.rows();std::vector<double>w(n);
 for(int i=0;i<n;i++){
  w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*x[j];
  if(!std::isfinite(w[i])||!std::isfinite(x[i]))return std::numeric_limits<double>::infinity();
 }
 double maximum=0;
 for(int k=0;k<n;k++)if(dep[k]<0){
  std::vector<int> tangents;for(int i=0;i<n;i++)if(dep[i]==k)tangents.push_back(i);
  if(tangents.size()!=2||!(A(k,k)>0))return std::numeric_limits<double>::infinity();
  const int t=tangents[0],s=tangents[1];
  const double eigen=.5*(A(t,t)+A(s,s)+std::hypot(A(t,t)-A(s,s),2*A(t,s)));
  if(!(eigen>0&&std::isfinite(eigen)))return std::numeric_limits<double>::infinity();
  maximum=std::max(maximum,std::abs(x[k]-std::max(0.,static_cast<double>(x[k])-w[k]/A(k,k)))*A(k,k));
  const double zt=x[t]-w[t]/eigen,zs=x[s]-w[s]/eigen,length=std::hypot(zt,zs),cap=hi[t]*x[k];
  const double factor=length>cap&&length>0?cap/length:1.;
  maximum=std::max(maximum,std::hypot(x[t]-zt*factor,x[s]-zs*factor)*eigen);
 }
 return maximum;
}

inline bool coulombSolve(const btMatrixXu& A,const btVectorXu& b,btVectorXu& x,
 const btVectorXu& lo,const btVectorXu& hi,const btAlignedObjectArray<int>& dep,
 int budget,double tolerance,CoulombStats& stats,std::vector<double>* rejected_impulses=nullptr,bool allow_recovery=true,bool early_component_recovery=false){
 std::vector<double> rejected;
 const bool recover=allow_recovery&&budget>=64;
 const int first_budget=recover?std::min(budget,256):budget;
 if(coulombIterate(A,b,x,lo,hi,dep,first_budget,tolerance,stats,&rejected))return true;
 // Short initial iteration phase, followed by original-law recovery. Numerical
 // trials are not applied. If search fails, preserve the remaining sweep budget.
 btVectorXu candidate=x;for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
#ifdef SPATIAL_LAPACK_RECOVERY
 // RESEARCH OPTION: default false preserves the frozen lane ordering.
 // Use the actual first256 rejected iterate, not a cold/captured proxy seed.
 if(early_component_recovery&&recover&&first_budget==256&&b.rows()<=4096){
  support_restart_v3::Stats early; // Fresh independent caps for THIS call.
  stats.early_component_attempts++;
  const bool accepted=support_restart_v3::solve(A,b,candidate,hi,dep,tolerance,early);
  stats.early_component_helper_calls+=early.helper_calls;
  stats.early_component_skipped_components+=early.skipped_components;
  stats.early_component_cap_rejections+=early.component_cap_rejections;
  stats.early_component_passes+=early.passes;
  stats.early_component_largest_rows=std::max(stats.early_component_largest_rows,early.largest_reduced_rows);
  stats.early_component_expanded_contacts+=early.expanded_contacts;
  stats.early_component_svd_calls+=early.svd_calls;
  stats.early_component_iteration_steps+=early.iteration_steps;
  stats.early_component_pressure_svd_calls+=early.pressure_svd_calls;
  stats.early_component_pressure_attempts+=early.pressure_attempts;
  stats.early_component_pivot_calls+=early.pivot_attempts;
  // Aggregate BOTH early and existing later tail work, including declines.
  stats.support_helper_calls+=early.helper_calls;
  stats.support_skipped_components+=early.skipped_components;
  stats.support_component_cap_rejections+=early.component_cap_rejections;
  stats.support_passes+=early.passes;
  stats.support_largest_rows=std::max(stats.support_largest_rows,early.largest_reduced_rows);
  stats.support_expanded_contacts+=early.expanded_contacts;
  stats.support_svd_calls+=early.svd_calls;
  stats.support_iteration_steps+=early.iteration_steps;
  stats.support_pressure_svd_calls+=early.pressure_svd_calls;
  stats.support_pivot_calls+=early.pivot_attempts;
  if(accepted){
   // V3 returns true only after its ORIGINAL full-row/bounds/passivity gate.
   x=candidate;stats.solves++;stats.support_solves++;stats.early_component_solves++;
   stats.last_residual=early.residual;stats.residual_max=std::max(stats.residual_max,early.residual);
   stats.sweeps_max=std::max(stats.sweeps_max,first_budget);
   double change=0;for(int i=0;i<b.rows();i++){double w=-b[i];for(int j=0;j<b.rows();j++)w+=A(i,j)*x[j];change+=.5*x[i]*(w-b[i]);}
   stats.passive_change_max=std::max(stats.passive_change_max,change);
   return true;
  }
  stats.early_component_declines++;
  // A declined trial NEVER reaches x; restore the exact original next-lane seed.
  candidate=x;for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
 }
#endif
 if(recover){
  bool normal_only=true;std::vector<int> normals;
  for(int i=0;i<b.rows();i++){if(dep[i]<0)normals.push_back(i);else normal_only&=hi[i]==0;}
  if(normal_only){
   const int n=static_cast<int>(normals.size());btMatrixXu N(n,n);btVectorXu rhs(n),upper(n),seed(n),answer(n);
   for(int i=0;i<n;i++){rhs[i]=b[normals[i]];upper[i]=hi[normals[i]];seed[i]=rejected[normals[i]];for(int j=0;j<n;j++)N.setElem(i,j,A(normals[i],normals[j]));}
   const bool null_ok=normal_null::solve(N,rhs,upper,seed,answer,tolerance,stats.null_pressure);
   if(null_ok||normal_pressure::solve(N,rhs,upper,seed,answer,tolerance,stats.pressure)){
    x.setZero();for(int i=0;i<n;i++)x[normals[i]]=answer[i];stats.solves++;if(!null_ok)stats.pressure_solves++;
    stats.last_residual=null_ok?stats.null_pressure.residual:stats.pressure.residual;stats.residual_max=std::max(stats.residual_max,stats.last_residual);
    double change=0;for(int i=0;i<n;i++){double w=-rhs[i];for(int j=0;j<n;j++)w+=N(i,j)*answer[j];change+=.5*answer[i]*(w-rhs[i]);}
    stats.passive_change_max=std::max(stats.passive_change_max,change);stats.sweeps_max=std::max(stats.sweeps_max,first_budget);return true;
   }
  }
 }
 if(recover&&circular_active::solve(A,b,candidate,hi,dep,tolerance,stats.active)){
  x=candidate;stats.solves++;stats.active_solves++;stats.last_residual=stats.active.residual;
  stats.residual_max=std::max(stats.residual_max,stats.last_residual);
  double change=0;for(int i=0;i<b.rows();i++){double w=-b[i];for(int j=0;j<b.rows();j++)w+=A(i,j)*x[j];change+=.5*x[i]*(w-b[i]);}
  stats.passive_change_max=std::max(stats.passive_change_max,change);stats.sweeps_max=std::max(stats.sweeps_max,first_budget);return true;
 }
 if(recover&&circular_trust::solve(A,b,candidate,hi,dep,tolerance,stats.continuation)){
  x=candidate;stats.solves++;stats.continuation_solves++;
  double change=0;for(int i=0;i<b.rows();i++){double w=-b[i];for(int j=0;j<b.rows();j++)w+=A(i,j)*x[j];change+=.5*x[i]*(w-b[i]);}
  stats.passive_change_max=std::max(stats.passive_change_max,change);
  stats.last_residual=stats.continuation.residual;
  stats.residual_max=std::max(stats.residual_max,stats.last_residual);
  stats.sweeps_max=std::max(stats.sweeps_max,first_budget);return true;
 }
 if(recover&&circular_polish::solve(A,b,candidate,hi,dep,tolerance,stats)){
  x=candidate;stats.sweeps_max=std::max(stats.sweeps_max,first_budget);return true;
 }
 if(first_budget<budget){
  for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
  if(coulombIterate(A,b,candidate,lo,hi,dep,budget-first_budget,tolerance,stats,&rejected)){x=candidate;return true;}
 }
#ifdef SPATIAL_LAPACK_RECOVERY
 if(recover&&b.rows()<=64){
  // Fresh per-call work counters: prior contacts must not consume this budget.
  candidate=x;circular_restart::Stats supplemental;
  const bool accepted=circular_restart::solve(A,b,candidate,hi,dep,tolerance,supplemental);
  stats.supplemental_svd_calls+=supplemental.svd_calls;
  stats.supplemental_iteration_steps+=supplemental.iteration_steps;
  stats.supplemental_damped_steps+=supplemental.direct.damped_steps+supplemental.neutral.damped_steps;
  stats.supplemental_pressure_svd_calls+=supplemental.neutral.pressure_svd_calls;
  stats.supplemental_pressure_attempts+=supplemental.neutral.pressure_attempts;
  stats.supplemental_pivot_calls+=supplemental.neutral.normal_pivot_attempts;
  stats.supplemental_projector_calls+=supplemental.neutral.projector_calls;
  stats.supplemental_restarts+=supplemental.neutral.restarts;
  stats.supplemental_null_response_max=std::max(stats.supplemental_null_response_max,supplemental.neutral.maximum_neutral_velocity);
  if(accepted){
   x=candidate;stats.solves++;stats.supplemental_solves++;
   stats.last_residual=supplemental.residual;stats.residual_max=std::max(stats.residual_max,stats.last_residual);
   double change=0;for(int i=0;i<b.rows();i++){double w=-b[i];for(int j=0;j<b.rows();j++)w+=A(i,j)*x[j];change+=.5*x[i]*(w-b[i]);}
   stats.passive_change_max=std::max(stats.passive_change_max,change);return true;
  }
 }
 if(recover&&b.rows()<=4096){
  // Match the actual world's final rejected PGS seed; never apply that seed
  // unless the complete original system passes the supplemental physical gate.
  for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
  support_restart_v3::Stats support;
  const bool accepted=support_restart_v3::solve(A,b,candidate,hi,dep,tolerance,support);
  stats.support_helper_calls+=support.helper_calls;stats.support_skipped_components+=support.skipped_components;
  stats.support_component_cap_rejections+=support.component_cap_rejections;stats.support_passes+=support.passes;
  stats.support_largest_rows=std::max(stats.support_largest_rows,support.largest_reduced_rows);
  stats.support_expanded_contacts+=support.expanded_contacts;stats.support_svd_calls+=support.svd_calls;
  stats.support_iteration_steps+=support.iteration_steps;stats.support_pressure_svd_calls+=support.pressure_svd_calls;
  stats.support_pivot_calls+=support.pivot_attempts;
  if(accepted){
   x=candidate;stats.solves++;stats.support_solves++;
   stats.last_residual=support.residual;stats.residual_max=std::max(stats.residual_max,stats.last_residual);
   double change=0;for(int i=0;i<b.rows();i++){double w=-b[i];for(int j=0;j<b.rows();j++)w+=A(i,j)*x[j];change+=.5*x[i]*(w-b[i]);}
   stats.passive_change_max=std::max(stats.passive_change_max,change);return true;
  }
 }

 if(recover&&b.rows()<=64){
  // A fresh, bounded search only after ALL prior original-law lanes decline.
  for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
  projection_recovery_v2::Stats projection;
  stats.projection_attempts++;
  const bool found=projection_recovery_v2::solve(A,b,candidate,hi,dep,tolerance,projection);
  stats.projection_svd_calls+=projection.svd_calls;
  stats.projection_iteration_steps+=projection.iteration_steps;
  stats.projection_newton_steps+=projection.newton_steps;
  const double original_residual=found?coulombResidual(A,b,candidate,hi,dep):std::numeric_limits<double>::infinity();
  if(found&&std::isfinite(original_residual)&&original_residual<=tolerance){
   double change=0,scale=1;bool bounds=true;
   for(int i=0;i<b.rows();i++){
    double w=-b[i];for(int j=0;j<b.rows();j++)w+=A(i,j)*candidate[j];
    change+=.5*candidate[i]*(w-b[i]);scale+=std::abs(candidate[i]*b[i]);
    if(dep[i]<0)bounds&=std::isfinite(candidate[i])&&candidate[i]>=lo[i]&&candidate[i]<=hi[i];
   }
   if(bounds&&std::isfinite(change)&&std::isfinite(scale)&&change<=tolerance*scale){
    x=candidate;stats.solves++;stats.projection_solves++;
    stats.last_residual=original_residual;stats.residual_max=std::max(stats.residual_max,original_residual);
    stats.passive_change_max=std::max(stats.passive_change_max,change);return true;
   }
  }
  stats.projection_declines++;
 }
 // Preserve every previously accepted lane. New bounded search starts only on failure.
 if(recover&&b.rows()<=4096){
  for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
  null_traction_seed::Stats trial_stats;stats.null_seed_attempts++;
  const bool found=null_traction_seed::solve(A,b,candidate,hi,dep,candidate,tolerance,trial_stats);
  stats.null_seed.components+=trial_stats.components;stats.null_seed.largest_rows=std::max(stats.null_seed.largest_rows,trial_stats.largest_rows);stats.null_seed.cap_rejections+=trial_stats.cap_rejections;stats.null_seed.seed_attempts+=trial_stats.seed_attempts;stats.null_seed.null_svd_calls+=trial_stats.null_svd_calls;stats.null_seed.seed_svd_calls+=trial_stats.seed_svd_calls;stats.null_seed.iteration_steps+=trial_stats.iteration_steps;stats.null_seed.svd_calls+=trial_stats.svd_calls;stats.null_seed.newton_steps+=trial_stats.newton_steps;stats.null_seed.seed_response_change_max=std::max(stats.null_seed.seed_response_change_max,trial_stats.seed_response_change_max);
  CoulombStats gate_stats;
  // This original production gate includes eager cone projection and full passivity.
  if(found&&coulombIterate(A,b,candidate,lo,hi,dep,0,tolerance,gate_stats)){
   x=candidate;stats.solves++;stats.null_seed_solves++;stats.last_residual=gate_stats.last_residual;stats.residual_max=std::max(stats.residual_max,gate_stats.residual_max);stats.passive_change_max=std::max(stats.passive_change_max,gate_stats.passive_change_max);return true;
  }
  stats.null_seed_declines++;
 }

 // Search-only matrices are discarded; the authoritative original gate accepts.
 if(recover&&b.rows()<=4096){
  for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
  mobility_continuation::Stats trial;stats.mobility_attempts++;
  auto original_gate=[](const btMatrixXu&M,const btVectorXu&rhs,btVectorXu&q,const btVectorXu&lower,const btVectorXu&upper,const btAlignedObjectArray<int>&dependencies,double tol){CoulombStats gate;return coulombIterate(M,rhs,q,lower,upper,dependencies,0,tol,gate);};
  bool found=mobility_continuation::solve(A,b,candidate,lo,hi,dep,tolerance,trial,original_gate);
  stats.mobility.components+=trial.components;stats.mobility.largest_rows=std::max(stats.mobility.largest_rows,trial.largest_rows);stats.mobility.stage_attempts+=trial.stage_attempts;stats.mobility.stage_accepts+=trial.stage_accepts;stats.mobility.iteration_steps+=trial.iteration_steps;stats.mobility.svd_calls+=trial.svd_calls;
  CoulombStats gate;
  if(found&&coulombIterate(A,b,candidate,lo,hi,dep,0,tolerance,gate)){
   x=candidate;stats.solves++;stats.mobility_solves++;stats.last_residual=gate.last_residual;stats.residual_max=std::max(stats.residual_max,gate.residual_max);stats.passive_change_max=std::max(stats.passive_change_max,gate.passive_change_max);return true;
  }
  stats.mobility_declines++;
 }

 if(recover&&b.rows()<=4096){
  for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
  reduced_mobility_continuation::Stats trial;stats.reduced_mobility_attempts++;
  auto original_gate=[](const btMatrixXu&M,const btVectorXu&rhs,btVectorXu&q,const btVectorXu&lower,const btVectorXu&upper,const btAlignedObjectArray<int>&dependencies,double tol){CoulombStats gate;return coulombIterate(M,rhs,q,lower,upper,dependencies,0,tol,gate);};
  bool found=reduced_mobility_continuation::solve(A,b,candidate,lo,hi,dep,tolerance,trial,original_gate);
  auto&s=stats.reduced_mobility;s.components+=trial.components;s.largest_rows=std::max(s.largest_rows,trial.largest_rows);s.stage_attempts+=trial.stage_attempts;s.stage_accepts+=trial.stage_accepts;s.iteration_steps+=trial.iteration_steps;s.svd_calls+=trial.svd_calls;s.support_passes+=trial.support_passes;s.reduced_rows_max=std::max(s.reduced_rows_max,trial.reduced_rows_max);
  CoulombStats gate;
  if(found&&coulombIterate(A,b,candidate,lo,hi,dep,0,tolerance,gate)){
   x=candidate;stats.solves++;stats.reduced_mobility_solves++;stats.last_residual=gate.last_residual;stats.residual_max=std::max(stats.residual_max,gate.residual_max);stats.passive_change_max=std::max(stats.passive_change_max,gate.passive_change_max);return true;
  }
  stats.reduced_mobility_declines++;
 }

 // One bounded restart from the terminal rejected seed; all old lanes unchanged.
 if(recover&&b.rows()<=4096){
  for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
  terminal_component_polish::Stats trial;stats.terminal_polish_attempts++;
  auto gate=[](const btMatrixXu&M,const btVectorXu&rhs,btVectorXu&q,const btVectorXu&lower,const btVectorXu&upper,const btAlignedObjectArray<int>&d,double tol){CoulombStats s;return coulombIterate(M,rhs,q,lower,upper,d,0,tol,s);};
  auto iteration=[](const btMatrixXu&M,const btVectorXu&rhs,btVectorXu&q,const btVectorXu&lower,const btVectorXu&upper,const btAlignedObjectArray<int>&d,double tol,std::vector<double>&last,int&sweeps){CoulombStats s;bool ok=coulombIterate(M,rhs,q,lower,upper,d,256,tol,s,&last);sweeps=s.iteration_sweeps_total;return ok;};
  auto polish=[](const btMatrixXu&M,const btVectorXu&rhs,btVectorXu&q,const btVectorXu&upper,const btAlignedObjectArray<int>&d,double tol,int&svds,int&steps){CoulombStats s;bool ok=circular_polish::solve(M,rhs,q,upper,d,tol,s);svds=s.polish_svd_calls;steps=s.polish_steps;return ok;};
  bool found=terminal_component_polish::solve(A,b,candidate,lo,hi,dep,tolerance,trial,gate,iteration,polish);
  auto&s=stats.terminal_polish;s.components+=trial.components;s.largest_rows=std::max(s.largest_rows,trial.largest_rows);s.iteration_steps+=trial.iteration_steps;s.svd_calls+=trial.svd_calls;s.polish_steps+=trial.polish_steps;
  CoulombStats final;
  if(found&&coulombIterate(A,b,candidate,lo,hi,dep,0,tolerance,final)){
   x=candidate;stats.solves++;stats.terminal_polish_solves++;stats.last_residual=final.last_residual;stats.residual_max=std::max(stats.residual_max,final.residual_max);stats.passive_change_max=std::max(stats.passive_change_max,final.passive_change_max);return true;
  }
  stats.terminal_polish_declines++;
 }

#endif
 if(rejected_impulses)*rejected_impulses=rejected;
 return false;
}

#include "shared_contact.h"

class CoulombMLCP : public RecordedMLCP {
protected:
 btScalar solveGroupCacheFriendlyIterations(btCollisionObject** bodies,int count,
  btPersistentManifold** manifolds,int manifold_count,btTypedConstraint** constraints,
  int constraint_count,const btContactSolverInfo& info,btIDebugDraw* debug) override {
  auto result=RecordedMLCP::solveGroupCacheFriendlyIterations(bodies,count,manifolds,
      manifold_count,constraints,constraint_count,info,debug);
  if(translation_split&&info.m_splitImpulse){
   translation_pose_ledger_updates++;
   for(int i=0;i<m_tmpSolverBodyPool.size();i++){
    auto change=translationPoseChange(m_tmpSolverBodyPool[i],info.m_timeStep);
    translation_pose_displacement_max_m=std::max(translation_pose_displacement_max_m,static_cast<double>(change.displacement.length()));
    translation_pose_potential_change_J+=change.potential_energy;
    translation_pose_absolute_potential_change_J+=std::abs(change.potential_energy);
    translation_pose_orbital_change+=change.orbital_momentum;
    translation_pose_absolute_orbital_change+=change.orbital_momentum.length();
   }
   clearPositionTurns(m_tmpSolverBodyPool);
  }
  return result;
 }
 void transportContactRows(){
  if(!shared_contact_point)return;
  for(int i=0;i<m_allConstraintPtrArray.size();i++){
   auto& row=*m_allConstraintPtrArray[i];int dep=m_limitDependencies[i];
   auto* cp=static_cast<btManifoldPoint*>(m_allConstraintPtrArray[dep>=0?dep:i]->m_originalContactPoint);
   if(!cp)throw std::runtime_error("Shared contact row has no manifold point");
   double moved=transportSharedContactRow(row,m_tmpSolverBodyPool[row.m_solverBodyIdA],m_tmpSolverBodyPool[row.m_solverBodyIdB],*cp,dep>=0);
   shared_point_transport_max_m=std::max(shared_point_transport_max_m,moved);shared_point_rows++;
  }
 }
 void createMLCPFast(const btContactSolverInfo& info) override {transportContactRows();RecordedMLCP::createMLCPFast(info);}
 void createMLCP(const btContactSolverInfo& info) override {transportContactRows();RecordedMLCP::createMLCP(info);}
 bool solveMLCP(const btContactSolverInfo& info) override {
  if(!m_A.rows())return true;
  for(int i=0;i<m_b.rows();i++)if(m_limitDependencies[i]>=0){
   const auto& c=*m_allConstraintPtrArray[i];
   double corrected=consistentTangentRHS(m_b[i],c,m_tmpSolverBodyPool[c.m_solverBodyIdA],m_tmpSolverBodyPool[c.m_solverBodyIdB]);
   gyro_correction_max=std::max(gyro_correction_max,std::abs(corrected-m_b[i]));m_b[i]=corrected;
  }
  // Touching within the declared geometric tolerance is treated as touching,
  // avoiding inconsistent gap/h targets on nearly redundant face points.
  for(int i=0;i<m_b.rows();i++)if(m_limitDependencies[i]<0){
   auto* cp=static_cast<btManifoldPoint*>(m_allConstraintPtrArray[i]->m_originalContactPoint);
   if(cp&&cp->getDistance()>0&&cp->getDistance()<=contact_slop_m)m_b[i]+=cp->getDistance()/info.m_timeStep;
   if(cp&&std::abs(cp->getDistance())<=contact_slop_m)m_bSplit[i]=0;
   if(translation_clearance&&cp)m_bSplit[i]=translationGapTarget(cp->getDistance(),info.m_timeStep,contact_slop_m,m_bSplit[i]);
  }
  std::vector<double> rejected;
  if(!coulombSolve(m_A,m_b,m_x,m_lo,m_hi,m_limitDependencies,info.m_numIterations,tolerance,stats,rejection_observer?&rejected:nullptr,recovery_enabled,early_component_recovery))
   {if(rejection_observer)rejection_observer(m_A,m_b,rejected,m_lo,m_hi,m_limitDependencies,"velocity",stats.last_residual,info.m_timeStep);std::ostringstream message;message<<"Coulomb residual gate failed ("<<stats.last_residual<<" m/s): increase iterations or refine timestep; no friction-law fallback";throw std::runtime_error(message.str());}
  if(info.m_splitImpulse){
   std::vector<int> normals;for(int i=0;i<m_b.rows();i++)if(m_limitDependencies[i]<0)normals.push_back(i);
   int k=static_cast<int>(normals.size());btMatrixXu A(k,k);btVectorXu b(k),upper(k),x(k);
   for(int i=0;i<k;i++){b[i]=m_bSplit[normals[i]];upper[i]=m_hi[normals[i]];x[i]=0;for(int j=0;j<k;j++)A.setElem(i,j,m_A(normals[i],normals[j]));}
   if(translation_combined){
    std::vector<double> distances;for(int row:normals){
     auto* cp=static_cast<btManifoldPoint*>(m_allConstraintPtrArray[row]->m_originalContactPoint);
     if(!cp)throw std::runtime_error("Combined position repair requires contact geometry");
     distances.push_back(cp->getDistance());
    }
    if(!combined_translation_review::acceptedPhysicalBodies(m_allConstraintPtrArray,m_tmpSolverBodyPool,m_x,translation_final_physical_bodies)||
       !combined_translation_review::targets(m_allConstraintPtrArray,normals,distances,translation_final_physical_bodies,m_bSplit,info.m_timeStep,contact_slop_m,b,&translation_physical_rates,&translation_desired_rates))
     throw std::runtime_error("Nonfinite accepted physical motion for combined position repair");
    translation_accepted_physical_impulses=m_x;
    for(int i=0;i<k;i++)m_bSplit[normals[i]]=b[i];
   }
   if(translation_split)A=assembleTranslationSplitMobility(m_allConstraintPtrArray,m_tmpSolverBodyPool,normals);
   m_xSplit.setZero();
   if(translation_split){
    double residual=0;
    if(!translationSplitSolve(A,b,upper,x,tolerance,info.m_numIterations,&residual,&position_stats.null_pressure,recovery_enabled)){
     if(position_geometry_observer)position_geometry_observer(m_allConstraintPtrArray,m_tmpSolverBodyPool,normals,m_bSplit,info.m_timeStep);
     if(rejection_observer){
      btVectorXu lower(k);btAlignedObjectArray<int> independent;independent.resize(k);std::vector<double> rejected_position(k);
      for(int i=0;i<k;i++){lower[i]=0;independent[i]=-1;rejected_position[i]=x[i];}
      rejection_observer(A,b,rejected_position,lower,upper,independent,"position_translation",residual,info.m_timeStep);
     }
     throw std::runtime_error("Translation-only position projection failed; repair initial overlap or refine timestep");
    }
    translation_split_solves++;translation_split_residual_max=std::max(translation_split_residual_max,residual);
    for(int i=0;i<k;i++)m_xSplit[normals[i]]=x[i];
   }
   else if(normalQP(A,b,upper,x)){for(int i=0;i<k;i++)m_xSplit[normals[i]]=x[i];}
   else{
    auto lower=m_lo,upper_full=m_hi;for(int i=0;i<m_b.rows();i++)if(m_limitDependencies[i]>=0)lower[i]=upper_full[i]=0;
    if(!coulombSolve(m_A,m_bSplit,m_xSplit,lower,upper_full,m_limitDependencies,info.m_numIterations,tolerance,position_stats,rejection_observer?&rejected:nullptr,recovery_enabled)){
     if(rejection_observer)rejection_observer(m_A,m_bSplit,rejected,lower,upper_full,m_limitDependencies,"position",position_stats.last_residual,info.m_timeStep);
     throw std::runtime_error("Normal-only position projection residual failed; repair initial overlap or refine timestep");
    }
   }
  }
  return true;
 }
public:
 // Observation only: exact failed-step geometry goes to a separate companion.
 std::function<void(const btAlignedObjectArray<btSolverConstraint*>&,
  const btAlignedObjectArray<btSolverBody>&,const std::vector<int>&,
  const btVectorXu&,double)> position_geometry_observer;
 // Opt-in diagnostic only. Rejection remains an error; never reuse a rejected iterate.
 std::function<void(const btMatrixXu&,const btVectorXu&,const std::vector<double>&,const btVectorXu&,const btVectorXu&,const btAlignedObjectArray<int>&,const char*,double,double)> rejection_observer;
 bool early_component_recovery=false; // Explicit research opt-in; velocity solve only.
 bool recovery_enabled=true,shared_contact_point=true;
 bool translation_clearance=false,translation_combined=false;
 btAlignedObjectArray<btSolverBody> translation_final_physical_bodies;
 btVectorXu translation_physical_rates,translation_desired_rates,translation_accepted_physical_impulses;
 unsigned long long translation_pose_ledger_updates=0;
 double translation_pose_displacement_max_m=0,translation_pose_potential_change_J=0,translation_pose_absolute_potential_change_J=0,translation_pose_absolute_orbital_change=0;
 btVector3 translation_pose_orbital_change{0,0,0};
 bool translation_split=false;int translation_split_solves=0;double translation_split_residual_max=0;
 unsigned long long shared_point_rows=0;double shared_point_transport_max_m=0;
 double tolerance=1e-8,contact_slop_m=1e-9,gyro_correction_max=0;CoulombStats stats,position_stats;
 explicit CoulombMLCP(btMLCPSolverInterface* solver):RecordedMLCP(solver){}
};
