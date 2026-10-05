// Project adapter: circular, non-associated Coulomb friction. Bullet is unmodified.
#pragma once
#include "normal_qp.h"
#include <stdexcept>
#include <sstream>
#include <functional>
#include <limits>

// Upstream friction RHS omits this angular free-velocity increment, although
// normal RHS and final body writeback include it. Preserve the signed B row.
inline double consistentTangentRHS(double rhs,const btSolverConstraint& c,
 const btSolverBody& a,const btSolverBody& b){
 return rhs-c.m_relpos1CrossNormal.dot(a.m_externalTorqueImpulse)
           -c.m_relpos2CrossNormal.dot(b.m_externalTorqueImpulse);
}

struct CoulombStats {
 int solves=0,sweeps_max=0,fast_solves=0,newton_steps=0,polish_solves=0,polish_steps=0,gauge_restarts=0,cold_restarts=0,polish_svd_calls=0,polish_budget_rejections=0,polish_svd_rejections=0,rank_restarts=0,opposing_restarts=0;
 double residual_max=0,passive_change_max=0,last_residual=0;
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

inline bool coulombSolve(const btMatrixXu& A,const btVectorXu& b,btVectorXu& x,
 const btVectorXu& lo,const btVectorXu& hi,const btAlignedObjectArray<int>& dep,
 int budget,double tolerance,CoulombStats& stats,std::vector<double>* rejected_impulses=nullptr,bool allow_recovery=true){
 std::vector<double> rejected;
 if(coulombIterate(A,b,x,lo,hi,dep,budget,tolerance,stats,&rejected))return true;
 // Recovery has a separately bounded numerical budget, and the original gate.
 // Preserve the warm rejected iterate only as a starting guess, never as output.
 btVectorXu candidate=x;for(int i=0;i<b.rows();i++)candidate[i]=rejected[i];
 if(allow_recovery&&budget>=64&&circular_polish::solve(A,b,candidate,hi,dep,tolerance,stats)){x=candidate;stats.sweeps_max=std::max(stats.sweeps_max,budget);return true;}
 if(rejected_impulses)*rejected_impulses=rejected;
 return false;
}

#include "shared_contact.h"

class CoulombMLCP : public RecordedMLCP {
protected:
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
  }
  std::vector<double> rejected;
  if(!coulombSolve(m_A,m_b,m_x,m_lo,m_hi,m_limitDependencies,info.m_numIterations,tolerance,stats,rejection_observer?&rejected:nullptr,recovery_enabled))
   {if(rejection_observer)rejection_observer(m_A,m_b,rejected,m_lo,m_hi,m_limitDependencies,"velocity",stats.last_residual,info.m_timeStep);std::ostringstream message;message<<"Coulomb residual gate failed ("<<stats.last_residual<<" m/s): increase iterations or refine timestep; no friction-law fallback";throw std::runtime_error(message.str());}
  if(info.m_splitImpulse){
   std::vector<int> normals;for(int i=0;i<m_b.rows();i++)if(m_limitDependencies[i]<0)normals.push_back(i);
   int k=static_cast<int>(normals.size());btMatrixXu A(k,k);btVectorXu b(k),upper(k),x(k);
   for(int i=0;i<k;i++){b[i]=m_bSplit[normals[i]];upper[i]=m_hi[normals[i]];x[i]=0;for(int j=0;j<k;j++)A.setElem(i,j,m_A(normals[i],normals[j]));}
   m_xSplit.setZero();
   if(normalQP(A,b,upper,x)){for(int i=0;i<k;i++)m_xSplit[normals[i]]=x[i];}
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
 // Opt-in diagnostic only. Rejection remains an error; never reuse a rejected iterate.
 std::function<void(const btMatrixXu&,const btVectorXu&,const std::vector<double>&,const btVectorXu&,const btVectorXu&,const btAlignedObjectArray<int>&,const char*,double,double)> rejection_observer;
 bool recovery_enabled=true,shared_contact_point=true;
 unsigned long long shared_point_rows=0;double shared_point_transport_max_m=0;
 double tolerance=1e-8,contact_slop_m=1e-9,gyro_correction_max=0;CoulombStats stats,position_stats;
 explicit CoulombMLCP(btMLCPSolverInterface* solver):RecordedMLCP(solver){}
};
