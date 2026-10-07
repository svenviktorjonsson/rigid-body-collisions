#pragma once
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "coulomb.h"
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <vector>
static long long physicsGlobalCalls=0,physicsGlobalSolves=0,physicsGlobalDeclines=0;
static double physicsGlobalResidual=0,physicsGlobalBodyResidual=0;
extern "C" long long b2PhysicsGlobalCalls(){return physicsGlobalCalls;}
extern "C" long long b2PhysicsGlobalSolves(){return physicsGlobalSolves;}
extern "C" long long b2PhysicsGlobalDeclines(){return physicsGlobalDeclines;}
extern "C" double b2PhysicsGlobalResidual(){return physicsGlobalResidual;}
extern "C" double b2PhysicsGlobalBodyResidual(){return physicsGlobalBodyResidual;}
static bool physicsRestitutionConfigured=false;
static double physicsNormalRestitution=0,physicsTangentialRestitution=0;
extern "C" void b2PhysicsSetRestitution(double normal,double tangent){
 if(!std::isfinite(normal)||normal<0||normal>1||!std::isfinite(tangent)||tangent< -1||tangent>1)throw std::runtime_error("Invalid normal/tangential restitution");
 physicsRestitutionConfigured=true;physicsNormalRestitution=normal;physicsTangentialRestitution=tangent;
}
struct PhysicsPlanarPoint { b2ContactVelocityConstraint* vc; b2VelocityConstraintPoint* cp; };
static void physicsGlobalCapture(const btMatrixXu&A,const btVectorXu&b,const btVectorXu&p,
 const btVectorXu&lo,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double residual,const char*reason) {
 const char* path=std::getenv("PHYSICS_GLOBAL_REJECTION");if(!path)return;
 std::ofstream o(path);o<<std::setprecision(17)<<"{\"schema\":\"circular-coulomb-rejection-v1\",\"phase\":\"planar_embedded_velocity\",\"reason\":\""<<reason<<"\",\"tolerance_m_s\":1e-10,\"residual_m_s\":"<<residual<<",\"iteration_budget\":4096,\"A\":[";
 int n=b.rows();for(int i=0;i<n;i++){if(i)o<<',';o<<'[';for(int j=0;j<n;j++){if(j)o<<',';o<<A(i,j);}o<<']';}o<<']';
 const btVectorXu* vectors[]={&b,&p,&lo,&hi};const char*names[]={"b","p","lo","hi"};
 for(int at=0;at<4;at++){o<<",\""<<names[at]<<"\":[";for(int i=0;i<n;i++){if(i)o<<',';o<<(*vectors[at])[i];}o<<']';}
 o<<",\"dependencies\":[";for(int i=0;i<n;i++){if(i)o<<',';o<<dep[i];}o<<"],\"virtual_tangent_convention\":\"third row per point has unit diagonal, zero rhs and final impulse, no physical body response\"}\n";
}
static bool physicsGlobalContact(b2ContactVelocityConstraint*constraints,int count,b2Velocity*velocities) {
 static const bool enabled=std::getenv("PHYSICS_DISABLE_GLOBAL")==nullptr;
 if(!enabled){if(physicsRestitutionConfigured)throw std::runtime_error("Explicit restitution cannot bypass the simultaneous solver");return false;}
 physicsGlobalCalls++;
 std::vector<PhysicsPlanarPoint>points;int bodyCount=0;
 for(int i=0;i<count;i++) {
  auto&vc=constraints[i];bodyCount=std::max(bodyCount,std::max(vc.indexA,vc.indexB)+1);
  for(int j=0;j<vc.pointCount;j++)points.push_back({&vc,&vc.points[j]});
 }
 if(points.empty())return true;
 const int n=3*points.size();const double tol=1e-10;
 if(n>4096)throw std::runtime_error("Planar global contact row cap exceeded");
 btMatrixXu A(n,n);btVectorXu b(n),p(n),old(n),lo(n),hi(n);btAlignedObjectArray<int>dep;dep.resize(n);
 std::vector<double>massInv(bodyCount),inertiaInv(bodyCount);
 auto direction=[](const PhysicsPlanarPoint&point,int row){return row==0?point.vc->normal:b2Cross(point.vc->normal,1.0);};
 for(int i=0;i<n;i++)for(int j=0;j<n;j++)A.setElem(i,j,0);
 for(size_t at=0;at<points.size();at++) {
  auto&point=points[at];auto&vc=*point.vc;auto&cp=*point.cp;int k=3*at;
  massInv[vc.indexA]=vc.invMassA;inertiaInv[vc.indexA]=vc.invIA;massInv[vc.indexB]=vc.invMassB;inertiaInv[vc.indexB]=vc.invIB;
  auto dv=velocities[vc.indexB].v+b2Cross(velocities[vc.indexB].w,cp.rB)-velocities[vc.indexA].v-b2Cross(velocities[vc.indexA].w,cp.rA);
  b[k]=cp.velocityBias-b2Dot(dv,vc.normal);b[k+1]=vc.tangentSpeed-b2Dot(dv,b2Cross(vc.normal,1.0));b[k+2]=0;
  old[k]=p[k]=cp.normalImpulse;old[k+1]=p[k+1]=cp.tangentImpulse;old[k+2]=p[k+2]=0;
  dep[k]=-1;lo[k]=0;hi[k]=1e30;dep[k+1]=dep[k+2]=k;lo[k+1]=lo[k+2]=-vc.friction;hi[k+1]=hi[k+2]=vc.friction;A.setElem(k+2,k+2,1);
 }
 for(size_t ai=0;ai<points.size();ai++)for(size_t bi=0;bi<points.size();bi++) {
  auto&a=points[ai];auto&bb=points[bi];auto&va=*a.vc;auto&vb=*bb.vc;
  int idsA[2]={va.indexA,va.indexB},idsB[2]={vb.indexA,vb.indexB};b2Vec2 armsA[2]={a.cp->rA,a.cp->rB},armsB[2]={bb.cp->rA,bb.cp->rB};
  for(int i=0;i<2;i++)for(int j=0;j<2;j++) {
   b2Vec2 di=direction(a,i),dj=direction(bb,j);double value=0;
   for(int u=0;u<2;u++)for(int v=0;v<2;v++)if(idsA[u]==idsB[v]) {
    int body=idsA[u];double sign=u==v?1:-1;
    value+=sign*(massInv[body]*b2Dot(di,dj)+inertiaInv[body]*b2Cross(armsA[u],di)*b2Cross(armsB[v],dj));
   }
   A.setElem(3*ai+i,3*bi+j,value);
  }
 }
 for(int i=0;i<n;i++)for(int j=0;j<n;j++)b[i]+=A(i,j)*old[j];
 std::vector<double> normalTarget(points.size()),tangentTarget(points.size());
 for(size_t at=0;at<points.size();at++){
  const int k=3*at;const auto& vc=*points[at].vc;const auto& cp=*points[at].cp;
  normalTarget[at]=cp.velocityBias;tangentTarget[at]=vc.tangentSpeed;
  const double un=cp.velocityBias-b[k],ut=vc.tangentSpeed-b[k+1];
  if(physicsRestitutionConfigured&&un< -tol){
   normalTarget[at]=-physicsNormalRestitution*un;
   tangentTarget[at]=-physicsTangentialRestitution*ut;
   b[k]=normalTarget[at]-un;b[k+1]=tangentTarget[at]-ut;
  }
 }
 CoulombStats stats;std::vector<double>rejected;
 bool found=coulombSolve(A,b,p,lo,hi,dep,4096,.25*tol,stats,&rejected,true,true);
 if(!found) {
  if(rejected.size()==static_cast<size_t>(n))for(int i=0;i<n;i++)p[i]=rejected[i];
  physicsGlobalDeclines++;physicsGlobalCapture(A,b,p,lo,hi,dep,stats.last_residual,"native_original_law_decline");
  throw std::runtime_error("Planar global original contact-law gate declined");
 }
 for(size_t at=0;at<points.size();at++)p[3*at+2]=0;
 CoulombStats gateStats;
 bool gated=coulombIterate(A,b,p,lo,hi,dep,0,tol,gateStats);
 double residual=gateStats.last_residual;
 if(!gated) {
  physicsGlobalDeclines++;physicsGlobalCapture(A,b,p,lo,hi,dep,residual,"zero_virtual_original_gate_decline");
  throw std::runtime_error("Planar zero-virtual original contact-law gate declined");
 }
 std::vector<b2Velocity>candidate(velocities,velocities+bodyCount),free(velocities,velocities+bodyCount);
 double wallWork=0;
 for(size_t at=0;at<points.size();at++) {
  auto&point=points[at];auto&vc=*point.vc;auto&cp=*point.cp;int k=3*at;b2Vec2 tangent=b2Cross(vc.normal,1.0);
  b2Vec2 delta=(p[k]-old[k])*vc.normal+(p[k+1]-old[k+1])*tangent,previous=old[k]*vc.normal+old[k+1]*tangent,total=p[k]*vc.normal+p[k+1]*tangent;
  candidate[vc.indexA].v-=vc.invMassA*delta;candidate[vc.indexA].w-=vc.invIA*b2Cross(cp.rA,delta);
  candidate[vc.indexB].v+=vc.invMassB*delta;candidate[vc.indexB].w+=vc.invIB*b2Cross(cp.rB,delta);
  free[vc.indexA].v+=vc.invMassA*previous;free[vc.indexA].w+=vc.invIA*b2Cross(cp.rA,previous);
  free[vc.indexB].v-=vc.invMassB*previous;free[vc.indexB].w-=vc.invIB*b2Cross(cp.rB,previous);
  if(vc.invMassA==0 && vc.invMassB>0)wallWork+=b2Dot(total,velocities[vc.indexA].v+b2Cross(velocities[vc.indexA].w,cp.rA));
  if(vc.invMassB==0 && vc.invMassA>0)wallWork-=b2Dot(total,velocities[vc.indexB].v+b2Cross(velocities[vc.indexB].w,cp.rB));
 }
 double bodyResidual=0;bool finite=true;
 for(size_t at=0;at<points.size();at++) {
  auto&point=points[at];auto&vc=*point.vc;auto&cp=*point.cp;int k=3*at;
  auto dv=candidate[vc.indexB].v+b2Cross(candidate[vc.indexB].w,cp.rB)-candidate[vc.indexA].v-b2Cross(candidate[vc.indexA].w,cp.rA);
  double wn=b2Dot(dv,vc.normal)-normalTarget[at],wt=b2Dot(dv,b2Cross(vc.normal,1.0))-tangentTarget[at];
  double cap=vc.friction*p[k],zn=p[k]-wn/A(k,k),zt=p[k+1]-wt/A(k+1,k+1);
  double rn=std::abs(p[k]-std::max(0.,zn))*A(k,k),rt=std::abs(p[k+1]-std::max(-cap,std::min(zt,cap)))*A(k+1,k+1);
  finite &= std::isfinite(wn)&&std::isfinite(wt)&&std::isfinite(rn)&&std::isfinite(rt);bodyResidual=std::max(bodyResidual,std::max(rn,rt));
 }
 double before=0,after=0;
 for(int i=0;i<bodyCount;i++)if(massInv[i]>0) {
  before+=.5*b2Dot(free[i].v,free[i].v)/massInv[i];after+=.5*b2Dot(candidate[i].v,candidate[i].v)/massInv[i];
  if(inertiaInv[i]>0){before+=.5*free[i].w*free[i].w/inertiaInv[i];after+=.5*candidate[i].w*candidate[i].w/inertiaInv[i];}
 }
 if(!finite || !std::isfinite(before+after+wallWork) || bodyResidual>tol || after-before-wallWork>tol*(1+before+std::abs(wallWork))) {
  physicsGlobalDeclines++;physicsGlobalCapture(A,b,p,lo,hi,dep,bodyResidual,"actual_body_velocity_or_passivity_decline");
  throw std::runtime_error("Planar actual-body contact/passivity gate declined");
 }
 // Atomic commit of the independently checked velocities and actual impulses.
 for(int i=0;i<bodyCount;i++)velocities[i]=candidate[i];
 for(size_t at=0;at<points.size();at++){points[at].cp->normalImpulse=p[3*at];points[at].cp->tangentImpulse=p[3*at+1];}
 physicsGlobalSolves++;physicsGlobalResidual=std::max(physicsGlobalResidual,residual);physicsGlobalBodyResidual=std::max(physicsGlobalBodyResidual,bodyResidual);
 return true;
}
