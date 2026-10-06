#pragma once
#include <algorithm>
#include <cmath>
#include <cstdlib>
// Numerical local face search; no candidate is applied before all original
// unilateral/friction/maximum-dissipation equations pass independently.
static bool physicsJointContact(b2ContactVelocityConstraint* vc,b2Vec2& vA,double& wA,b2Vec2& vB,double& wB) {
  static const bool enabled=std::getenv("PHYSICS_DISABLE_JOINT")==nullptr;
  if(!enabled)return false;
  const int points=vc->pointCount,n=2*points;
  if(points<1 || points>2)return false;
  const double tol=1e-10,mu=vc->friction;
  b2Vec2 normal=vc->normal,tangent=b2Cross(normal,1.0),dirs[4];
  double armsA[4],armsB[4],M[4][4]{},rhs[4]{},warm[4]{},best[4]{};
  for(int i=0;i<n;i++) {
    auto& cp=vc->points[i/2];dirs[i]=i%2?tangent:normal;
    armsA[i]=b2Cross(cp.rA,dirs[i]);armsB[i]=b2Cross(cp.rB,dirs[i]);
    warm[i]=i%2?cp.tangentImpulse:cp.normalImpulse;
    auto dv=vB+b2Cross(wB,cp.rB)-vA-b2Cross(wA,cp.rA);
    rhs[i]=-b2Dot(dv,dirs[i])+(i%2?vc->tangentSpeed:cp.velocityBias);
  }
  for(int i=0;i<n;i++)for(int j=0;j<n;j++) {
    M[i][j]=(vc->invMassA+vc->invMassB)*b2Dot(dirs[i],dirs[j])+vc->invIA*armsA[i]*armsA[j]+vc->invIB*armsB[i]*armsB[j];
    rhs[i]+=M[i][j]*warm[j];
  }
  bool found=false;double bestScore=1e300;
  for(int face=0;face<(points==1?4:16);face++) {
    int modes[2]{face%4,face/4},rows[4]{},k=0;
    double C[4][4]{},seed[4]{},B[4][4]{},r[4]{},delta[4]{},q[4]{};
    for(int point=0;point<points;point++) {
      int mode=modes[point],a=2*point;
      if(!mode)continue;
      rows[k]=a;C[a][k]=1;
      if(mode==1)seed[k]=warm[a];
      else {double s=mode==2?-mu:mu;C[a+1][k]=s;seed[k]=(warm[a]+s*warm[a+1])/(1+s*s);}
      k++;
      if(mode==1){rows[k]=a+1;C[a+1][k]=1;seed[k]=warm[a+1];k++;}
    }
    double scale=0;
    for(int i=0;i<k;i++) {
      r[i]=rhs[rows[i]];
      for(int j=0;j<k;j++) {
        for(int at=0;at<n;at++)B[i][j]+=M[rows[i]][at]*C[at][j];
        r[i]-=B[i][j]*seed[j];scale=std::max(scale,std::abs(B[i][j]));
      }
    }
    int columns[4]{0,1,2,3},rank=0;
    for(int at=0;at<k;at++) {
      int pi=at,pj=at;double pivot=0;
      for(int i=at;i<k;i++)for(int j=at;j<k;j++)if(std::abs(B[i][j])>pivot){pivot=std::abs(B[i][j]);pi=i;pj=j;}
      if(pivot<=1e-12*std::max(scale,1e-300))break;
      for(int j=0;j<k;j++)std::swap(B[at][j],B[pi][j]);std::swap(r[at],r[pi]);
      for(int i=0;i<k;i++)std::swap(B[i][at],B[i][pj]);std::swap(columns[at],columns[pj]);
      for(int i=at+1;i<k;i++){double factor=B[i][at]/B[at][at];for(int j=at;j<k;j++)B[i][j]-=factor*B[at][j];r[i]-=factor*r[at];}
      rank++;
    }
    bool consistent=true;for(int i=rank;i<k;i++)if(std::abs(r[i])>tol)consistent=false;
    if(!consistent)continue;
    double permuted[4]{};
    for(int i=rank-1;i>=0;i--){double value=r[i];for(int j=i+1;j<k;j++)value-=B[i][j]*permuted[j];permuted[i]=value/B[i][i];}
    for(int i=0;i<k;i++)delta[columns[i]]=permuted[i];
    for(int i=0;i<n;i++)for(int j=0;j<k;j++)q[i]+=C[i][j]*(seed[j]+delta[j]);
    // The same cone cleanup as an actual accumulated impulse, then recheck.
    for(int point=0;point<points;point++) {
      int a=2*point;q[a]=std::max(0.,q[a]);q[a+1]=std::max(-mu*q[a],std::min(q[a+1],mu*q[a]));
    }
    double w[4]{};bool accepted=true;
    for(int i=0;i<n;i++){w[i]=-rhs[i];for(int j=0;j<n;j++)w[i]+=M[i][j]*q[j];if(!std::isfinite(q[i]) || !std::isfinite(w[i]))accepted=false;}
    for(int point=0;point<points;point++) {
      int a=2*point,mode=modes[point];
      if(w[a]<-tol || (q[a]>0 && std::abs(w[a])>tol))accepted=false;
      if(mode==0 && (q[a]!=0 || q[a+1]!=0))accepted=false;
      if(mode==1 && std::abs(w[a+1])>tol)accepted=false;
      if(mode==2 && w[a+1]<-tol)accepted=false;
      if(mode==3 && w[a+1]>tol)accepted=false;
      if(std::abs(q[a+1])>mu*q[a] || q[a+1]*w[a+1]>tol*std::abs(q[a+1]))accepted=false;
    }
    if(!accepted)continue;
    double score=0;for(int i=0;i<n;i++)score+=(q[i]-warm[i])*(q[i]-warm[i]);
    if(!found || score<bestScore){found=true;bestScore=score;std::copy(q,q+n,best);}
  }
  if(!found)return false;
  for(int point=0;point<points;point++) {
    auto& cp=vc->points[point];int a=2*point;
    b2Vec2 P=(best[a]-cp.normalImpulse)*normal+(best[a+1]-cp.tangentImpulse)*tangent;
    vA-=vc->invMassA*P;wA-=vc->invIA*b2Cross(cp.rA,P);
    vB+=vc->invMassB*P;wB+=vc->invIB*b2Cross(cp.rB,P);
    cp.normalImpulse=best[a];cp.tangentImpulse=best[a+1];
  }
  return true;
}
