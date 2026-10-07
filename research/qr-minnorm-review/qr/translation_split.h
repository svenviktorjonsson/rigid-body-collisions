// Numerical position projection only. Physical velocity rows stay unchanged.
#pragma once
#include <BulletDynamics/Dynamics/btRigidBody.h>
#include <BulletDynamics/ConstraintSolver/btSolverBody.h>
#include <BulletDynamics/ConstraintSolver/btSolverConstraint.h>
#include <LinearMath/btMatrixX.h>
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "normal_qp.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

inline btScalar translationRowMobility(const btSolverConstraint& a,
 const btSolverConstraint& b,const btAlignedObjectArray<btSolverBody>& bodies){
 btScalar value=0;
 const int ai[2]={a.m_solverBodyIdA,a.m_solverBodyIdB},bi[2]={b.m_solverBodyIdA,b.m_solverBodyIdB};
 const btVector3 an[2]={a.m_contactNormal1,a.m_contactNormal2},bn[2]={b.m_contactNormal1,b.m_contactNormal2};
 for(int i=0;i<2;i++)for(int j=0;j<2;j++)if(ai[i]==bi[j]){
  const auto& body=bodies[ai[i]];
  if(body.m_originalBody)value+=body.m_originalBody->getInvMass()*an[i].dot(bn[j]);
 }
 return value;
}

inline void clearPositionTurns(btAlignedObjectArray<btSolverBody>& bodies){
 for(int i=0;i<bodies.size();i++)bodies[i].internalGetTurnVelocity().setZero();
}

inline btMatrixXu assembleTranslationSplitMobility(
 const btAlignedObjectArray<btSolverConstraint*>& rows,
 const btAlignedObjectArray<btSolverBody>& bodies,const std::vector<int>& normals){
 const int n=static_cast<int>(normals.size());btMatrixXu A(n,n);
 for(int i=0;i<n;i++)for(int j=0;j<n;j++)A.setElem(i,j,
     translationRowMobility(*rows[normals[i]],*rows[normals[j]],bodies));
 return A;
}

// Normal complementarity on the linear-only Gram matrix. A pressure gauge is
// allowed; no compliance, eigenvalue shift or angular pose repair is introduced.
// Every returned impulse passes the caller's ABSOLUTE velocity residual gate.
inline bool translationSplitSolve(const btMatrixXu& A,const btVectorXu& b,
 const btVectorXu& upper,btVectorXu& x,double tolerance,int budget,double* final_residual=nullptr){
 const int n=b.rows();if(A.rows()!=n||A.cols()!=n||upper.rows()!=n||x.rows()!=n||
     !(tolerance>0)||!std::isfinite(tolerance)||budget<0)return false;
 auto gate=[&](){
  double error=0;
  for(int i=0;i<n;i++){
   if(!(A(i,i)>0)||!std::isfinite(A(i,i))||!std::isfinite(x[i])||x[i]<0||x[i]>upper[i])return false;
   double w=-b[i];for(int j=0;j<n;j++)w+=A(i,j)*x[j];
   if(!std::isfinite(w))return false;
   error=std::max(error,std::abs(x[i]-std::max(0.,static_cast<double>(x[i])-w/A(i,i)))*A(i,i));
  }
  if(final_residual)*final_residual=error;
  return error<=tolerance;
 };
 if(final_residual)*final_residual=std::numeric_limits<double>::infinity();
 x.setZero();
 if(normalQP(A,b,upper,x)&&gate())return true;
 // Cholesky can reject a singular face even when its pressure redistribution
 // is harmless. Projected iterations operate on the unchanged PSD mobility.
 for(int i=0;i<n;i++)if(!(A(i,i)>0)||!std::isfinite(A(i,i)))return false;
 for(int sweep=0;sweep<=budget;sweep++){
  if((sweep%8==0||sweep==budget)&&gate())return true;
  if(sweep==budget)break;
  for(int i=0;i<n;i++){
   double w=-b[i];for(int j=0;j<n;j++)w+=A(i,j)*x[j];
   x[i]=std::max(0.,static_cast<double>(x[i])-w/A(i,i));
  }
 }
 return false;
}
