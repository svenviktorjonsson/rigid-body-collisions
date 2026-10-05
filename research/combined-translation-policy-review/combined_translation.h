// Research proposal only. Not included in the frozen source108 world solver.
#pragma once
#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/ConstraintSolver/btSolverBody.h>
#include <BulletDynamics/ConstraintSolver/btSolverConstraint.h>
#include <LinearMath/btMatrixX.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace combined_translation_review {
inline bool finiteVector(const btVector3& v){
 for(int k=0;k<3;k++)if(!std::isfinite(static_cast<double>(v[k])))return false;
 return true;
}
inline bool finitePhysicalBody(const btSolverBody& b){
 if(!b.m_originalBody)return true;
 return finiteVector(b.m_linearVelocity)&&finiteVector(b.m_angularVelocity)&&
  finiteVector(b.m_deltaLinearVelocity)&&finiteVector(b.m_deltaAngularVelocity)&&
  finiteVector(b.m_externalForceImpulse)&&finiteVector(b.m_externalTorqueImpulse)&&
  finiteVector(b.m_linearFactor)&&finiteVector(b.m_angularFactor)&&finiteVector(b.m_invMass);
}

// Retain all current warm deltas. Match upstream btMLCPSolver.cpp:596-600,
// including row order, signed B axes, angular factors and external gyro.
// Applying final x directly would count warm-start contributions twice.
// No original solver body, row, manifold or rigid-body state is mutated.
inline bool acceptedPhysicalBodies(
 const btAlignedObjectArray<btSolverConstraint*>& rows,
 const btAlignedObjectArray<btSolverBody>& bodies,const btVectorXu& accepted,
 btAlignedObjectArray<btSolverBody>& output){
 if(&output==&bodies||accepted.rows()!=rows.size())return false;
 btAlignedObjectArray<btSolverBody> trial=bodies;
 for(int i=0;i<trial.size();i++)if(!finitePhysicalBody(trial[i]))return false;
 for(int i=0;i<rows.size();i++){
  if(!rows[i])return false;
  const auto& c=*rows[i];const int a=c.m_solverBodyIdA,b=c.m_solverBodyIdB;
  if(a<0||b<0||a>=trial.size()||b>=trial.size()||
     !std::isfinite(static_cast<double>(accepted[i]))||
     !std::isfinite(static_cast<double>(c.m_appliedImpulse))||
     !finiteVector(c.m_contactNormal1)||!finiteVector(c.m_contactNormal2)||
     !finiteVector(c.m_angularComponentA)||!finiteVector(c.m_angularComponentB))return false;
  const btScalar delta=accepted[i]-c.m_appliedImpulse;
  if(!std::isfinite(static_cast<double>(delta)))return false;
  auto& A=trial[a];auto& B=trial[b];
  A.internalApplyImpulse(c.m_contactNormal1*A.internalGetInvMass(),c.m_angularComponentA,delta);
  B.internalApplyImpulse(c.m_contactNormal2*B.internalGetInvMass(),c.m_angularComponentB,delta);
  if(!finitePhysicalBody(A)||!finitePhysicalBody(B))return false;
 }
 output=trial;return true;
}

inline bool normalRate(const btSolverConstraint& row,
 const btAlignedObjectArray<btSolverBody>& physical,double& output){
 const int a=row.m_solverBodyIdA,b=row.m_solverBodyIdB;
 if(a<0||b<0||a>=physical.size()||b>=physical.size()||
    !finiteVector(row.m_contactNormal1)||!finiteVector(row.m_contactNormal2)||
    !finiteVector(row.m_relpos1CrossNormal)||!finiteVector(row.m_relpos2CrossNormal))return false;
 auto contribution=[](const btSolverBody& s,const btVector3& axis,const btVector3& cross){
  if(!s.m_originalBody)return btScalar(0);
  const auto v=s.m_linearVelocity+s.m_deltaLinearVelocity+s.m_externalForceImpulse;
  const auto w=s.m_angularVelocity+s.m_deltaAngularVelocity+s.m_externalTorqueImpulse;
  return axis.dot(v)+cross.dot(w);
 };
 if(!finitePhysicalBody(physical[a])||!finitePhysicalBody(physical[b]))return false;
 const double rate=contribution(physical[a],row.m_contactNormal1,row.m_relpos1CrossNormal)+
                   contribution(physical[b],row.m_contactNormal2,row.m_relpos2CrossNormal);
 if(!std::isfinite(rate))return false;
 output=rate;return true;
}

// Desired COMBINED geometric rate; units m/s. Never clamp signed remaining
// clearance after physical motion. legacy is the original penetrating ERP
// target, saved BEFORE any independent-gap position-target override.
inline bool target(double distance,double h,double slop,double legacy,double physical,
 double& pose,double* desired_output=nullptr){
 if(desired_output==&pose||!std::isfinite(distance)||!std::isfinite(h)||h<=0||!std::isfinite(slop)||slop<0||
    !std::isfinite(legacy)||!std::isfinite(physical))return false;
 const double desired=std::abs(distance)<=slop?0.:
                      (distance>slop?-(distance-slop)/h:legacy);
 const double correction=desired-physical;
 if(!std::isfinite(desired)||!std::isfinite(correction))return false;
 pose=correction;if(desired_output)*desired_output=desired;return true;
}

// Normal-order vectors are separate from the full physical row array. Caller
// selects original independent normal indices and supplies exact current
// signed manifold distances, never estimates from output-frame states.
inline bool targets(const btAlignedObjectArray<btSolverConstraint*>& rows,
 const std::vector<int>& normals,const std::vector<double>& distances,
 const btAlignedObjectArray<btSolverBody>& physical,const btVectorXu& legacy_split,
 double h,double slop,btVectorXu& output,btVectorXu* physical_rates=nullptr,
 btVectorXu* desired_rates=nullptr){
 if(&output==&legacy_split||physical_rates==&output||desired_rates==&output||
    (physical_rates&&physical_rates==desired_rates)||physical_rates==&legacy_split||desired_rates==&legacy_split||
    legacy_split.rows()!=rows.size()||distances.size()!=normals.size()||
    !std::isfinite(h)||h<=0||!std::isfinite(slop)||slop<0)return false;
 const int n=static_cast<int>(normals.size());btVectorXu proposed(n),physical_values(n),desired_values(n);
 std::vector<bool> seen(rows.size(),false);
 for(int i=0;i<n;i++){
  const int row=normals[i];double u=0,correction=0,desired=0;
  if(row<0||row>=rows.size()||seen[row]||!rows[row])return false;
  seen[row]=true;
  if(!normalRate(*rows[row],physical,u)||
     !target(distances[i],h,slop,legacy_split[row],u,correction,&desired))return false;
  proposed[i]=correction;physical_values[i]=u;desired_values[i]=desired;
 }
 output=proposed;if(physical_rates)*physical_rates=physical_values;
 if(desired_rates)*desired_rates=desired_values;return true;
}
} // namespace combined_translation_review
