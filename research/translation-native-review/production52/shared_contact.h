// Project geometry convention: a contact wrench acts at one world point.
// This changes lever arms, not the material/friction law. Bullet is unmodified.
#pragma once
#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/ConstraintSolver/btSolverConstraint.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

inline bool finiteContactBody(const btSolverBody& body){
 return body.m_originalBody&&body.m_originalBody->getInvMass()>0;
}

inline btVector3 sharedContactPoint(const btManifoldPoint& cp,
 const btSolverBody& a,const btSolverBody& b){
 const bool finiteA=finiteContactBody(a),finiteB=finiteContactBody(b);
 if(finiteA&&!finiteB)return cp.getPositionWorldOnA();
 if(finiteB&&!finiteA)return cp.getPositionWorldOnB();
 return cp.getPositionWorldOnA()*btScalar(.5)+cp.getPositionWorldOnB()*btScalar(.5);
}

inline btScalar contactRowFreeVelocity(const btSolverConstraint& row,
 const btSolverBody& a,const btSolverBody& b,bool tangent){
 auto contribution=[&](const btSolverBody& body,const btVector3& axis,
                       const btVector3& cross){
  if(!body.m_originalBody)return btScalar(0);
  // Upstream normal rows already include gyro; tangent correction is later.
  const btVector3 angular=body.m_angularVelocity+
      (tangent?btVector3(0,0,0):body.m_externalTorqueImpulse);
  return axis.dot(body.m_linearVelocity+body.m_externalForceImpulse)+cross.dot(angular);
 };
 return contribution(a,row.m_contactNormal1,row.m_relpos1CrossNormal)+
        contribution(b,row.m_contactNormal2,row.m_relpos2CrossNormal);
}

inline btScalar contactRowMobility(const btSolverConstraint& row,
 const btSolverBody& a,const btSolverBody& b){
 auto contribution=[&](const btSolverBody& body,const btVector3& axis,
                       const btVector3& cross,const btVector3& angular){
  if(!body.m_originalBody)return btScalar(0);
  return body.m_originalBody->getInvMass()*axis.length2()+cross.dot(angular);
 };
 return contribution(a,row.m_contactNormal1,row.m_relpos1CrossNormal,row.m_angularComponentA)+
        contribution(b,row.m_contactNormal2,row.m_relpos2CrossNormal,row.m_angularComponentB);
}

// Call after upstream row conversion, before ALL mobility assembly and before
// consistentTangentRHS. Repeating a call is mechanically idempotent.
// Return maximum endpoint relocation in metres for numerical provenance.
inline double transportSharedContactRow(btSolverConstraint& row,
 btSolverBody& a,btSolverBody& b,const btManifoldPoint& cp,bool tangent){
 const int unsupported=BT_CONTACT_FLAG_HAS_CONTACT_CFM|
     BT_CONTACT_FLAG_CONTACT_STIFFNESS_DAMPING|BT_CONTACT_FLAG_FRICTION_ANCHOR;
 if(row.m_cfm!=0||(cp.m_contactPointFlags&unsupported))
  throw std::runtime_error("Shared contact point does not support CFM, contact stiffness/damping or friction anchors");
 if(!(row.m_jacDiagABInv>0)||!std::isfinite(row.m_jacDiagABInv))
  throw std::runtime_error("Invalid original shared-contact row diagonal");
 const btVector3 point=sharedContactPoint(cp,a,b);
 const btScalar oldMobility=contactRowMobility(row,a,b);
 const btScalar oldFree=contactRowFreeVelocity(row,a,b,tangent);
 const btScalar oldJac=row.m_jacDiagABInv;
 const btVector3 oldAngularA=row.m_angularComponentA,oldAngularB=row.m_angularComponentB;
 btSolverConstraint next=row;
 auto transport=[&](btSolverBody& body,const btVector3& axis,
                    btVector3& cross,btVector3& angular){
  if(body.m_originalBody){
   cross=(point-body.m_worldTransform.getOrigin()).cross(axis);
   angular=body.m_originalBody->getInvInertiaTensorWorld()*cross*
       body.m_originalBody->getAngularFactor();
  }else{cross.setZero();angular.setZero();}
 };
 transport(a,next.m_contactNormal1,next.m_relpos1CrossNormal,next.m_angularComponentA);
 transport(b,next.m_contactNormal2,next.m_relpos2CrossNormal,next.m_angularComponentB);
 const btScalar newMobility=contactRowMobility(next,a,b);
 if(!(oldMobility>0&&newMobility>0)||!std::isfinite(oldMobility)||!std::isfinite(newMobility))
  throw std::runtime_error("Degenerate shared-contact row mobility");
 // Preserve upstream relaxation and velocity/position targets exactly.
 const btScalar relaxation=oldJac*oldMobility;
 next.m_jacDiagABInv=relaxation/newMobility;
 next.m_rhs=(row.m_rhs/oldJac+oldFree-contactRowFreeVelocity(next,a,b,tangent))*next.m_jacDiagABInv;
 next.m_rhsPenetration=row.m_rhsPenetration/oldJac*next.m_jacDiagABInv;
 if(!std::isfinite(next.m_jacDiagABInv)||!std::isfinite(next.m_rhs)||!std::isfinite(next.m_rhsPenetration))
  throw std::runtime_error("Nonfinite shared-contact transport");
 // convertContact already applied the normal warm start. Replace its angular
 // contribution in place; signed row B already has the opposite impulse axis.
 const btScalar impulse=row.m_appliedImpulse;
 if(a.m_originalBody)a.m_deltaAngularVelocity+=(next.m_angularComponentA-oldAngularA)*impulse*a.m_angularFactor;
 if(b.m_originalBody)b.m_deltaAngularVelocity+=(next.m_angularComponentB-oldAngularB)*impulse*b.m_angularFactor;
 row=next;
 return std::max(static_cast<double>((point-cp.getPositionWorldOnA()).length()),
                 static_cast<double>((point-cp.getPositionWorldOnB()).length()));
}
