// Opt-in observer metadata only; no solver, body or manifold mutations.
#pragma once
#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/ConstraintSolver/btSolverBody.h>
#include <BulletDynamics/ConstraintSolver/btSolverConstraint.h>
#include <LinearMath/btMatrixX.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace position_geometry {
using Json=nlohmann::json;
inline Json vector(const btVector3& v){return Json::array({v.x(),v.y(),v.z()});}
inline Json matrix(const btMatrix3x3& m){
 Json a=Json::array();for(int i=0;i<3;i++)a.push_back(vector(m[i]));return a;
}
inline Json transform(const btTransform& t){
 const auto q=t.getRotation();return {{"position",vector(t.getOrigin())},
  {"quaternion_xyzw",Json::array({q.x(),q.y(),q.z(),q.w()})},
  {"rotation",matrix(t.getBasis())}};
}
inline Json body(const btSolverBody& s,int solver_id){
 Json j={{"solver_body_id",solver_id},{"solver_transform",transform(s.m_worldTransform)},
  {"solver_linear_velocity",vector(s.m_linearVelocity)},
  {"solver_angular_velocity",vector(s.m_angularVelocity)},
  {"external_force_velocity",vector(s.m_externalForceImpulse)},
  {"external_torque_velocity",vector(s.m_externalTorqueImpulse)}};
 const auto* rb=s.m_originalBody;j["has_original_body"]=rb!=nullptr;
 if(!rb){j["body_id"]=-1;j["inverse_mass"]=0.;j["mass"]=nullptr;return j;}
 const double invmass=rb->getInvMass();j["body_id"]=rb->getUserIndex();
 j["inverse_mass"]=invmass;j["mass"]=invmass>0?Json(1./invmass):Json(nullptr);
 j["kinematic"]=rb->isKinematicObject();j["static"]=rb->isStaticObject();
 j["world_transform"]=transform(rb->getWorldTransform());
 j["inverse_world_inertia"]=matrix(rb->getInvInertiaTensorWorld());
 j["linear_factor"]=vector(rb->getLinearFactor());j["angular_factor"]=vector(rb->getAngularFactor());
 j["linear_velocity"]=vector(rb->getLinearVelocity());j["angular_velocity"]=vector(rb->getAngularVelocity());
 return j;
}
}

inline nlohmann::json positionGeometry(
 const btAlignedObjectArray<btSolverConstraint*>& rows,
 const btAlignedObjectArray<btSolverBody>& bodies,const std::vector<int>& normals,
 const btVectorXu& bSplit,double h){
 using namespace position_geometry;
 if(!(h>0)||!std::isfinite(h)||bSplit.rows()!=rows.size())
  throw std::runtime_error("Invalid position geometry observer dimensions/time");
 Json out={{"schema","normal-position-geometry-v1"},{"internal_dt_s",h},
  {"scope","Exact pre-repair solver/body/manifold metadata; observation only"},
  {"shared_point_convention","Finite/fixed: finite surface endpoint; finite/finite: endpoint midpoint"},
  {"bodies",Json::array()},{"rows",Json::array()}};
 for(int i=0;i<bodies.size();i++)out["bodies"].push_back(body(bodies[i],i));
 for(size_t i=0;i<normals.size();i++){
  const int index=normals[i];
  if(index<0||index>=rows.size()||!rows[index])throw std::runtime_error("Invalid position geometry normal index");
  const auto& r=*rows[index];const int a=r.m_solverBodyIdA,b=r.m_solverBodyIdB;
  if(a<0||b<0||a>=bodies.size()||b>=bodies.size())throw std::runtime_error("Invalid position geometry body index");
  const auto* cp=static_cast<const btManifoldPoint*>(r.m_originalContactPoint);
  Json row={{"normal_row_index",i},{"original_row_index",index},
   {"solver_body_id_a",a},{"solver_body_id_b",b},
   {"body_id_a",bodies[a].m_originalBody?bodies[a].m_originalBody->getUserIndex():-1},
   {"body_id_b",bodies[b].m_originalBody?bodies[b].m_originalBody->getUserIndex():-1},
   {"linear_jacobian_a",vector(r.m_contactNormal1)},
   {"linear_jacobian_b",vector(r.m_contactNormal2)},
   {"angular_jacobian_a",vector(r.m_relpos1CrossNormal)},
   {"angular_jacobian_b",vector(r.m_relpos2CrossNormal)},
   {"angular_mobility_a",vector(r.m_angularComponentA)},
   {"angular_mobility_b",vector(r.m_angularComponentB)},
   {"split_target_m_s",bSplit[index]},{"row_rhs_penetration",r.m_rhsPenetration},
   {"row_jacobian_diagonal_inverse",r.m_jacDiagABInv},
   {"has_manifold_point",cp!=nullptr}};
  if(cp){
   const auto pa=cp->getPositionWorldOnA(),pb=cp->getPositionWorldOnB();
   const bool fa=bodies[a].m_originalBody&&bodies[a].m_originalBody->getInvMass()>0;
   const bool fb=bodies[b].m_originalBody&&bodies[b].m_originalBody->getInvMass()>0;
   const btVector3 common=fa&&!fb?pa:(fb&&!fa?pb:(pa+pb)*btScalar(.5));
   row["world_endpoint_a"]=vector(pa);row["world_endpoint_b"]=vector(pb);
   row["world_endpoint_midpoint"]=vector((pa+pb)*btScalar(.5));
   row["shared_world_point"]=vector(common);row["normal_world_on_b"]=vector(cp->m_normalWorldOnB);
   row["signed_distance_m"]=cp->getDistance();row["local_point_a"]=vector(cp->m_localPointA);
   row["local_point_b"]=vector(cp->m_localPointB);row["contact_flags"]=cp->m_contactPointFlags;
  }
  out["rows"].push_back(row);
 }
 return out;
}
