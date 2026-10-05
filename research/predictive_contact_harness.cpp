// Independent row-mechanics diagnostic; no collision discovery or native edits.
#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include <nlohmann/json.hpp>
#include "coulomb.h"
#include <iostream>
using json=nlohmann::json;
json array(const btVector3& v){return {v.x(),v.y(),v.z()};}
btVector3 momentum(btRigidBody& a,btRigidBody& b){return a.getLinearVelocity()+b.getLinearVelocity();}
btVector3 angular(btRigidBody& a,btRigidBody& b){return a.getCenterOfMassPosition().cross(a.getLinearVelocity())+b.getCenterOfMassPosition().cross(b.getLinearVelocity())+.004*(a.getAngularVelocity()+b.getAngularVelocity());}
double energy(btRigidBody& a,btRigidBody& b){return .5*(a.getLinearVelocity().length2()+b.getLinearVelocity().length2())+.002*(a.getAngularVelocity().length2()+b.getAngularVelocity().length2());}
int main(){json input;std::cin>>input;json out=json::array();
 for(const auto& c:input){double g=c.at("gap_m"),mu=c.at("pair_mu"),h=c.at("step_s");bool common=c.at("common_point");
  btSphereShape shape(.1);btRigidBody a(1,nullptr,&shape,btVector3(.004,.004,.004)),b(1,nullptr,&shape,btVector3(.004,.004,.004));
  a.setWorldTransform(btTransform(btQuaternion::getIdentity(),btVector3(0,0,.1+g/2)));b.setWorldTransform(btTransform(btQuaternion::getIdentity(),btVector3(0,0,-.1-g/2)));a.updateInertiaTensor();b.updateInertiaTensor();
  a.setLinearVelocity(btVector3(1,0,-1));b.setLinearVelocity(btVector3(-1,0,1));
  btVector3 xa(0,0,g/2),xb(0,0,-g/2);if(common)xa=xb=btVector3(0,0,0);
  btManifoldPoint cp(xa-a.getCenterOfMassPosition(),xb-b.getCenterOfMassPosition(),btVector3(0,0,1),g);
  cp.m_positionWorldOnA=xa;cp.m_positionWorldOnB=xb;cp.m_combinedFriction=mu;cp.m_combinedRestitution=0;
  btPersistentManifold manifold(&a,&b,0,.1,1e30);manifold.addManifoldPoint(cp,true);btPersistentManifold* manifolds[]={&manifold};btCollisionObject* bodies[]={&a,&b};
  btContactSolverInfo info;info.m_timeStep=h;info.m_numIterations=4096;info.m_erp=info.m_erp2=0;info.m_splitImpulse=false;info.m_solverMode=SOLVER_USE_WARMSTARTING|SOLVER_USE_2_FRICTION_DIRECTIONS|SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION;
  btDantzigSolver dantzig;CoulombMLCP solver(&dantzig);solver.tolerance=1e-10;solver.contact_slop_m=1e-9;
  auto p0=momentum(a,b),l0=angular(a,b),v0=a.getLinearVelocity();double e0=energy(a,b);
  solver.solveGroup(bodies,2,manifolds,1,nullptr,0,info,nullptr,nullptr);
  auto impulse=a.getLinearVelocity()-v0;json r=c;r["point_A_m"]=array(xa);r["point_B_m"]=array(xb);r["impulse_A_N_s"]=array(impulse);r["delta_linear_momentum_kg_m_s"]=array(momentum(a,b)-p0);r["delta_angular_momentum_kg_m2_s"]=array(angular(a,b)-l0);r["endpoint_predicted_delta_L"]=array((xa-xb).cross(impulse));r["energy_before_J"]=e0;r["energy_after_J"]=energy(a,b);r["residual_m_s"]=solver.stats.residual_max;r["normal_impulse_N_s"]=manifold.getContactPoint(0).m_appliedImpulse;r["omega_A_rad_s"]=array(a.getAngularVelocity());r["omega_B_rad_s"]=array(b.getAngularVelocity());out.push_back(r);
 }std::cout<<out.dump(2)<<"\n";
}
