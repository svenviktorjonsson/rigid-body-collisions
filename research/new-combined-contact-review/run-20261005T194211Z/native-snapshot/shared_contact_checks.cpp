// Independent upstream-row and mechanics checks for shared contact points.
#include "shared_contact.h"
#include <BulletDynamics/ConstraintSolver/btSequentialImpulseConstraintSolver.h>
#include <iostream>

namespace {
void scalar(btScalar a,btScalar b,const char* message,btScalar tolerance=2e-11){
 if(!std::isfinite(a)||!std::isfinite(b)||btFabs(a-b)>tolerance*(1+btFabs(b)))throw std::runtime_error(message);
}
void vector(const btVector3& a,const btVector3& b,const char* message,btScalar tolerance=2e-11){
 if((a-b).length()>tolerance*(1+b.length()))throw std::runtime_error(message);
}
btScalar explicitMobility(btRigidBody& a,btRigidBody& b,const btVector3& point,const btVector3& axis){
 const auto ta=(point-a.getCenterOfMassPosition()).cross(axis),tb=(point-b.getCenterOfMassPosition()).cross(axis);
 return a.getInvMass()+b.getInvMass()+ta.dot(a.getInvInertiaTensorWorld()*ta)+tb.dot(b.getInvInertiaTensorWorld()*tb);
}
btScalar explicitFree(btRigidBody& a,btRigidBody& b,const btSolverBody& sa,const btSolverBody& sb,
 const btVector3& point,const btVector3& axis,bool tangent){
 const auto wa=sa.m_angularVelocity+(tangent?btVector3(0,0,0):sa.m_externalTorqueImpulse);
 const auto wb=sb.m_angularVelocity+(tangent?btVector3(0,0,0):sb.m_externalTorqueImpulse);
 return axis.dot(sa.m_linearVelocity+sa.m_externalForceImpulse+wa.cross(point-a.getCenterOfMassPosition())
                -sb.m_linearVelocity-sb.m_externalForceImpulse-wb.cross(point-b.getCenterOfMassPosition()));
}
class Inspect:public btSequentialImpulseConstraintSolver {
public:
 void checks(bool swap,const btQuaternion& rotation){
  btBoxShape shapeA(btVector3(.1,.2,.3)),shapeB(btVector3(.2,.1,.15));btVector3 ia,ib;
  shapeA.calculateLocalInertia(1,ia);shapeB.calculateLocalInertia(2,ib);
  btRigidBody first(1,nullptr,&shapeA,ia),second(2,nullptr,&shapeB,ib);
  const btMatrix3x3 Q(rotation);
  first.setWorldTransform(btTransform(rotation,Q*btVector3(.2,.3,.1)));
  second.setWorldTransform(btTransform(rotation,Q*btVector3(-.3,.1,-.1)));
  first.updateInertiaTensor();second.updateInertiaTensor();
  first.setAngularVelocity(Q*btVector3(3,4,5));second.setAngularVelocity(Q*btVector3(-1,2,3));
  first.setLinearVelocity(Q*btVector3(2,.3,-1));second.setLinearVelocity(Q*btVector3(-1,.2,1));
  btRigidBody& a=swap?second:first;btRigidBody& b=swap?first:second;
  btVector3 pa=Q*btVector3(.04,.05,.07),pb=Q*btVector3(.03,.08,.06);
  btVector3 n=Q*btVector3(.2,.5,.8).normalized();if(swap){std::swap(pa,pb);n=-n;}
  btContactSolverInfo info;info.m_timeStep=.01;info.m_numIterations=4096;
  info.m_solverMode=SOLVER_USE_WARMSTARTING;info.m_erp=info.m_erp2=0;info.m_splitImpulse=false;
  btCollisionObject* bodies[]={&a,&b};convertBodies(bodies,2,info);
  auto& sa=m_tmpSolverBodyPool[a.getCompanionId()];auto& sb=m_tmpSolverBodyPool[b.getCompanionId()];
  btManifoldPoint cp(pa-a.getCenterOfMassPosition(),pb-b.getCenterOfMassPosition(),n,.005);
  cp.m_positionWorldOnA=pa;cp.m_positionWorldOnB=pb;cp.m_combinedFriction=1;cp.m_combinedRestitution=0;cp.m_appliedImpulse=.4;
  const btVector3 common=(pa+pb)*.5;
  vector(sharedContactPoint(cp,sa,sb),common,"Finite-body midpoint");
  const btVector3 ra=pa-a.getCenterOfMassPosition(),rb=pb-b.getCenterOfMassPosition();
  btVector3 t=n.cross(Q*btVector3(1,0,0)).normalized();
  for(bool tangent:{false,true}){
   const btVector3 axis=tangent?t:n;btSolverConstraint c;
   if(tangent)setupFrictionConstraint(c,axis,a.getCompanionId(),b.getCompanionId(),cp,ra,rb,&a,&b,.73,info);
   else{btScalar relaxation=0;setupContactConstraint(c,a.getCompanionId(),b.getCompanionId(),cp,info,relaxation,ra,rb);}
   const btScalar oldJac=c.m_jacDiagABInv;
   const btVector3 ta=ra.cross(axis),tb=rb.cross(axis);
   const btScalar originalMob=a.getInvMass()+b.getInvMass()+ta.dot(a.getInvInertiaTensorWorld()*ta)+tb.dot(b.getInvInertiaTensorWorld()*tb);
   const btScalar target=c.m_rhs/oldJac+contactRowFreeVelocity(c,sa,sb,tangent);
   // Exercise nonzero split target and arbitrary target preserved from upstream.
   c.m_rhs+=.37*oldJac;c.m_rhsPenetration=.12*oldJac;
   if(tangent){c.m_appliedImpulse=.2;sa.m_deltaAngularVelocity+=c.m_angularComponentA*.2;sb.m_deltaAngularVelocity+=c.m_angularComponentB*.2;}
   const btVector3 beforeA=sa.m_deltaAngularVelocity,beforeB=sb.m_deltaAngularVelocity;
   const btVector3 angularA=c.m_angularComponentA,angularB=c.m_angularComponentB;
   const btScalar applied=c.m_appliedImpulse;
   const double move=transportSharedContactRow(c,sa,sb,cp,tangent);
   scalar(move,(pa-pb).length()/2,"Relocation diagnostic");
   const btVector3 newA=(common-a.getCenterOfMassPosition()).cross(axis);
   const btVector3 newB=(common-b.getCenterOfMassPosition()).cross(-axis);
   vector(c.m_relpos1CrossNormal,newA,"A signed lever");vector(c.m_relpos2CrossNormal,newB,"B signed lever");
   vector(c.m_angularComponentA,a.getInvInertiaTensorWorld()*newA,"Full A inertia");
   vector(c.m_angularComponentB,b.getInvInertiaTensorWorld()*newB,"Full B inertia");
   scalar(c.m_jacDiagABInv,oldJac*originalMob/explicitMobility(a,b,common,axis),"Relaxation preserved");
   scalar(c.m_rhs/c.m_jacDiagABInv,target+.37-explicitFree(a,b,sa,sb,common,axis,tangent),"Free velocity and target preserved");
   scalar(c.m_rhsPenetration/c.m_jacDiagABInv,.12,"Split target preserved");
   vector(sa.m_deltaAngularVelocity,beforeA+(c.m_angularComponentA-angularA)*applied,"A warm start transported");
   vector(sb.m_deltaAngularVelocity,beforeB+(c.m_angularComponentB-angularB)*applied,"B warm start transported");
   if(tangent){
    const btScalar corrected=c.m_rhs/c.m_jacDiagABInv-c.m_relpos1CrossNormal.dot(sa.m_externalTorqueImpulse)-c.m_relpos2CrossNormal.dot(sb.m_externalTorqueImpulse);
    scalar(corrected,target+.37-explicitFree(a,b,sa,sb,common,axis,false),"Tangent gyro correction follows transport");
   }
   const auto stableA=sa.m_deltaAngularVelocity,stableB=sb.m_deltaAngularVelocity;
   const btScalar stableRhs=c.m_rhs,stableJac=c.m_jacDiagABInv;
   transportSharedContactRow(c,sa,sb,cp,tangent);
   vector(sa.m_deltaAngularVelocity,stableA,"Idempotent A warm start");vector(sb.m_deltaAngularVelocity,stableB,"Idempotent B warm start");
   scalar(c.m_rhs,stableRhs,"Idempotent RHS");scalar(c.m_jacDiagABInv,stableJac,"Idempotent diagonal");
   for(int flag:{BT_CONTACT_FLAG_HAS_CONTACT_CFM,BT_CONTACT_FLAG_CONTACT_STIFFNESS_DAMPING,BT_CONTACT_FLAG_FRICTION_ANCHOR}){
    cp.m_contactPointFlags=flag;bool rejected=false;try{transportSharedContactRow(c,sa,sb,cp,tangent);}catch(const std::runtime_error&){rejected=true;}
    if(!rejected)throw std::runtime_error("Unsupported contact accepted");cp.m_contactPointFlags=0;
   }
  }
 }
 void momentum(){
  btBoxShape shapeA(btVector3(.1,.2,.3)),shapeB(btVector3(.2,.1,.15));btVector3 ia,ib;
  shapeA.calculateLocalInertia(1,ia);shapeB.calculateLocalInertia(2,ib);
  btRigidBody a(1,nullptr,&shapeA,ia),b(2,nullptr,&shapeB,ib);
  a.setWorldTransform(btTransform(btQuaternion(btVector3(1,2,3).normalized(),.7),btVector3(.2,.3,.1)));
  b.setWorldTransform(btTransform(btQuaternion(btVector3(3,1,2).normalized(),-.4),btVector3(-.3,.1,-.1)));
  a.updateInertiaTensor();b.updateInertiaTensor();a.setLinearVelocity(btVector3(.2,.1,-1));b.setLinearVelocity(btVector3(-.1,-.05,1));
  btContactSolverInfo info;info.m_timeStep=.01;info.m_solverMode=0;info.m_erp=info.m_erp2=0;info.m_splitImpulse=false;
  btCollisionObject* bodies[]={&a,&b};convertBodies(bodies,2,info);
  auto& sa=m_tmpSolverBodyPool[a.getCompanionId()];auto& sb=m_tmpSolverBodyPool[b.getCompanionId()];
  const btVector3 pa(.04,.05,.07),pb(.03,.08,.06),point=(pa+pb)*.5;
  btManifoldPoint cp(pa-a.getCenterOfMassPosition(),pb-b.getCenterOfMassPosition(),btVector3(0,0,1),.005);cp.m_positionWorldOnA=pa;cp.m_positionWorldOnB=pb;cp.m_combinedFriction=1;
  btSolverConstraint rows[3];btVector3 axes[]={btVector3(1,0,0),btVector3(0,1,0),btVector3(0,0,1)};
  for(int i=0;i<3;i++){
   if(i<2)setupFrictionConstraint(rows[i],axes[i],a.getCompanionId(),b.getCompanionId(),cp,cp.m_localPointA,cp.m_localPointB,&a,&b,1,info);
   else{btScalar relaxation=0;setupContactConstraint(rows[i],a.getCompanionId(),b.getCompanionId(),cp,info,relaxation,cp.m_localPointA,cp.m_localPointB);}
   transportSharedContactRow(rows[i],sa,sb,cp,i<2);
  }
  const btVector3 ra=point-a.getCenterOfMassPosition(),rb=point-b.getCenterOfMassPosition();
  btMatrix3x3 K;for(int i=0;i<3;i++)for(int j=0;j<3;j++){
   const auto ta=ra.cross(axes[i]),tb=rb.cross(axes[i]);
   K[i][j]=(i==j?a.getInvMass()+b.getInvMass():0)+ta.dot(a.getInvInertiaTensorWorld()*ra.cross(axes[j]))+tb.dot(b.getInvInertiaTensorWorld()*rb.cross(axes[j]));
  }
  const btVector3 u=a.getLinearVelocity()-b.getLinearVelocity();const btVector3 p=K.inverse()*(-u);
  if(p.z()<=0)throw std::runtime_error("Momentum check requires compressive impulse");
  for(int i=0;i<3;i++){
   sa.internalApplyImpulse(rows[i].m_contactNormal1*sa.m_invMass,rows[i].m_angularComponentA,p[i]);
   sb.internalApplyImpulse(rows[i].m_contactNormal2*sb.m_invMass,rows[i].m_angularComponentB,p[i]);
  }
  vector(sa.m_deltaLinearVelocity/a.getInvMass()+sb.m_deltaLinearVelocity/b.getInvMass(),btVector3(0,0,0),"Total linear impulse");
  const auto spinA=a.getInvInertiaTensorWorld().inverse()*sa.m_deltaAngularVelocity;
  const auto spinB=b.getInvInertiaTensorWorld().inverse()*sb.m_deltaAngularVelocity;
  vector(a.getCenterOfMassPosition().cross(sa.m_deltaLinearVelocity/a.getInvMass())+b.getCenterOfMassPosition().cross(sb.m_deltaLinearVelocity/b.getInvMass())+spinA+spinB,btVector3(0,0,0),"Total orbital plus spin impulse");
  const auto newVA=a.getLinearVelocity()+sa.m_deltaLinearVelocity,newVB=b.getLinearVelocity()+sb.m_deltaLinearVelocity;
  vector(newVA+sa.m_deltaAngularVelocity.cross(ra)-newVB-sb.m_deltaAngularVelocity.cross(rb),btVector3(0,0,0),"Sticking contact velocity");
  const double before=.5*a.getLinearVelocity().length2()/a.getInvMass()+.5*b.getLinearVelocity().length2()/b.getInvMass();
  const double after=.5*newVA.length2()/a.getInvMass()+.5*newVB.length2()/b.getInvMass()+.5*sa.m_deltaAngularVelocity.dot(spinA)+.5*sb.m_deltaAngularVelocity.dot(spinB);
  scalar(after-before,u.dot(p)+.5*p.dot(K*p),"Full inertia energy identity");
  if(after>before+1e-12)throw std::runtime_error("Sticking impulse is not passive");
 }
};
}
int main(){
 for(bool swap:{false,true})for(const auto q:{btQuaternion::getIdentity(),btQuaternion(btVector3(1,2,3).normalized(),.7)}){Inspect inspect;inspect.checks(swap,q);}
 Inspect momentum;momentum.momentum();
 btSolverBody a{},b{};btSphereShape shape(.1);btRigidBody rigid(1,nullptr,&shape,btVector3(.004,.004,.004));
 btManifoldPoint cp;cp.m_positionWorldOnA=btVector3(1,2,3);cp.m_positionWorldOnB=btVector3(4,5,6);
 a.m_originalBody=&rigid;vector(sharedContactPoint(cp,a,b),cp.getPositionWorldOnA(),"Finite A endpoint");
 a.m_originalBody=nullptr;b.m_originalBody=&rigid;vector(sharedContactPoint(cp,a,b),cp.getPositionWorldOnB(),"Finite B endpoint");
 b.m_originalBody=nullptr;vector(sharedContactPoint(cp,a,b),btVector3(2.5,3.5,4.5),"No finite midpoint");
 cp.m_positionWorldOnA=cp.m_positionWorldOnB=btVector3(1e308,0,0);
 const auto extreme=sharedContactPoint(cp,a,b);
 if(!std::isfinite(extreme.x())||extreme.x()!=1e308)throw std::runtime_error("Midpoint overflow");
 std::cout<<"Shared contact checks PASS (full inertia, rotated/swapped rows, normal/tangent targets, split RHS, warm start, gyro, P/L/energy, endpoint selection, unsupported flags)\n";
}
