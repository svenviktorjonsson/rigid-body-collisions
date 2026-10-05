// Compile only after root publishes the prospective plan and authorizes it.
// Uses actual read-only Bullet MLCP writeback as the parity oracle.
#include "combined_translation.h"
#include <BulletDynamics/MLCPSolvers/btMLCPSolver.h>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace ctr=combined_translation_review;
void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
void close(double a,double b){require(std::abs(a-b)<=1e-12*std::max({1.,std::abs(a),std::abs(b)}),"numeric mismatch");}
bool equal(const btVector3& a,const btVector3& b){for(int k=0;k<3;k++)if(a[k]!=b[k])return false;return true;}
bool equalBody(const btSolverBody& a,const btSolverBody& b){
 bool transform=equal(a.m_worldTransform.getOrigin(),b.m_worldTransform.getOrigin());
 for(int i=0;i<3;i++)transform=transform&&equal(a.m_worldTransform.getBasis()[i],b.m_worldTransform.getBasis()[i]);
 return transform&&a.m_originalBody==b.m_originalBody&&equal(a.m_linearVelocity,b.m_linearVelocity)&&
  equal(a.m_angularVelocity,b.m_angularVelocity)&&equal(a.m_deltaLinearVelocity,b.m_deltaLinearVelocity)&&
  equal(a.m_deltaAngularVelocity,b.m_deltaAngularVelocity)&&equal(a.m_externalForceImpulse,b.m_externalForceImpulse)&&
  equal(a.m_externalTorqueImpulse,b.m_externalTorqueImpulse)&&equal(a.m_invMass,b.m_invMass)&&
  equal(a.m_linearFactor,b.m_linearFactor)&&equal(a.m_angularFactor,b.m_angularFactor)&&
  equal(a.m_pushVelocity,b.m_pushVelocity)&&equal(a.m_turnVelocity,b.m_turnVelocity);
}
bool equalBodies(const btAlignedObjectArray<btSolverBody>& a,const btAlignedObjectArray<btSolverBody>& b){
 if(a.size()!=b.size())return false;for(int i=0;i<a.size();i++)if(!equalBody(a[i],b[i]))return false;return true;
}
bool equalVector(const btVectorXu& a,const btVectorXu& b){
 if(a.rows()!=b.rows())return false;for(int i=0;i<a.rows();i++)if(a[i]!=b[i])return false;return true;
}
btSolverBody solverBody(btRigidBody* original){
 btSolverBody s;s.m_originalBody=original;s.m_worldTransform.setIdentity();
 s.m_linearVelocity=btVector3(.2,-.4,.3);s.m_angularVelocity=btVector3(-.3,.6,.2);
 s.m_deltaLinearVelocity=btVector3(0,0,0);s.m_deltaAngularVelocity=btVector3(0,0,0);
 s.m_externalForceImpulse=btVector3(.07,-.11,.13);s.m_externalTorqueImpulse=btVector3(-.09,.05,.04);
 s.m_invMass=original?btVector3(original->getInvMass(),original->getInvMass(),original->getInvMass()):btVector3(0,0,0);
 s.m_linearFactor=btVector3(.8,1.1,.9);s.m_angularFactor=btVector3(.7,1.2,1.05);
 // These separate fake quantities must not enter the physical rate predictor.
 s.m_pushVelocity=btVector3(20,30,40);s.m_turnVelocity=btVector3(-10,15,11);
 if(original)s.m_worldTransform=original->getWorldTransform();return s;
}
btSolverConstraint row(int a,int b,const btVector3& axis,const btVector3& point,
 const btAlignedObjectArray<btSolverBody>& pool,double warm){
 btSolverConstraint c{};c.m_solverBodyIdA=a;c.m_solverBodyIdB=b;c.m_appliedImpulse=warm;c.m_appliedPushImpulse=0;
 c.m_contactNormal1=axis;c.m_contactNormal2=-axis;
 c.m_relpos1CrossNormal=(point-pool[a].m_worldTransform.getOrigin()).cross(axis);
 c.m_relpos2CrossNormal=(point-pool[b].m_worldTransform.getOrigin()).cross(-axis);
 c.m_angularComponentA=pool[a].m_originalBody?pool[a].m_originalBody->getInvInertiaTensorWorld()*c.m_relpos1CrossNormal*pool[a].m_angularFactor:btVector3(0,0,0);
 c.m_angularComponentB=pool[b].m_originalBody?pool[b].m_originalBody->getInvInertiaTensorWorld()*c.m_relpos2CrossNormal*pool[b].m_angularFactor:btVector3(0,0,0);
 return c;
}
class UpstreamOracle:public btMLCPSolver{
 bool solveMLCP(const btContactSolverInfo&)override{return true;}
public:
 UpstreamOracle():btMLCPSolver(nullptr){}
 btAlignedObjectArray<btSolverBody> apply(const btAlignedObjectArray<btSolverConstraint*>& rows,
  const btAlignedObjectArray<btSolverBody>& bodies,const btVectorXu& accepted){
  m_tmpSolverBodyPool=bodies;m_x=accepted;btAlignedObjectArray<btSolverConstraint> copy;
  copy.resize(rows.size());m_allConstraintPtrArray.clear();
  for(int i=0;i<rows.size();i++){copy[i]=*rows[i];m_allConstraintPtrArray.push_back(&copy[i]);}
  btContactSolverInfo info;info.m_splitImpulse=false;
  btMLCPSolver::solveGroupCacheFriendlyIterations(nullptr,0,nullptr,0,nullptr,0,info,nullptr);
  auto result=m_tmpSolverBodyPool;m_allConstraintPtrArray.clear();m_tmpSolverBodyPool.clear();return result;
 }
};

int main(){try{
 static_assert(sizeof(btScalar)==8,"Double precision is mandatory");
 btRigidBody A(2,nullptr,nullptr,btVector3(1.2,2.3,4.1));
 btRigidBody B(3,nullptr,nullptr,btVector3(2.1,1.8,3.7));
 btRigidBody K(0,nullptr,nullptr);K.setCollisionFlags(K.getCollisionFlags()|btCollisionObject::CF_KINEMATIC_OBJECT);
 A.setWorldTransform(btTransform(btQuaternion(btVector3(.2,.6,.3).normalized(),.7),btVector3(.1,-.2,.3)));A.updateInertiaTensor();
 B.setWorldTransform(btTransform(btQuaternion(btVector3(.5,-.2,.4).normalized(),-.4),btVector3(-.5,.1,-.4)));B.updateInertiaTensor();
 const auto originalAV=A.getLinearVelocity(),originalAO=A.getAngularVelocity();
 btAlignedObjectArray<btSolverBody> pool;pool.push_back(solverBody(&A));pool.push_back(solverBody(&B));pool.push_back(solverBody(nullptr));pool.push_back(solverBody(&K));
 pool[3].m_linearVelocity=btVector3(.8,.9,-.2);pool[3].m_angularVelocity=btVector3(.3,-.6,1.4);
 pool[3].m_externalForceImpulse.setZero();pool[3].m_externalTorqueImpulse.setZero();
 const btVector3 n=btVector3(1,2,-1).normalized(),t=n.cross(btVector3(.2,-.1,.8)).normalized(),s=n.cross(t);
 btAlignedObjectArray<btSolverConstraint> storage;storage.resize(5);
 storage[0]=row(0,1,n,btVector3(.4,.1,.2),pool,.4);
 storage[1]=row(0,1,t,btVector3(.4,.1,.2),pool,.02);
 storage[2]=row(0,1,s,btVector3(.4,.1,.2),pool,.03);
 storage[3]=row(0,2,btVector3(0,1,0),btVector3(.7,.2,-.1),pool,.1);
 storage[4]=row(0,3,btVector3(0,0,1),btVector3(.3,.6,.1),pool,.02);
 btAlignedObjectArray<btSolverConstraint*> rows;for(int i=0;i<storage.size();i++)rows.push_back(&storage[i]);
 // Construct a consistent nonzero existing warm delta. Actual oracle retains it.
 for(int i=0;i<rows.size();i++){
  auto& c=*rows[i];auto& a=pool[c.m_solverBodyIdA];auto& b=pool[c.m_solverBodyIdB];
  a.internalApplyImpulse(c.m_contactNormal1*a.internalGetInvMass(),c.m_angularComponentA,c.m_appliedImpulse);
  b.internalApplyImpulse(c.m_contactNormal2*b.internalGetInvMass(),c.m_angularComponentB,c.m_appliedImpulse);
 }
 const auto before=pool;btVectorXu pressure(5);pressure[0]=1.2;pressure[1]=.05;pressure[2]=-.07;pressure[3]=.3;pressure[4]=.06;
 btAlignedObjectArray<btSolverBody> predicted;require(ctr::acceptedPhysicalBodies(rows,pool,pressure,predicted),"predictor rejected fixture");
 UpstreamOracle oracle;const auto expected=oracle.apply(rows,pool,pressure);
 require(equalBodies(predicted,expected),"copied predictor differs from actual upstream accepted-delta writeback");
 require(equalBodies(pool,before),"original pool mutated");
 require(equal(A.getLinearVelocity(),originalAV)&&equal(A.getAngularVelocity(),originalAO),"original rigid body mutated");
 const double warm[5]={.4,.02,.03,.1,.02};for(int i=0;i<5;i++)require(rows[i]->m_appliedImpulse==warm[i],"original row cache mutated");
 require(equalBody(predicted[3],before[3]),"accepted impulses changed prescribed zero-mass kinematic motion");
 std::cout<<"PASS actual upstream parity: warm normal/two tangents/fixed and rotating kinematic slots/rotated anisotropic inertia/external force and gyro/nonunit factors; originals unchanged\n";
 double u=0;require(ctr::normalRate(*rows[0],predicted,u),"normal rate rejected");
 const auto va=predicted[0].m_linearVelocity+predicted[0].m_deltaLinearVelocity+predicted[0].m_externalForceImpulse;
 const auto vb=predicted[1].m_linearVelocity+predicted[1].m_deltaLinearVelocity+predicted[1].m_externalForceImpulse;
 require(std::abs(u-n.dot(va-vb))>1e-5,"fixture fails to exercise angular normal contribution");
 auto noFake=predicted;for(int i=0;i<noFake.size();i++){noFake[i].m_pushVelocity.setZero();noFake[i].m_turnVelocity.setZero();}
 double withoutFake=0;require(ctr::normalRate(*rows[0],noFake,withoutFake)&&u==withoutFake,"fake pose rate contaminated physical normal rate");
 std::cout<<"PASS full angular physical normal rate; fake push/turn excluded\n";
 const auto savedOutput=predicted;btVectorXu bad=pressure;bad[1]=std::numeric_limits<double>::quiet_NaN();
 require(!ctr::acceptedPhysicalBodies(rows,pool,bad,predicted)&&equalBodies(predicted,savedOutput),"nonfinite pressure mutated output");
 const auto savedRow=storage[0];storage[0].m_solverBodyIdA=-1;
 require(!ctr::acceptedPhysicalBodies(rows,pool,pressure,predicted)&&equalBodies(predicted,savedOutput),"invalid body id mutated output");storage[0]=savedRow;
 require(!ctr::acceptedPhysicalBodies(rows,pool,pressure,pool)&&equalBodies(pool,before),"aliased original pool accepted");
 btVectorXu legacy(5);legacy.setZero();legacy[0]=.2;legacy[3]=.1;
 btVectorXu targets(1),rates(1),desired(1);targets[0]=123;rates[0]=456;desired[0]=789;
 require(ctr::targets(rows,{0,3},{.01,-.0005},predicted,legacy,.001,1e-9,targets,&rates,&desired),"all-row target builder rejected");
 close(targets[0],-(.01-1e-9)/.001-u);close(targets[1],.1-rates[1]);
 const auto savedTargets=targets,savedRates=rates,savedDesired=desired;
 require(!ctr::targets(rows,{0,0},{.01,.01},predicted,legacy,.001,1e-9,targets,&rates,&desired)&&
  equalVector(targets,savedTargets)&&equalVector(rates,savedRates)&&equalVector(desired,savedDesired),"duplicate normal changed outputs");
 require(!ctr::targets(rows,{0,3},{.01,-.0005},predicted,legacy,.001,1e-9,targets,&targets,&desired)&&equalVector(targets,savedTargets),"aliased target outputs accepted");
 std::cout<<"PASS finite/dimension/index/alias rejection and output preservation; normal-order target bookkeeping\n";
 const double distances[5]={-.002,-1e-9,0.,1e-9,.002};const double velocities[3]={-2.,0.,2.};
 for(double d:distances)for(double v:velocities){
  double output=123,desiredValue=456;require(ctr::target(d,.001,1e-9,.4,v,output,&desiredValue),"distance branch rejected");
  const double want=std::abs(d)<=1e-9?0.:(d>1e-9?-(d-1e-9)/.001:.4);close(output,want-v);close(desiredValue,want);
 }
 double unchanged=123;require(!ctr::target(.01,0.,1e-9,.4,0.,unchanged)&&unchanged==123,"zero timestep accepted");
 require(!ctr::target(.01,.001,1e-9,.4,std::numeric_limits<double>::infinity(),unchanged)&&unchanged==123,"nonfinite physical rate accepted");
 double negativeRemaining=0;require(ctr::target(.001000001,.001,1e-9,.4,-2.,negativeRemaining)&&negativeRemaining>0,"negative remaining clearance hidden");
 double left=0,right=0;require(ctr::target(-.0005,.001,1e-9,.1,.95,left)&&ctr::target(.001000001,.001,1e-9,0.,-.95,right),"opposing-face control failed");
 require(left<0&&right<0,"physical overlap resolution did not suppress unnecessary pose impulse");
 close(.001000001-.001*.95,.000050001);
 std::cout<<"PASS all 15 distance/rate branches, signed remaining clearance and opposing-face zero-push control\n";
 return EXIT_SUCCESS;
 }catch(const std::exception& e){std::cerr<<"FAIL "<<e.what()<<'\n';return EXIT_FAILURE;}}
