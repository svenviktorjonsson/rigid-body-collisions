#include <btBulletDynamicsCommon.h>
#include "translation_split.h"
#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(){try{
 btSphereShape shape(.1);
 btRigidBody a(btRigidBody::btRigidBodyConstructionInfo(2,nullptr,&shape,btVector3(.02,.03,.04)));
 btRigidBody b(btRigidBody::btRigidBodyConstructionInfo(5,nullptr,&shape,btVector3(.05,.06,.07)));
 btAlignedObjectArray<btSolverBody> bodies;bodies.resize(3);
 bodies[0].m_originalBody=&a;bodies[1].m_originalBody=&b;bodies[2].m_originalBody=nullptr;
 btSolverConstraint rows[3];
 rows[0].m_solverBodyIdA=0;rows[0].m_solverBodyIdB=1;
 rows[0].m_contactNormal1=btVector3(1,0,0);rows[0].m_contactNormal2=btVector3(-1,0,0);
 rows[1].m_solverBodyIdA=1;rows[1].m_solverBodyIdB=2;
 rows[1].m_contactNormal1=btVector3(.6,.8,0);rows[1].m_contactNormal2=btVector3(-.6,-.8,0);
 rows[2].m_solverBodyIdA=2;rows[2].m_solverBodyIdB=0;
 rows[2].m_contactNormal1=btVector3(0,-1,0);rows[2].m_contactNormal2=btVector3(0,1,0);
 double J[3][6]={{1,0,0,-1,0,0},{0,0,0,.6,.8,0},{0,1,0,0,0,0}};
 double inverse_mass[6]={.5,.5,.5,.2,.2,.2};
 for(int i=0;i<3;i++)for(int j=0;j<3;j++){
  double reference=0;for(int k=0;k<6;k++)reference+=J[i][k]*inverse_mass[k]*J[j][k];
  require(std::abs(translationRowMobility(rows[i],rows[j],bodies)-reference)<1e-14,"Signed translation Gram assembly mismatch");
 }
 for(int i=0;i<3;i++){
  bodies[i].m_deltaAngularVelocity=btVector3(1,2,3);bodies[i].m_deltaLinearVelocity=btVector3(4,5,6);
  bodies[i].m_pushVelocity=btVector3(7,8,9);bodies[i].m_turnVelocity=btVector3(10,11,12);
 }
 clearPositionTurns(bodies);
 for(int i=0;i<3;i++){
  require(bodies[i].m_turnVelocity.length2()==0,"Position turn was retained");
  require(bodies[i].m_deltaAngularVelocity==btVector3(1,2,3),"Physical angular impulse was modified");
  require(bodies[i].m_deltaLinearVelocity==btVector3(4,5,6),"Physical linear impulse was modified");
  require(bodies[i].m_pushVelocity==btVector3(7,8,9),"Linear position repair was modified");
 }
 btMatrixXu redundant(2,2);btVectorXu rhs(2),upper(2),impulse(2);
 for(int i=0;i<2;i++){rhs[i]=.02;upper[i]=1e30;for(int j=0;j<2;j++)redundant.setElem(i,j,.5);}
 double residual=0;
 require(translationSplitSolve(redundant,rhs,upper,impulse,1e-10,64,&residual),"Redundant consistent position constraints rejected");
 require(residual<=1e-10&&std::abs(impulse[0]+impulse[1]-.04)<1e-10,"Redundant position target was not satisfied");
 redundant.setElem(0,1,-.5);redundant.setElem(1,0,-.5);
 require(!translationSplitSolve(redundant,rhs,upper,impulse,1e-10,64,&residual),"Contradictory opposed position targets were accepted");
 require(residual>1e-10,"Contradictory target residual was not retained");
 // An open contact must permit displacement toward it while correcting the
 // opposed penetration. The old zero target makes this physical clearance
 // artificially unusable.
 redundant.setElem(0,0,1);redundant.setElem(1,1,1);redundant.setElem(0,1,-1);redundant.setElem(1,0,-1);
 rhs[0]=0;rhs[1]=.1;upper[0]=upper[1]=1e10;
 require(!translationSplitSolve(redundant,rhs,upper,impulse,1e-8,64,&residual,nullptr,false),"Opposed zero-clearance target should reject");
 rhs[0]=translationGapTarget(.01,.01,1e-9,0.);
 require(translationSplitSolve(redundant,rhs,upper,impulse,1e-8,64,&residual),"Available physical clearance must permit repair");
 const double dx=.01*(impulse[0]-impulse[1]);
 require(.01+dx>=1e-9&&-.005-dx>-.005,"Gap repair must preserve clearance and reduce penetration");
 require(translationGapTarget(-.005,.01,1e-9,.1)==.1,"Penetration target unchanged");
 require(translationGapTarget(5e-10,.01,1e-9,.1)==0,"Within-slop target is zero");
 bool invalid=false;try{translationGapTarget(.01,0,1e-9,0.);}catch(const std::runtime_error&){invalid=true;}require(invalid,"Invalid timestep rejected");
 bodies[0].m_linearVelocity=btVector3(1,2,3);bodies[0].m_deltaLinearVelocity=btVector3(4,5,6);
 bodies[0].m_externalForceImpulse=btVector3(.5,.5,.5);a.setGravity(btVector3(0,0,-9.81));
 auto ledger=translationPoseChange(bodies[0],.01);
 require((ledger.displacement-btVector3(.07,.08,.09)).length()<1e-14,"Actual split displacement ledger");
 require((ledger.orbital_momentum-btVector3(.17,-.34,.17)).length()<1e-13,"Ledger must use final accepted physical velocity");
 require(std::abs(ledger.potential_energy-1.7658)<1e-12,"Gravity potential ledger sign and mass");
 require(bodies[0].m_deltaLinearVelocity==btVector3(4,5,6)&&bodies[0].m_turnVelocity.length2()==0,"Ledger observes without mutation");
 std::cout<<"Translation split checks PASS (Gram, isolated pushes, redundant and infeasible constraints)\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
