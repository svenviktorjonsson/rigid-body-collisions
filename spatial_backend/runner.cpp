// Project adapter; Bullet itself is unmodified and retains its upstream license.
#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/MLCPSolvers/btMLCPSolver.h>
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include <nlohmann/json.hpp>
#include <chrono>
#include <iostream>
#include <memory>
#include <vector>
#include <fstream>
using json=nlohmann::json;
#include "coulomb.h"
btVector3 vec(const json& j){return {j[0].get<double>(),j[1].get<double>(),j[2].get<double>()};}
btQuaternion quat(const json& j){return {j[0].get<double>(),j[1].get<double>(),j[2].get<double>(),j[3].get<double>()};}
json array(const btVector3& v){return {v.x(),v.y(),v.z()};}
class Dantzig : public btDantzigSolver {public: Dantzig(){m_acceptableUpperLimitSolution=btScalar(1e30);}};
class PrescribedWorld : public btDiscreteDynamicsWorld {
public:
 bool start_phase=false;
 using btDiscreteDynamicsWorld::btDiscreteDynamicsWorld;
protected:
 void saveKinematicState(btScalar h) override {if(!start_phase)btDiscreteDynamicsWorld::saveKinematicState(h);}
};
struct Body {std::unique_ptr<btRigidBody> rb; bool kin; btVector3 pos,v,w; btQuaternion q; double radius; json schedule;};
int main(){try{
 static_assert(sizeof(btScalar)==8,"Double precision required");
 json in; std::cin>>in;
 btDefaultCollisionConfiguration config; btCollisionDispatcher dispatch(&config); btDbvtBroadphase broad;
 Dantzig dantzig;NormalFirstDantzig normal;
 bool normal_solver=in.at("solver")=="normal_coupled";
 bool compact=in.value("preassembly_elimination",true);
 RecordedMLCP regular_mlcp(&dantzig),post_normal_mlcp(&normal);NormalMLCP normal_mlcp(&normal);
 bool coulomb_solver=in.at("solver")=="coulomb";CoulombMLCP coulomb_mlcp(&dantzig);
 coulomb_mlcp.recovery_enabled=in.value("contact_recovery",true);
 coulomb_mlcp.tolerance=in.value("contact_tolerance_m_s",1e-8);coulomb_mlcp.contact_slop_m=in.value("contact_slop_m",1e-9);
 if(in.contains("rejected_contact_path")){
  std::string path=in.at("rejected_contact_path");
  coulomb_mlcp.rejection_observer=[&,path](const auto& A,const auto& b,const auto& p,const auto& lo,const auto& hi,const auto& dep,const char* phase,double residual,double h){
   json matrix=json::array(),rhs=json::array(),lower=json::array(),upper=json::array(),dependencies=json::array();
   for(int i=0;i<b.rows();i++){json row=json::array();for(int j=0;j<b.rows();j++)row.push_back(A(i,j));matrix.push_back(row);rhs.push_back(b[i]);lower.push_back(lo[i]);upper.push_back(hi[i]);dependencies.push_back(dep[i]);}
   json snapshot={{"schema","circular-coulomb-rejection-v1"},{"phase",phase},{"A",matrix},{"b",rhs},{"p",p},{"lo",lower},{"hi",upper},{"dependencies",dependencies},{"residual_m_s",residual},{"tolerance_m_s",coulomb_mlcp.tolerance},{"internal_dt_s",h},{"iteration_budget",in.at("iterations")}};
   std::ofstream file(path);if(!file)throw std::runtime_error("Cannot write rejected contact diagnostic");file<<snapshot.dump()<<"\n";file.close();if(!file)throw std::runtime_error("Failed writing rejected contact diagnostic");
  };
 }

 btMLCPSolver& mlcp=normal_solver?(compact?static_cast<btMLCPSolver&>(normal_mlcp):static_cast<btMLCPSolver&>(post_normal_mlcp)):(coulomb_solver?static_cast<btMLCPSolver&>(coulomb_mlcp):static_cast<btMLCPSolver&>(regular_mlcp)); btSequentialImpulseConstraintSolver sequential;
 bool adaptive=in.at("solver")=="adaptive",coupled=in.at("solver")=="coupled"||normal_solver||coulomb_solver;
 int dense_steps=0,fast_steps=0,dwell=0,contacts_previous=0;double residual_previous=0;
 PrescribedWorld world(&dispatch,&broad,coupled?static_cast<btConstraintSolver*>(&mlcp):static_cast<btConstraintSolver*>(&sequential),&config);
 world.start_phase=in.value("kinematic_contact_phase",std::string("end"))=="start";
 world.setGravity(vec(in.at("gravity")));
 auto& info=world.getSolverInfo(); info.m_numIterations=in.at("iterations"); info.m_splitImpulse=true;
 info.m_splitImpulsePenetrationThreshold=0; info.m_restitutionVelocityThreshold=0;
 if(in.value("position_stabilization",std::string("split"))=="velocity_only"){info.m_splitImpulse=false;info.m_erp=0;info.m_erp2=0;}
 info.m_solverMode=SOLVER_USE_WARMSTARTING|SOLVER_USE_2_FRICTION_DIRECTIONS|SOLVER_DISABLE_VELOCITY_DEPENDENT_FRICTION_DIRECTION;
 std::vector<std::unique_ptr<btCollisionShape>> shapes; std::vector<Body> bodies;
 for(const auto& b:in.at("bodies")){
  auto compound=std::make_unique<btCompoundShape>(); compound->setMargin(0);
  for(const auto& s:b.at("shapes")){
   std::unique_ptr<btCollisionShape> shape;
   if(s.at("kind")=="sphere") shape=std::make_unique<btSphereShape>(s.at("radius").get<double>());
   else if(s.at("kind")=="box") {shape=std::make_unique<btBoxShape>(vec(s.at("half_extents"))); shape->setMargin(in.at("margin_m").get<double>());}
   else {auto hull=std::make_unique<btConvexHullShape>(); for(const auto& p:s.at("vertices"))hull->addPoint(vec(p),false);hull->recalcLocalAabb(); hull->setMargin(in.at("margin_m").get<double>());shape=std::move(hull);}
   btTransform local(quat(s.at("orientation")),vec(s.at("center")));
   compound->addChildShape(local,shape.get()); shapes.push_back(std::move(shape));
  }
  double mass=b.at("mass"); bool kin=b.at("type")=="kinematic";
  btRigidBody::btRigidBodyConstructionInfo ci(mass,nullptr,compound.get(),vec(b.at("principal_inertia")));
  auto rb=std::make_unique<btRigidBody>(ci); rb->setCenterOfMassTransform(btTransform(quat(b.at("orientation")),vec(b.at("position"))));
  rb->setInterpolationWorldTransform(rb->getWorldTransform());
  rb->setLinearVelocity(vec(b.at("velocity")));rb->setAngularVelocity(vec(b.at("omega")));
  rb->setFriction(b.at("friction").get<double>()); rb->setRestitution(b.at("restitution").get<double>());
  rb->setActivationState(DISABLE_DEACTIVATION);
  if(kin){rb->setCollisionFlags((rb->getCollisionFlags() & ~btCollisionObject::CF_STATIC_OBJECT)|btCollisionObject::CF_KINEMATIC_OBJECT);}
  rb->setUserIndex(static_cast<int>(bodies.size())); world.addRigidBody(rb.get());
  bodies.push_back({std::move(rb),kin,vec(b.at("position")),vec(b.at("velocity")),vec(b.at("omega")),quat(b.at("orientation")),b.at("radius"),b.at("schedule")});
  shapes.push_back(std::move(compound));
 }
 json states=json::array(),times=json::array(),updates=json::array();
 auto record=[&](double t){json frame=json::array();for(auto& b:bodies){auto tr=b.rb->getWorldTransform();auto q=tr.getRotation();json row=array(tr.getOrigin());for(double x:{q.x(),q.y(),q.z(),q.w()})row.push_back(x);for(auto v:{b.rb->getLinearVelocity(),b.rb->getAngularVelocity()})for(int k=0;k<3;k++)row.push_back(v[k]);frame.push_back(row);}states.push_back(frame);times.push_back(t);};
 record(0); double dt=in.at("dt"), feature=in.at("minimum_feature_m"), fraction=in.at("travel_fraction"),work=0,maxpenetration=0,maxresidual=0;
 double surface_excess=-BT_LARGE_FLOAT;
 int frames=in.at("frames"),primary=in.at("primary_steps"),total=0;
 auto start=std::chrono::steady_clock::now();
 for(int f=0;f<frames;f++){
  double left=dt;int count=0;
  while(left>dt*1e-12){
   double t=(f+1)*dt-left, speed=0;
   for(auto& b:bodies){if(b.kin){for(const auto& cmd:b.schedule)if(cmd.at("time_s").get<double>()<=t+1e-12){b.v=vec(cmd.at("velocity"));b.w=vec(cmd.at("omega"));}}
    auto v=b.kin?b.v:b.rb->getLinearVelocity();auto w=b.kin?b.w:b.rb->getAngularVelocity();speed=std::max(speed,static_cast<double>(v.length()+w.length()*b.radius));}
   // Relative translation <=2*maxspeed; include angular tip speed and gravity.
   double h=std::min(left,dt/primary);
   double accel=world.getGravity().length();
   if(fraction>0)h=std::min(h,fraction*feature/(2*speed+std::sqrt(2*accel*fraction*feature)+1e-30));
   for(auto& b:bodies)if(b.kin)for(const auto& cmd:b.schedule){double event=cmd.at("time_s").get<double>();if(event>t+1e-12)h=std::min(h,event-t);}
   if(h<1e-12||++count>100000)throw std::runtime_error("Travel guard exhausted; reject rather than tunnel");
   // A schedule change must reach contact discovery on this update, including
   // start-phase integration where saveKinematicState is intentionally skipped.
   for(auto& b:bodies)if(b.kin){b.rb->setLinearVelocity(b.v);b.rb->setAngularVelocity(b.w);}
   if(!world.start_phase){
   for(auto& b:bodies)if(b.kin){b.rb->setInterpolationWorldTransform(b.rb->getWorldTransform());b.pos+=b.v*h; double w=b.w.length();if(w>0){b.q=btQuaternion(b.w/w,w*h)*b.q;b.q.normalize();}b.rb->setWorldTransform(btTransform(b.q,b.pos));b.rb->setLinearVelocity(b.v);b.rb->setAngularVelocity(b.w);world.updateSingleAabb(b.rb.get());}
   }
   if(adaptive){
    if(contacts_previous>=12||residual_previous>.01)dwell=24;
    coupled=dwell>0;if(dwell>0)dwell--;
    world.setConstraintSolver(coupled?static_cast<btConstraintSolver*>(&mlcp):static_cast<btConstraintSolver*>(&sequential));
    info.m_numIterations=coupled?in.at("iterations").get<int>():8;
   }
   world.stepSimulation(h,0,h); total++;
   if(coupled)dense_steps++;else fast_steps++;
   contacts_previous=0;residual_previous=0;
   for(int k=0;k<dispatch.getNumManifolds();k++){
    auto m=dispatch.getManifoldByIndexInternal(k); auto a=static_cast<const btRigidBody*>(m->getBody0());auto b=static_cast<const btRigidBody*>(m->getBody1());
    for(int c=0;c<m->getNumContacts();c++){auto& p=m->getContactPoint(c);
     if(p.getAppliedImpulse()>0)contacts_previous++;
     maxpenetration=std::max(maxpenetration,-static_cast<double>(p.getDistance()));
     auto va=a->getVelocityInLocalPoint(p.getPositionWorldOnA()-a->getCenterOfMassPosition());auto vb=b->getVelocityInLocalPoint(p.getPositionWorldOnB()-b->getCenterOfMassPosition());
     if(p.getDistance()<=0){residual_previous=std::max(residual_previous,std::max(0.,-static_cast<double>((va-vb).dot(p.m_normalWorldOnB))));maxresidual=std::max(maxresidual,residual_previous);}
     btVector3 impulse=p.m_normalWorldOnB*p.getAppliedImpulse()+p.m_lateralFrictionDir1*p.m_appliedImpulseLateral1+p.m_lateralFrictionDir2*p.m_appliedImpulseLateral2;
     if(bodies[a->getUserIndex()].kin)work-=impulse.dot(va);
     if(bodies[b->getUserIndex()].kin)work+=impulse.dot(vb);
    }
   }
   if(world.start_phase){
   for(auto& b:bodies)if(b.kin){b.rb->setInterpolationWorldTransform(b.rb->getWorldTransform());b.pos+=b.v*h; double w=b.w.length();if(w>0){b.q=btQuaternion(b.w/w,w*h)*b.q;b.q.normalize();}b.rb->setWorldTransform(btTransform(b.q,b.pos));b.rb->setLinearVelocity(b.v);b.rb->setAngularVelocity(b.w);world.updateSingleAabb(b.rb.get());}
   }
   if(in.contains("container_half")){
    auto box=bodies[0].rb->getWorldTransform();
    btMatrix3x3 axes(box.getRotation()*quat(in.at("container_frame_rotation")));
    auto half=vec(in.at("container_half"));
    for(size_t i=1;i<bodies.size();i++){
     auto tr=bodies[i].rb->getWorldTransform();auto compound=static_cast<btCompoundShape*>(bodies[i].rb->getCollisionShape());
     for(int axis=0;axis<3;axis++)for(int sign:{-1,1}){
      auto direction=axes.getColumn(axis)*sign;
      for(int child=0;child<compound->getNumChildShapes();child++){
       auto local=compound->getChildTransform(child);auto convex=static_cast<btConvexShape*>(compound->getChildShape(child));
       auto local_direction=local.getBasis().transpose()*tr.getBasis().transpose()*direction;
       auto point=tr*(local*convex->localGetSupportingVertex(local_direction));
       surface_excess=std::max(surface_excess,static_cast<double>((point-box.getOrigin()).dot(direction)-half[axis]));
      }
     }
    }
   }
   left-=h;
  }
  updates.push_back(count);record((f+1)*dt);
 }
 double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 json out={{"states",states},{"times",times},{"updates",updates},{"step_s",seconds},{"collision_updates",total},{"boundary_work_J",work},{"max_contact_penetration_m",maxpenetration},{"max_closing_contact_speed_m_s",maxresidual},{"coupled_fallbacks",mlcp.getNumFallbacks()},{"coupled_updates",dense_steps},{"sequential_updates",fast_steps},{"scalar_precision","float64"},{"normal_qp_solves",normal.normal_solves},{"normal_qp_rejections",normal.normal_rejections},{"normal_matrix_rows_max",normal_mlcp.rows_max},{"eliminated_tangent_rows_max",normal_mlcp.removed_rows_max}};
 int matrix_rows=normal_solver?(compact?normal_mlcp.rows_max:post_normal_mlcp.rows_max):(coulomb_solver?coulomb_mlcp.rows_max:regular_mlcp.rows_max);
 out["tangent_gyro_correction_max_m_s"]=coulomb_mlcp.gyro_correction_max;out["coulomb_newton_steps"]=coulomb_mlcp.stats.newton_steps;out["position_iterative_solves"]=coulomb_mlcp.position_stats.solves;out["coulomb_solves"]=coulomb_mlcp.stats.solves;out["coulomb_fast_solves"]=coulomb_mlcp.stats.fast_solves;out["coulomb_sweeps_max"]=coulomb_mlcp.stats.sweeps_max;out["coulomb_residual_max_m_s"]=coulomb_mlcp.stats.residual_max;out["coulomb_passive_change_max_J"]=coulomb_mlcp.stats.passive_change_max;
 out["mobility_rows_max"]=matrix_rows;out["mobility_matrix_bytes_max"]=8ULL*matrix_rows*matrix_rows;
 if(in.contains("container_half"))out["max_container_surface_excess_m"]=surface_excess;
 for(auto& b:bodies)world.removeRigidBody(b.rb.get());
 out["coulomb_polish_solves"]=coulomb_mlcp.stats.polish_solves;out["coulomb_opposing_restarts"]=coulomb_mlcp.stats.opposing_restarts;out["coulomb_polish_svd_calls"]=coulomb_mlcp.stats.polish_svd_calls;out["coulomb_polish_budget_rejections"]=coulomb_mlcp.stats.polish_budget_rejections;out["coulomb_polish_svd_rejections"]=coulomb_mlcp.stats.polish_svd_rejections;out["coulomb_polish_steps"]=coulomb_mlcp.stats.polish_steps;out["coulomb_gauge_restarts"]=coulomb_mlcp.stats.gauge_restarts;out["coulomb_cold_restarts"]=coulomb_mlcp.stats.cold_restarts;std::cout<<out.dump()<<'\n';
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
