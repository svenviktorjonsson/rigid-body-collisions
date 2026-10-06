#include <cstdio>
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
#include "position_geometry.h"
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
 std::string point_policy=in.value("contact_point_policy",coulomb_solver?std::string("shared"):std::string("separate"));
 if((point_policy!="shared"&&point_policy!="separate")||(point_policy=="shared"&&!coulomb_solver))throw std::runtime_error("Invalid contact point policy for solver");
 coulomb_mlcp.shared_contact_point=point_policy=="shared";
 coulomb_mlcp.recovery_enabled=in.value("contact_recovery",true);
 coulomb_mlcp.early_component_recovery=in.value("early_component_recovery",false);
 if(coulomb_mlcp.early_component_recovery&&(!coulomb_solver||!coulomb_mlcp.recovery_enabled||!coulombLapackRecoveryEnabled()))throw std::runtime_error("Early component schedule requires Coulomb and enabled compiled recovery");
 const std::string position_policy=in.value("position_stabilization",std::string("split"));
 coulomb_mlcp.translation_clearance=position_policy=="split_translation_gap";
 coulomb_mlcp.translation_combined=position_policy=="split_translation_combined";
 coulomb_mlcp.translation_split=position_policy=="split_translation"||coulomb_mlcp.translation_clearance||coulomb_mlcp.translation_combined;
 if(coulomb_mlcp.translation_split&&!coulomb_solver)throw std::runtime_error("Translation-only split requires coulomb solver");
 coulomb_mlcp.tolerance=in.value("contact_tolerance_m_s",1e-8);coulomb_mlcp.contact_slop_m=in.value("contact_slop_m",1e-9);

 auto projection_policy=[&](){const auto& p=coulomb_mlcp.stats;return json{
  {"compiled",coulombLapackRecoveryEnabled()},
  {"enabled",coulomb_solver&&coulomb_mlcp.recovery_enabled&&coulombLapackRecoveryEnabled()},
  {"stage","after_all_existing_pipeline_failure"},
  {"max_rows",64},{"max_svd_calls",2048},{"max_iteration_steps",2048},
  {"attempts",p.projection_attempts},{"solves",p.projection_solves},{"declines",p.projection_declines},
  {"svd_calls",p.projection_svd_calls},{"iteration_steps",p.projection_iteration_steps},{"newton_steps",p.projection_newton_steps},
  {"claim","bounded original-law numerical search; no trajectory qualification"}};};
 auto terminal_polish_policy=[&](){const auto&s=coulomb_mlcp.stats;const auto&t=s.terminal_polish;return json{{"compiled",coulombLapackRecoveryEnabled()},{"enabled",coulomb_solver&&coulomb_mlcp.recovery_enabled&&coulombLapackRecoveryEnabled()},{"stage","after ALL earlier lanes decline"},{"max_full_rows",4096},{"max_search_component_rows",192},{"max_iteration_sweeps_per_failed_component",256},{"max_polish_svd_calls_per_failed_component",256},{"attempts",s.terminal_polish_attempts},{"solves",s.terminal_polish_solves},{"declines",s.terminal_polish_declines},{"iteration_steps",t.iteration_steps},{"svd_calls",t.svd_calls},{"polish_steps",t.polish_steps},{"final_gate","original eager-cone/bounds/residual/passivity before atomic application"}};};
 auto reduced_mobility_policy=[&](){const auto&s=coulomb_mlcp.stats;const auto&t=s.reduced_mobility;return json{{"compiled",coulombLapackRecoveryEnabled()},{"enabled",coulomb_solver&&coulomb_mlcp.recovery_enabled&&coulombLapackRecoveryEnabled()},{"stage","after ALL earlier lanes including full-component mobility continuation decline"},{"max_full_rows",4096},{"minimum_failed_component_rows",193},{"max_search_rows",192},{"max_support_passes_per_failed_component",8},{"max_stages_per_pass",17},{"max_iteration_steps_per_stage",2048},{"max_svd_calls_per_stage",2048},{"inactive_candidate_impulses",0},{"support_growth","only original violated inactive normal constraints"},{"attempts",s.reduced_mobility_attempts},{"solves",s.reduced_mobility_solves},{"declines",s.reduced_mobility_declines},{"support_passes",t.support_passes},{"reduced_rows_max",t.reduced_rows_max},{"stage_attempts",t.stage_attempts},{"svd_calls",t.svd_calls},{"physical_mobility_changed",false},{"final_gate","all full original components and global production zero-budget eager-cone/bounds/residual/passivity before body application"},{"claim","bounded numerical search only; no world/reference/performance qualification"}};};
 auto mobility_policy=[&](){const auto&s=coulomb_mlcp.stats;return json{{"compiled",coulombLapackRecoveryEnabled()},{"enabled",coulomb_solver&&coulomb_mlcp.recovery_enabled&&coulombLapackRecoveryEnabled()},{"stage","after ALL existing lanes including null-traction seeds decline"},{"max_full_rows",4096},{"max_component_rows",192},{"row_cap_scope","nonlinear search only; already accepted larger components pass original gate without new search"},{"max_search_component_rows",192},{"max_stages_per_failed_component",17},{"max_iteration_steps_per_stage",2048},{"max_svd_calls_per_stage",2048},{"search_diagonal_levels",json::array({.1,.03,.01,.003,.001,.0003,.0001,.00003,.00001,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,0.})},{"attempts",s.mobility_attempts},{"solves",s.mobility_solves},{"declines",s.mobility_declines},{"components",s.mobility.components},{"largest_component_rows",s.mobility.largest_rows},{"stage_attempts",s.mobility.stage_attempts},{"stage_accepts",s.mobility.stage_accepts},{"iteration_steps",s.mobility.iteration_steps},{"svd_calls",s.mobility.svd_calls},{"physical_mobility_changed",false},{"final_gate","original production zero-budget eager cone projection, bounds, residual and finite passivity"},{"claim","bounded numerical search; no reference or performance qualification"}};};
 auto null_seed_policy=[&](){json result={{"compiled",coulombLapackRecoveryEnabled()},{"enabled",coulomb_solver&&coulomb_mlcp.recovery_enabled&&coulombLapackRecoveryEnabled()},{"stage","after existing projection tail decline"},{"max_full_rows",4096},{"max_component_rows",192},{"max_seeds_per_component",6},{"max_searches_per_component",7},{"max_iteration_steps_per_search",2048},{"max_svd_calls_per_search",2048},{"null_jacobi_sweeps_per_factorization",64},{"seed_svd_calls_per_component",7},{"graph_coupling_threshold",0},{"final_gate","original production eager cone projection, bounds, residual and passivity"},{"claim","bounded numerical search; no trajectory or performance qualification"}};
#ifdef SPATIAL_LAPACK_RECOVERY
 const auto& s=coulomb_mlcp.stats;const auto& t=s.null_seed;result.update({{"attempts",s.null_seed_attempts},{"solves",s.null_seed_solves},{"declines",s.null_seed_declines},{"components",t.components},{"largest_component_rows",t.largest_rows},{"component_cap_rejections",t.cap_rejections},{"seed_attempts",t.seed_attempts},{"null_svd_calls",t.null_svd_calls},{"seed_svd_calls",t.seed_svd_calls},{"svd_calls",t.svd_calls},{"iteration_steps",t.iteration_steps},{"newton_steps",t.newton_steps},{"seed_response_change_max_m_s",t.seed_response_change_max}});
#endif
 return result;};
 json position_geometry_snapshot;
 if(in.contains("rejected_contact_path")){
  std::string path=in.at("rejected_contact_path");
  coulomb_mlcp.position_geometry_observer=[&](const auto& rows,const auto& solver_bodies,const auto& normals,const auto& split,double h){
   position_geometry_snapshot=positionGeometry(rows,solver_bodies,normals,split,h);
   position_geometry_snapshot["wire_bodies"]=in.at("bodies");
   position_geometry_snapshot["margin_m"]=in.value("margin_m",0.);
   if(coulomb_mlcp.translation_combined){
    auto values=[](const btVectorXu& v){json result=json::array();for(int i=0;i<v.rows();i++)result.push_back(v[i]);return result;};
    position_geometry_snapshot["accepted_physical_impulses"]=values(coulomb_mlcp.translation_accepted_physical_impulses);
    position_geometry_snapshot["accepted_physical_normal_rates_m_s"]=values(coulomb_mlcp.translation_physical_rates);
    position_geometry_snapshot["desired_combined_normal_rates_m_s"]=values(coulomb_mlcp.translation_desired_rates);
    json physical=json::array();
    for(int i=0;i<coulomb_mlcp.translation_final_physical_bodies.size();i++){
     const auto& body=coulomb_mlcp.translation_final_physical_bodies[i];
     physical.push_back({{"solver_body_id",i},{"has_original_body",body.m_originalBody!=nullptr},
      {"linear_velocity_m_s",array(body.m_linearVelocity+body.m_deltaLinearVelocity+body.m_externalForceImpulse)},
      {"angular_velocity_rad_s",array(body.m_angularVelocity+body.m_deltaAngularVelocity+body.m_externalTorqueImpulse)}});
    }
    position_geometry_snapshot["accepted_physical_solver_bodies"]=physical;
    position_geometry_snapshot["target_convention"]="desired combined normal rate minus accepted final physical normal rate, including angular motion";
   }
  };
  coulomb_mlcp.rejection_observer=[&,path](const auto& A,const auto& b,const auto& p,const auto& lo,const auto& hi,const auto& dep,const char* phase,double residual,double h){
   json matrix=json::array(),rhs=json::array(),lower=json::array(),upper=json::array(),dependencies=json::array();
   for(int i=0;i<b.rows();i++){json row=json::array();for(int j=0;j<b.rows();j++)row.push_back(A(i,j));matrix.push_back(row);rhs.push_back(b[i]);lower.push_back(lo[i]);upper.push_back(hi[i]);dependencies.push_back(dep[i]);}
   const bool normal_only=std::string(phase)=="position_translation";
   json snapshot={{"schema",normal_only?"normal-only-position-rejection-v1":"circular-coulomb-rejection-v1"},{"phase",phase},{"A",matrix},{"b",rhs},{"p",p},{"lo",lower},{"hi",upper},{"dependencies",dependencies},{"residual_m_s",residual},{"tolerance_m_s",coulomb_mlcp.tolerance},{"internal_dt_s",h},{"iteration_budget",in.at("iterations")}};
   if(normal_only&&!position_geometry_snapshot.is_null()){
    std::ofstream geometry(path+".geometry.json");if(!geometry)throw std::runtime_error("Cannot write position geometry diagnostic");
    geometry<<position_geometry_snapshot.dump()<<"\n";geometry.close();if(!geometry)throw std::runtime_error("Failed writing position geometry diagnostic");
   }
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
   else {auto hull=std::make_unique<btConvexHullShape>(); for(const auto& p:s.at("vertices"))hull->addPoint(vec(p),false);hull->setMargin(in.at("margin_m").get<double>()); hull->recalcLocalAabb();shape=std::move(hull);}
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
 auto checkpoint=[&](){
  if(!in.contains("progress_checkpoint_path"))return;
  const std::string path=in.at("progress_checkpoint_path"),temporary=path+".tmp";
  json progress={{"schema","native-spatial-progress-v1"},{"states_backend",states},{"times",times},
    {"orientation_frame","backend principal inertia axes"},{"wire_bodies",in.at("bodies")},
    {"boundary_work_J",work},{"completed_output_frames",states.size()-1},{"expected_output_frames",frames},
    {"complete",states.size()==static_cast<size_t>(frames+1)},{"collision_updates",total},
    {"coulomb_residual_max_m_s",coulomb_mlcp.stats.residual_max},
    {"max_container_surface_excess_m",surface_excess}};
  progress["translation_split_solves"]=coulomb_mlcp.translation_split_solves;
  progress["translation_split_residual_max_m_s"]=coulomb_mlcp.translation_split_residual_max;
  progress["projection_tail_policy"]=projection_policy();
  progress["null_traction_seed_policy"]=null_seed_policy();
  progress["mobility_continuation_policy"]=mobility_policy();
  progress["reduced_mobility_continuation_policy"]=reduced_mobility_policy();
  progress["terminal_component_polish_policy"]=terminal_polish_policy();
  progress["position_normal_search_policy"]={{"enabled",coulomb_mlcp.translation_split},{"max_normal_rows",512},{"max_active_states",128},{"stage","before projected position fallback"},{"final_gate","unchanged absolute original normal projection and bounds"}};
  progress["translation_pose_ledger_updates"]=coulomb_mlcp.translation_pose_ledger_updates;
  progress["translation_pose_displacement_max_m"]=coulomb_mlcp.translation_pose_displacement_max_m;
  progress["translation_pose_potential_change_J"]=coulomb_mlcp.translation_pose_potential_change_J;
  progress["translation_pose_absolute_potential_change_J"]=coulomb_mlcp.translation_pose_absolute_potential_change_J;
  progress["translation_pose_orbital_change_kg_m2_s"]=array(coulomb_mlcp.translation_pose_orbital_change);
  progress["translation_pose_absolute_orbital_change_kg_m2_s"]=coulomb_mlcp.translation_pose_absolute_orbital_change;
  std::ofstream file(temporary);if(!file)throw std::runtime_error("Cannot write progress checkpoint");
  file<<progress.dump()<<"\n";file.close();if(!file||std::rename(temporary.c_str(),path.c_str())!=0)
   throw std::runtime_error("Cannot publish progress checkpoint");
 };
 checkpoint();
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
     auto pointA=p.getPositionWorldOnA(),pointB=p.getPositionWorldOnB();
     if(coulomb_solver&&coulomb_mlcp.shared_contact_point){
      auto common=a->getInvMass()>0&&b->getInvMass()==0?pointA:(b->getInvMass()>0&&a->getInvMass()==0?pointB:pointA*.5+pointB*.5);
      pointA=pointB=common;
     }
     auto va=a->getVelocityInLocalPoint(pointA-a->getCenterOfMassPosition());auto vb=b->getVelocityInLocalPoint(pointB-b->getCenterOfMassPosition());
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
  updates.push_back(count);record((f+1)*dt);checkpoint();
 }
 double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
 json out={{"states",states},{"times",times},{"updates",updates},{"step_s",seconds},{"collision_updates",total},{"boundary_work_J",work},{"max_contact_penetration_m",maxpenetration},{"max_closing_contact_speed_m_s",maxresidual},{"coupled_fallbacks",mlcp.getNumFallbacks()},{"coupled_updates",dense_steps},{"sequential_updates",fast_steps},{"scalar_precision","float64"},{"normal_qp_solves",normal.normal_solves},{"normal_qp_rejections",normal.normal_rejections},{"normal_matrix_rows_max",normal_mlcp.rows_max},{"eliminated_tangent_rows_max",normal_mlcp.removed_rows_max}};
 int matrix_rows=normal_solver?(compact?normal_mlcp.rows_max:post_normal_mlcp.rows_max):(coulomb_solver?coulomb_mlcp.rows_max:regular_mlcp.rows_max);
 out["tangent_gyro_correction_max_m_s"]=coulomb_mlcp.gyro_correction_max;out["coulomb_newton_steps"]=coulomb_mlcp.stats.newton_steps;out["position_iterative_solves"]=coulomb_mlcp.position_stats.solves;out["coulomb_solves"]=coulomb_mlcp.stats.solves;out["coulomb_fast_solves"]=coulomb_mlcp.stats.fast_solves;out["coulomb_sweeps_max"]=coulomb_mlcp.stats.sweeps_max;out["coulomb_residual_max_m_s"]=coulomb_mlcp.stats.residual_max;out["coulomb_passive_change_max_J"]=coulomb_mlcp.stats.passive_change_max;
 out["shared_contact_rows"]=coulomb_mlcp.shared_point_rows;out["shared_contact_transport_max_m"]=coulomb_mlcp.shared_point_transport_max_m;out["contact_point_policy"]=point_policy;
 out["position_normal_search_policy"]={{"enabled",coulomb_mlcp.translation_split},{"max_normal_rows",512},{"max_active_states",128},{"stage","before projected position fallback"},{"final_gate","unchanged absolute original normal projection and bounds"}};
 out["translation_split_solves"]=coulomb_mlcp.translation_split_solves;out["translation_split_residual_max_m_s"]=coulomb_mlcp.translation_split_residual_max;
 out["coulomb_continuation_solves"]=coulomb_mlcp.stats.continuation_solves;
 out["coulomb_iteration_sweeps_total"]=coulomb_mlcp.stats.iteration_sweeps_total;
 out["lapack_contact_recovery_compiled"]=coulombLapackRecoveryEnabled();
 out["projection_tail_policy"]=projection_policy();
 out["null_traction_seed_policy"]=null_seed_policy();
 out["mobility_continuation_policy"]=mobility_policy();
 out["reduced_mobility_continuation_policy"]=reduced_mobility_policy();
 out["terminal_component_polish_policy"]=terminal_polish_policy();
 out["translation_pose_ledger_updates"]=coulomb_mlcp.translation_pose_ledger_updates;
 out["translation_pose_displacement_max_m"]=coulomb_mlcp.translation_pose_displacement_max_m;
 out["translation_pose_potential_change_J"]=coulomb_mlcp.translation_pose_potential_change_J;
 out["translation_pose_absolute_potential_change_J"]=coulomb_mlcp.translation_pose_absolute_potential_change_J;
 out["translation_pose_orbital_change_kg_m2_s"]=array(coulomb_mlcp.translation_pose_orbital_change);
 out["translation_pose_absolute_orbital_change_kg_m2_s"]=coulomb_mlcp.translation_pose_absolute_orbital_change;
 if(coulomb_mlcp.early_component_recovery){
  const auto& e=coulomb_mlcp.stats;
  out["early_component_policy"]={{"enabled",true},{"seed","actual first256 rejected PGS"},
   {"attempts",e.early_component_attempts},{"solves",e.early_component_solves},{"declines",e.early_component_declines},
   {"helper_calls",e.early_component_helper_calls},{"skipped_components",e.early_component_skipped_components},
   {"component_cap_rejections",e.early_component_cap_rejections},{"passes",e.early_component_passes},
   {"largest_rows",e.early_component_largest_rows},{"expanded_contacts",e.early_component_expanded_contacts},
   {"svd_calls",e.early_component_svd_calls},{"iteration_steps",e.early_component_iteration_steps},
   {"pressure_svd_calls",e.early_component_pressure_svd_calls},{"pressure_attempts",e.early_component_pressure_attempts},
   {"pivot_calls",e.early_component_pivot_calls},
   {"extra_budget","early and later tail have independent fresh caps; aggregate support counters include both"},
   {"claim","optional original-law schedule; accepted root and trajectory may differ"}};
 }
 out["coulomb_support_solves"]=coulomb_mlcp.stats.support_solves;
 out["coulomb_support_helper_calls"]=coulomb_mlcp.stats.support_helper_calls;
 out["coulomb_support_skipped_components"]=coulomb_mlcp.stats.support_skipped_components;
 out["coulomb_support_component_cap_rejections"]=coulomb_mlcp.stats.support_component_cap_rejections;
 out["coulomb_support_passes"]=coulomb_mlcp.stats.support_passes;
 out["coulomb_support_largest_rows"]=coulomb_mlcp.stats.support_largest_rows;
 out["coulomb_support_expanded_contacts"]=coulomb_mlcp.stats.support_expanded_contacts;
 out["coulomb_support_svd_calls"]=coulomb_mlcp.stats.support_svd_calls;
 out["coulomb_support_iteration_steps"]=coulomb_mlcp.stats.support_iteration_steps;
 out["coulomb_support_pressure_svd_calls"]=coulomb_mlcp.stats.support_pressure_svd_calls;
 out["coulomb_support_pivot_calls"]=coulomb_mlcp.stats.support_pivot_calls;
 out["coulomb_supplemental_solves"]=coulomb_mlcp.stats.supplemental_solves;
 out["coulomb_supplemental_svd_calls"]=coulomb_mlcp.stats.supplemental_svd_calls;
 out["coulomb_supplemental_iteration_steps"]=coulomb_mlcp.stats.supplemental_iteration_steps;
 out["coulomb_supplemental_damped_steps"]=coulomb_mlcp.stats.supplemental_damped_steps;
 out["coulomb_supplemental_pressure_svd_calls"]=coulomb_mlcp.stats.supplemental_pressure_svd_calls;
 out["coulomb_supplemental_pressure_attempts"]=coulomb_mlcp.stats.supplemental_pressure_attempts;
 out["coulomb_supplemental_pivot_calls"]=coulomb_mlcp.stats.supplemental_pivot_calls;
 out["coulomb_supplemental_projector_calls"]=coulomb_mlcp.stats.supplemental_projector_calls;
 out["coulomb_supplemental_restarts"]=coulomb_mlcp.stats.supplemental_restarts;
 out["coulomb_supplemental_null_response_max_m_s"]=coulomb_mlcp.stats.supplemental_null_response_max;
 const auto& active=coulomb_mlcp.stats.active;out["coulomb_active_solves"]=coulomb_mlcp.stats.active_solves;
 out["coulomb_active_subset_passes"]=active.passes;out["coulomb_active_mode_guesses"]=active.mode_guesses;
 out["coulomb_active_expanded_contacts"]=active.expanded_contacts;
 out["coulomb_active_svd_calls"]=active.search.svd_calls;out["coulomb_active_pressure_svd_calls"]=active.search.pressure_svd_calls;
 out["coulomb_active_damped_steps"]=active.search.damped_steps;out["coulomb_active_pivot_calls"]=active.search.normal_pivot_attempts;
 out["position_active_solves"]=coulomb_mlcp.position_stats.active_solves;
 out["position_active_svd_calls"]=coulomb_mlcp.position_stats.active.search.svd_calls;
 out["position_active_pressure_svd_calls"]=coulomb_mlcp.position_stats.active.search.pressure_svd_calls;
 out["coulomb_pressure_solves"]=coulomb_mlcp.stats.pressure_solves;out["coulomb_pressure_svd_calls"]=coulomb_mlcp.stats.pressure.svd_calls;
 out["position_pressure_solves"]=coulomb_mlcp.position_stats.pressure_solves;out["position_pressure_svd_calls"]=coulomb_mlcp.position_stats.pressure.svd_calls;
 out["coulomb_null_pressure_solves"]=coulomb_mlcp.stats.null_pressure.solves;
 out["coulomb_null_pressure_states"]=coulomb_mlcp.stats.null_pressure.states;
 out["coulomb_null_pressure_svd_calls"]=coulomb_mlcp.stats.null_pressure.svd_calls;
 out["coulomb_null_pressure_boundary_moves"]=coulomb_mlcp.stats.null_pressure.null_steps;
 out["coulomb_null_trial_response_max_m_s"]=coulomb_mlcp.stats.null_pressure.maximum_null_velocity_change;
 out["position_null_pressure_solves"]=coulomb_mlcp.position_stats.null_pressure.solves;
 out["position_null_pressure_states"]=coulomb_mlcp.position_stats.null_pressure.states;
 out["position_null_pressure_svd_calls"]=coulomb_mlcp.position_stats.null_pressure.svd_calls;
 out["position_null_pressure_boundary_moves"]=coulomb_mlcp.position_stats.null_pressure.null_steps;
 out["position_null_trial_response_max_m_s"]=coulomb_mlcp.position_stats.null_pressure.maximum_null_velocity_change;
 const auto& continuation=coulomb_mlcp.stats.continuation;
 out["coulomb_continuation_attempts"]=continuation.attempts;out["coulomb_continuation_stages"]=continuation.stages;
 out["coulomb_continuation_svd_calls"]=continuation.svd_calls;out["coulomb_continuation_damped_steps"]=continuation.damped_steps;
 out["coulomb_continuation_budget_rejections"]=continuation.budget_rejections;
 out["coulomb_continuation_newton_steps"]=continuation.newton_steps;
 out["coulomb_continuation_normal_qp_guides"]=continuation.normal_qp_guides;
 out["coulomb_continuation_normal_pivot_attempts"]=continuation.normal_pivot_attempts;
 out["coulomb_continuation_normal_pivot_guides"]=continuation.normal_pivot_guides;
 out["mobility_rows_max"]=matrix_rows;out["mobility_matrix_bytes_max"]=8ULL*matrix_rows*matrix_rows;
 if(in.contains("container_half"))out["max_container_surface_excess_m"]=surface_excess;
 for(auto& b:bodies)world.removeRigidBody(b.rb.get());
 out["coulomb_polish_solves"]=coulomb_mlcp.stats.polish_solves;out["coulomb_rank_restarts"]=coulomb_mlcp.stats.rank_restarts;out["coulomb_opposing_restarts"]=coulomb_mlcp.stats.opposing_restarts;out["coulomb_polish_svd_calls"]=coulomb_mlcp.stats.polish_svd_calls;out["coulomb_polish_budget_rejections"]=coulomb_mlcp.stats.polish_budget_rejections;out["coulomb_polish_svd_rejections"]=coulomb_mlcp.stats.polish_svd_rejections;out["coulomb_polish_steps"]=coulomb_mlcp.stats.polish_steps;out["coulomb_gauge_restarts"]=coulomb_mlcp.stats.gauge_restarts;out["coulomb_cold_restarts"]=coulomb_mlcp.stats.cold_restarts;std::cout<<out.dump()<<'\n';
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
