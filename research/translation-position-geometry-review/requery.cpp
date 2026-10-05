// Independent geometry clone/re-query; never changes the simulation world.
#include <btBulletDynamicsCommon.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <vector>
using J=nlohmann::json;
btVector3 vec(const J& x){return {x[0].get<double>(),x[1].get<double>(),x[2].get<double>()};}
btQuaternion quat(const J& x){return {x[0].get<double>(),x[1].get<double>(),x[2].get<double>(),x[3].get<double>()};}
int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("Usage: requery geometry.json guide.json output.json");
 if(std::ifstream(argv[3]).good())throw std::runtime_error("Refusing to overwrite a retained trial");
 J g,guide;std::ifstream(argv[1])>>g;std::ifstream(argv[2])>>guide;
 if(!g.contains("wire_bodies")||!g.contains("margin_m"))throw std::runtime_error("Missing authored backend shape geometry/margin");
 btDefaultCollisionConfiguration config;btCollisionDispatcher dispatcher(&config);btDbvtBroadphase broad;
 btCollisionWorld world(&dispatcher,&broad,&config);
 std::vector<std::unique_ptr<btCollisionShape>> shapes;std::vector<std::unique_ptr<btCollisionObject>> objects;
 std::map<int,btTransform> transforms;
 for(const auto& b:g.at("bodies"))if(b.at("has_original_body")){const auto& t=b.at("world_transform");transforms.emplace(b.at("body_id").get<int>(),btTransform(quat(t.at("quaternion_xyzw")),vec(t.at("position"))));}
 for(size_t i=0;i<g.at("wire_bodies").size();i++){
  const auto& b=g.at("wire_bodies")[i];auto compound=std::make_unique<btCompoundShape>();compound->setMargin(0);
  for(const auto& s:b.at("shapes")){
   std::unique_ptr<btCollisionShape> shape;
   if(s.at("kind")=="sphere")shape=std::make_unique<btSphereShape>(s.at("radius").get<double>());
   else if(s.at("kind")=="box"){shape=std::make_unique<btBoxShape>(vec(s.at("half_extents")));shape->setMargin(g.at("margin_m").get<double>());}
   else{auto hull=std::make_unique<btConvexHullShape>();for(const auto& p:s.at("vertices"))hull->addPoint(vec(p),false);hull->recalcLocalAabb();hull->setMargin(g.at("margin_m").get<double>());shape=std::move(hull);}
   compound->addChildShape(btTransform(quat(s.at("orientation")),vec(s.at("center"))),shape.get());shapes.push_back(std::move(shape));
  }
  auto object=std::make_unique<btCollisionObject>();object->setCollisionShape(compound.get());object->setUserIndex(static_cast<int>(i));
  if(!transforms.count(static_cast<int>(i)))throw std::runtime_error("No exact failed-step transform for authored body");
  object->setWorldTransform(transforms.at(static_cast<int>(i)));
  object->setCollisionFlags(b.at("type")=="kinematic"?btCollisionObject::CF_KINEMATIC_OBJECT:0);
  world.addCollisionObject(object.get());objects.push_back(std::move(object));shapes.push_back(std::move(compound));
 }
 auto query=[&](){
  world.performDiscreteCollisionDetection();J contacts=J::array();double penetration=0;
  for(int k=0;k<dispatcher.getNumManifolds();k++){
   auto* m=dispatcher.getManifoldByIndexInternal(k);
   for(int c=0;c<m->getNumContacts();c++){
    const auto& p=m->getContactPoint(c);penetration=std::max(penetration,-static_cast<double>(p.getDistance()));
    contacts.push_back({{"body_a",m->getBody0()->getUserIndex()},{"body_b",m->getBody1()->getUserIndex()},{"signed_distance_m",p.getDistance()}});
   }
  }
  return J{{"max_penetration_m",penetration},{"contacts",contacts},{"contact_count",contacts.size()}};
 };
 J out={{"schema","independent-position-geometry-requery-v1"},{"before",query()},{"trials",J::array()}};
 for(const auto& trial:guide.at("attempts")){
  for(size_t k=0;k<objects.size();k++){objects[k]->setWorldTransform(transforms.at(static_cast<int>(k)));world.updateSingleAabb(objects[k].get());}
  for(size_t k=0;k<guide.at("finite_body_ids").size();k++){
   int id=guide.at("finite_body_ids")[k];if(id<0||id>=static_cast<int>(objects.size()))throw std::runtime_error("Invalid guide body ID");
   const auto& d=trial.at("pose_increment")[k];auto t=transforms.at(id);btVector3 dx(d[0].get<double>(),d[1].get<double>(),d[2].get<double>()),theta(d[3].get<double>(),d[4].get<double>(),d[5].get<double>());
   const double angle=theta.length();t.setOrigin(t.getOrigin()+dx);if(angle>0){auto q=btQuaternion(theta/angle,angle)*t.getRotation();q.normalize();t.setRotation(q);}
   objects[id]->setWorldTransform(t);world.updateSingleAabb(objects[id].get());
  }
  // Clear retained contacts, then re-discover from geometry rather than trusting
  // the old frozen normal equations or the existing manifold pressure state.
  for(int k=0;k<dispatcher.getNumManifolds();k++)dispatcher.clearManifold(dispatcher.getManifoldByIndexInternal(k));
  out["trials"].push_back({{"method",trial.at("method")},{"after",query()},{"linear_qualified",trial.at("linear_qualified")},{"trajectory_qualified",false}});
 }
 for(auto& object:objects)world.removeCollisionObject(object.get());
 std::ofstream file(argv[3]);file<<out.dump(2)<<"\n";if(!file)throw std::runtime_error("Failed writing geometry receipt");
 std::cout<<out.dump()<<"\n";return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
