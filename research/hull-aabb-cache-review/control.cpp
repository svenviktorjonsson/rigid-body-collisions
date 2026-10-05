// Independent cache-construction controls only. No world/engine state changes.
#include <btBulletDynamicsCommon.h>
#include <nlohmann/json.hpp>
#include <fstream>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>
using J=nlohmann::json;
J vec(const btVector3& v){return J::array({v.x(),v.y(),v.z()});}
double error(const btVector3& a,const btVector3& b){return std::max({std::abs(a.x()-b.x()),std::abs(a.y()-b.y()),std::abs(a.z()-b.z())});}
struct Bounds{btVector3 lo,hi;};
Bounds aabb(const btCollisionShape& s,const btTransform& t){Bounds b;s.getAabb(t,b.lo,b.hi);return b;}
Bounds formula(const std::vector<btVector3>& points,const btTransform& t,double cached_margin,double declared_margin){
 btVector3 lo(BT_LARGE_FLOAT,BT_LARGE_FLOAT,BT_LARGE_FLOAT),hi=-lo;
 for(const auto& p:points){lo.setMin(p);hi.setMax(p);}
 const auto center=t((hi+lo)*.5);const auto half=(hi-lo)*.5+btVector3(cached_margin+declared_margin,cached_margin+declared_margin,cached_margin+declared_margin);const auto extent=t.getBasis().absolute()*half;
 return {center-extent,center+extent};
}
Bounds trueSupport(const std::vector<btVector3>& points,const btTransform& t,double margin){
 btVector3 lo(BT_LARGE_FLOAT,BT_LARGE_FLOAT,BT_LARGE_FLOAT),hi=-lo;
 for(const auto& p:points){auto q=t(p);lo.setMin(q);hi.setMax(q);}return {lo-btVector3(margin,margin,margin),hi+btVector3(margin,margin,margin)};
}
std::unique_ptr<btConvexHullShape> make(const std::vector<btVector3>& points,double margin,bool old){
 auto h=std::make_unique<btConvexHullShape>();for(auto p:points)h->addPoint(p,false);
 if(old){h->recalcLocalAabb();h->setMargin(margin);}else{h->setMargin(margin);h->recalcLocalAabb();}return h;
}
int main(int argc,char** argv){try{
 static_assert(sizeof(btScalar)==8,"Double precision required");
 if(argc!=3)throw std::runtime_error("Usage: control fixture.json output.json");if(std::ifstream(argv[2]).good())throw std::runtime_error("Refusing to replace a retained control");J fixture;std::ifstream(argv[1])>>fixture;J trials=J::array();bool all=true;int count=0;
 for(const auto& geometry:fixture.at("geometries")){
  std::vector<btVector3> points;for(const auto& p:geometry.at("vertices"))points.emplace_back(p[0].get<double>(),p[1].get<double>(),p[2].get<double>());
  for(double margin:{0.,.003})for(bool rotated:{false,true}){
   btTransform t;t.setIdentity();if(rotated)t=btTransform(btQuaternion(btVector3(1,2,3).normalized(),.37),btVector3(.014,-.003,.006));
   auto old=make(points,margin,true),corrected=make(points,margin,false);Bounds o=aabb(*old,t),c=aabb(*corrected,t),eo=formula(points,t,.04,margin),ec=formula(points,t,margin,margin),actual=trueSupport(points,t,margin);double formula_error=std::max({error(o.lo,eo.lo),error(o.hi,eo.hi),error(c.lo,ec.lo),error(c.hi,ec.hi)}),support_error=0;bool containment=true;
   for(int i=0;i<3;i++){
    btVector3 n(0,0,0);n[i]=1;auto local=t.getBasis().transpose()*n;auto oldplus=t(old->localGetSupportingVertex(local));auto newplus=t(corrected->localGetSupportingVertex(local));auto oldminus=t(old->localGetSupportingVertex(-local));auto newminus=t(corrected->localGetSupportingVertex(-local));support_error=std::max({support_error,std::abs(oldplus[i]-actual.hi[i]),std::abs(newplus[i]-actual.hi[i]),std::abs(oldminus[i]-actual.lo[i]),std::abs(newminus[i]-actual.lo[i])});containment&=c.lo[i]<=actual.lo[i]+1e-12&&c.hi[i]>=actual.hi[i]-1e-12;
   }
   btCompoundShape oldcompound,newcompound;oldcompound.setMargin(0);newcompound.setMargin(0);oldcompound.addChildShape(t,old.get());newcompound.addChildShape(t,corrected.get());btTransform identity;identity.setIdentity();const auto oc=aabb(oldcompound,identity),nc=aabb(newcompound,identity);double compound_error=std::max({error(oc.lo,o.lo),error(oc.hi,o.hi),error(nc.lo,c.lo),error(nc.hi,c.hi)});
   const double oldbreak=oldcompound.getContactBreakingThreshold(.02),newbreak=newcompound.getContactBreakingThreshold(.02);const double expectedbreak=.02*((c.hi-c.lo).length()*.5+((c.hi+c.lo)*.5).length());bool inflated=true;for(int i=0;i<3;i++)inflated&=o.lo[i]<c.lo[i]-.01&&o.hi[i]>c.hi[i]+.01;
   bool exact_zero_identity=margin!=0||rotated||(error(c.lo,actual.lo)<1e-12&&error(c.hi,actual.hi)<1e-12);bool pass=formula_error<1e-12&&support_error<1e-12&&compound_error<1e-12&&containment&&inflated&&exact_zero_identity&&newbreak<oldbreak&&std::abs(newbreak-expectedbreak)<1e-12;all&=pass;count++;
   trials.push_back({{"geometry",geometry.at("id")},{"declared_margin_m",margin},{"child_transform_rotated",rotated},{"passed",pass},{"old_aabb",{{"min",vec(o.lo)},{"max",vec(o.hi)}}},{"corrected_aabb",{{"min",vec(c.lo)},{"max",vec(c.hi)}}},{"actual_support_aabb",{{"min",vec(actual.lo)},{"max",vec(actual.hi)}}},{"expected_formula_error_m",formula_error},{"native_support_error_m",support_error},{"compound_formula_error_m",compound_error},{"old_default_margin_inflation_retained",inflated},{"corrected_contains_actual_declared_support",containment},{"zero_margin_identity_exact_vertex_bounds",margin==0&&!rotated?J(exact_zero_identity):J(nullptr)},{"old_compound_breaking_threshold_m",oldbreak},{"corrected_compound_breaking_threshold_m",newbreak},{"nonzero_margin_note","Bullet cached bounds include margin and getAabb adds margin again; conservative2margin padding, not exact support equality"}});
  }
 }
 J out={{"schema","native-hull-aabb-cache-controls-v1"},{"passed",all},{"controls",count},{"trials",trials},{"scope","Construction/cache consistency only. No solver/world execution or trajectory qualification."}};std::ofstream file(argv[2]);file<<out.dump(2)<<"\n";if(!file)throw std::runtime_error("Cannot write control receipt");std::cout<<out.dump()<<"\n";return all?0:1;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}
