// Prospective explicit signed-gap numerical-position-law replay only.
#include "frozen-c465/translation_split.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
using Json=nlohmann::json;
btVector3 vector(const Json& q){return {q.at(0).get<double>(),q.at(1).get<double>(),q.at(2).get<double>()};}
int main(int argc,char** argv){try{
 if(argc!=4)throw std::runtime_error("Usage: native_replay capture.json geometry.json output.json");
 if(std::ifstream(argv[3]).good())throw std::runtime_error("Refusing to replace retained evidence");
 Json d,g;std::ifstream(argv[1])>>d;std::ifstream(argv[2])>>g;
 if(d.at("schema")!="normal-only-position-rejection-v1"||g.at("schema")!="normal-position-geometry-v1")throw std::runtime_error("Wrong capture schema");
 const int n=d.at("b").size();const double h=d.at("internal_dt_s"),tol=d.at("tolerance_m_s"),slop=1e-9;
 if(n!=74||g.at("rows").size()!=static_cast<size_t>(n)||h!=g.at("internal_dt_s").get<double>())throw std::runtime_error("Wrong planned dimensions/time");
 btMatrixXu A(n,n);btVectorXu oldb(n),b(n),hi(n),x(n);Json target=Json::array();
 for(int i=0;i<n;i++){
  if(d.at("dependencies")[i]!=-1||d.at("lo")[i]!=0)throw std::runtime_error("Normal-only nonnegative capture required");
  oldb[i]=d.at("b")[i];b[i]=oldb[i];hi[i]=d.at("hi")[i];const double gap=g.at("rows")[i].at("signed_distance_m");
  if(gap>slop)b[i]=-(gap-slop)/h;
  if(std::abs(gap)<=slop&&oldb[i]!=0)throw std::runtime_error("Original within-slop target must remain zero");
  target.push_back(b[i]);x[i]=0;
  for(int j=0;j<n;j++)A.setElem(i,j,d.at("A")[i][j].get<double>());
 }
 btVectorXu oldx(n);oldx.setZero();double old_residual=0;normal_null::Stats old_stats;
 const bool original_solved=translationSplitSolve(A,oldb,hi,oldx,tol,d.at("iteration_budget"),&old_residual,&old_stats,true);
 double residual=0;normal_null::Stats stats;
 const bool solved=translationSplitSolve(A,b,hi,x,tol,d.at("iteration_budget"),&residual,&stats,true);
 auto gate=[&](const btVectorXu& target){
  double error=0,minw=std::numeric_limits<double>::infinity(),energy=0,scale=1,pmax=0;bool finite=true,bounds=true;
  for(int i=0;i<n;i++){
   double w=-target[i];for(int j=0;j<n;j++)w+=A(i,j)*x[j];
   finite&=std::isfinite(w)&&std::isfinite(x[i]);bounds&=x[i]>=0&&x[i]<=hi[i];
   error=std::max(error,std::abs(x[i]-std::max(0.,static_cast<double>(x[i])-w/A(i,i)))*A(i,i));
   minw=std::min(minw,w);energy+=.5*x[i]*(w-target[i]);scale+=std::abs(x[i]*target[i]);pmax=std::max(pmax,static_cast<double>(x[i]));
  }
  finite&=std::isfinite(error)&&std::isfinite(energy)&&std::isfinite(scale);
  return Json{{"accepted",finite&&bounds&&error<=tol&&minw>=-tol&&energy<=tol*scale},
   {"residual_m_s",error},{"minimum_normal_velocity_m_s",minw},{"passive_change_bound_J",energy},{"pressure_max_Ns",pmax}};
 };
 Json ids=Json::array(),pose=Json::array(),pressure=Json::array();std::map<int,int> index;std::map<int,double> inverse;std::vector<btVector3> dx;
 for(const auto& body:g.at("bodies"))if(body.at("inverse_mass").get<double>()>0){int id=body.at("solver_body_id");index[id]=dx.size();inverse[id]=body.at("inverse_mass");ids.push_back(body.at("body_id"));dx.emplace_back(0,0,0);}
 for(int i=0;i<n;i++){
  pressure.push_back(x[i]);const auto& row=g.at("rows")[i];
  for(const std::string side:{"a","b"}){int id=row.at("solver_body_id_"+side);if(index.count(id))dx[index[id]]+=vector(row.at("linear_jacobian_"+side))*(h*inverse.at(id)*x[i]);}
 }
 double movement=0;for(const auto& v:dx){movement=std::max(movement,static_cast<double>(v.length()));pose.push_back(Json::array({v.x(),v.y(),v.z(),0.,0.,0.}));}
 Json result={{"schema","native-prospective-signed-gap-position-v1"},{"native_solved",solved},
  {"original_policy_qualified",false},{"original_native_solved",original_solved},{"original_native_residual_m_s",old_residual},{"prospective_target",target},{"finite_body_ids",ids},
  {"stats",{{"states",stats.states},{"svd_calls",stats.svd_calls},{"null_steps",stats.null_steps},{"range_steps",stats.range_steps},{"released",stats.released},{"entered",stats.entered},{"svd_rejections",stats.svd_rejections},{"budget_rejections",stats.budget_rejections}}},
  {"attempts",Json::array({{{"method","native translationSplitSolve, prospective signed-gap position targets"},{"impulse",pressure},{"original_gate",gate(oldb)},{"prospective_gate",gate(b)},{"linear_qualified",solved&&gate(b).at("accepted").get<bool>()},{"pose_increment",pose},{"max_translation_norm_m",movement},{"max_rotation_norm_rad",0.},{"trajectory_qualified",false}}})}};
 std::ofstream out(argv[3]);out<<result.dump(2)<<"\n";if(!out)throw std::runtime_error("Cannot write receipt");std::cout<<result.dump()<<"\n";
 return !original_solved&&solved&&gate(b).at("accepted").get<bool>()?0:1;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 2;}}
