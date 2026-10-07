// Native frozen-system regression with independent physical bookkeeping.
// This executable does not simulate a hull trajectory or validate its geometry.
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "coulomb.h"
#include "null_seed192.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <string>
using json=nlohmann::json;

struct ReplayContact {int normal,t,s;double mu,tangent_eigen;};

int main(int argc,char** argv){
 static_assert(sizeof(btScalar)==8,"Frozen replay requires Float64 Bullet");
 try{
  if(argc<2||argc>3)throw std::runtime_error("Usage: spatial_coulomb_replay dump.json [iteration_budget]");
  std::ifstream input(argv[1]);if(!input)throw std::runtime_error("Cannot open frozen-system JSON");
  json data;input>>data;
  const auto& rhs=data.at("b");int n=static_cast<int>(rhs.size());
  if(!rhs.is_array()||n<=0||n>4096)throw std::runtime_error("Invalid frozen-system row count");
  for(const std::string key:{"A","p","lo","hi","dependencies"})if(!data.at(key).is_array()||static_cast<int>(data.at(key).size())!=n)
   throw std::runtime_error("Incompatible frozen-system shapes");
  double tolerance=data.at("tolerance_m_s").get<double>();
  int budget=argc==3?std::stoi(argv[2]):data.value("iteration_budget",4096);
  if(!(tolerance>0)||!std::isfinite(tolerance)||budget<0||budget>1000000)throw std::runtime_error("Invalid acceptance tolerance or work budget");
  btMatrixXu A(n,n);btVectorXu b(n),p(n),lo(n),hi(n);btAlignedObjectArray<int> dependencies;dependencies.resize(n);
  double matrix_scale=1;
  for(int i=0;i<n;i++){
   if(!data["A"][i].is_array()||static_cast<int>(data["A"][i].size())!=n)throw std::runtime_error("Mobility must be square");
   b[i]=rhs[i].get<double>();p[i]=data["p"][i].get<double>();lo[i]=data["lo"][i].get<double>();hi[i]=data["hi"][i].get<double>();
   if(!data["dependencies"][i].is_number_integer())throw std::runtime_error("Dependency indices must be integers");
   dependencies[i]=data["dependencies"][i].get<int>();
   if(!std::isfinite(b[i])||!std::isfinite(p[i])||!std::isfinite(lo[i])||!std::isfinite(hi[i]))throw std::runtime_error("Frozen vector data must be finite");
   for(int j=0;j<n;j++){
    double value=data["A"][i][j].get<double>();if(!std::isfinite(value))throw std::runtime_error("Frozen mobility must be finite");
    A.setElem(i,j,value);matrix_scale=std::max(matrix_scale,std::abs(value));
   }
  }
  for(int i=0;i<n;i++)for(int j=0;j<i;j++)if(std::abs(A(i,j)-A(j,i))>1e-12*matrix_scale)
   throw std::runtime_error("Mobility is not symmetric to its declared Float64 precision");
  std::vector<ReplayContact> contacts;std::vector<bool> covered(n,false);
  for(int k=0;k<n;k++)if(dependencies[k]<0){
   if(lo[k]!=0||hi[k]<1e9||!(A(k,k)>0))throw std::runtime_error("Unsupported unilateral normal row");
   std::vector<int> tangents;for(int j=0;j<n;j++)if(dependencies[j]==k)tangents.push_back(j);
   if(tangents.size()!=2)throw std::runtime_error("Two tangent rows per normal required");
   int t=tangents[0],s=tangents[1];
   if(lo[t]!=-hi[t]||lo[s]!=-hi[s]||hi[t]!=hi[s]||hi[t]<0)throw std::runtime_error("An isotropic circular coefficient is required");
   double a=A(t,t),d=A(s,s),off=A(t,s),eigen=.5*(a+d+std::hypot(a-d,2*off));
   if(!(a>0&&d>0&&a*d-off*off>0&&eigen>0))throw std::runtime_error("Positive tangent mobility block required");
   contacts.push_back({k,t,s,static_cast<double>(hi[t]),eigen});covered[k]=covered[t]=covered[s]=true;
  }
  for(int i=0;i<n;i++)if(!covered[i]||(dependencies[i]>=0&&(dependencies[i]>=n||dependencies[dependencies[i]]>=0)))
   throw std::runtime_error("Unsupported or unassigned contact dependency");
  CoulombStats stats;
  stats.null_seed_attempts++;
  bool found=null_seed192::solve(A,b,p,hi,dependencies,p,tolerance,stats.null_seed);
  bool solver_ok=found&&coulombIterate(A,b,p,lo,hi,dependencies,0,tolerance,stats);
  if(solver_ok)stats.null_seed_solves++;else stats.null_seed_declines++;
  std::vector<double>w(n),impulses(n);bool finite=true;
  double energy_change=0,energy_scale=1,impulse_scale=1;
  for(int i=0;i<n;i++){
   impulses[i]=p[i];w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*p[j];
   finite&=std::isfinite(p[i])&&std::isfinite(w[i]);
   energy_change+=.5*p[i]*(w[i]-b[i]);energy_scale+=std::abs(p[i]*b[i]);impulse_scale=std::max(impulse_scale,std::abs(static_cast<double>(p[i])));
  }
  // Independent physical law checks, rather than trusting optimizer status.
  double negative_normal_velocity=0,negative_normal_impulse_velocity=0,normal_complementarity_work=0,active_normal_velocity=0;
  double cone_violation_velocity=0,sticking_velocity=0,friction_work_gap=0,positive_friction_work=0;
  bool upper_bounds=true;json contact_checks=json::array();
  for(auto c:contacts){
   double pn=p[c.normal],wn=w[c.normal],pt=p[c.t],ps=p[c.s],wt=w[c.t],ws=w[c.s];
   double tangent_impulse=std::hypot(pt,ps),tangent_velocity=std::hypot(wt,ws),cap=c.mu*std::max(0.,pn);
   double work=pt*wt+ps*ws;
   negative_normal_velocity=std::max(negative_normal_velocity,-wn);
   negative_normal_impulse_velocity=std::max(negative_normal_impulse_velocity,-pn*A(c.normal,c.normal));
   normal_complementarity_work=std::max(normal_complementarity_work,std::abs(pn*wn));
   if(pn*A(c.normal,c.normal)>tolerance)active_normal_velocity=std::max(active_normal_velocity,std::abs(wn));
   cone_violation_velocity=std::max(cone_violation_velocity,(tangent_impulse-cap)*c.tangent_eigen);
   bool interior=cap-tangent_impulse>tolerance/c.tangent_eigen;
   if(interior)sticking_velocity=std::max(sticking_velocity,tangent_velocity);
   // Maximum dissipation is independently checked as a support-function gap:
   // pt dot wt + mu*pn*|wt| = 0, both for stick and saturated opposing slip.
   friction_work_gap=std::max(friction_work_gap,std::abs(work+cap*tangent_velocity));
   positive_friction_work=std::max(positive_friction_work,work);
   upper_bounds&=pn<=hi[c.normal];
   contact_checks.push_back({{"normal_row",c.normal},{"normal_impulse",pn},{"normal_velocity_m_s",wn},
                            {"tangent_impulse_norm",tangent_impulse},{"tangent_velocity_norm_m_s",tangent_velocity},
                            {"friction_capacity",cap},{"friction_support_gap_J",work+cap*tangent_velocity}});
  }
  bool law_ok=finite&&upper_bounds&&negative_normal_velocity<=tolerance&&negative_normal_impulse_velocity<=tolerance&&
              cone_violation_velocity<=tolerance&&sticking_velocity<=tolerance&&active_normal_velocity<=tolerance&&
              normal_complementarity_work<=tolerance*impulse_scale&&friction_work_gap<=tolerance*impulse_scale&&
              positive_friction_work<=tolerance*impulse_scale;
  bool energy_ok=std::isfinite(energy_change)&&std::isfinite(energy_scale)&&energy_change<=tolerance*energy_scale;
  json output={{"schema","native-circular-coulomb-replay-v1"},{"rows",n},{"tolerance_m_s",tolerance},{"iteration_budget",budget},
               {"solver_accepted",solver_ok},{"independent_law_accepted",law_ok},{"independent_passivity_accepted",energy_ok},
               {"accepted",solver_ok&&law_ok&&energy_ok},{"p",impulses},{"w",w},{"contacts",contact_checks},
               {"negative_normal_velocity_m_s",negative_normal_velocity},{"negative_normal_impulse_scaled_m_s",negative_normal_impulse_velocity},
               {"normal_complementarity_work_J",normal_complementarity_work},{"active_normal_velocity_m_s",active_normal_velocity},{"circle_violation_scaled_m_s",cone_violation_velocity},
               {"interior_sticking_velocity_m_s",sticking_velocity},{"maximum_dissipation_gap_J",friction_work_gap},
               {"positive_friction_work_J",positive_friction_work},{"passive_change_bound_J",energy_change},{"passivity_scale",energy_scale},
               {"stats",{{"lapack_recovery_compiled",coulombLapackRecoveryEnabled()},{"support_solves",stats.support_solves},{"support_helper_calls",stats.support_helper_calls},{"support_skipped_components",stats.support_skipped_components},{"support_component_cap_rejections",stats.support_component_cap_rejections},{"support_passes",stats.support_passes},{"support_largest_rows",stats.support_largest_rows},{"support_expanded_contacts",stats.support_expanded_contacts},{"support_svd_calls",stats.support_svd_calls},{"support_iteration_steps",stats.support_iteration_steps},{"support_pressure_svd_calls",stats.support_pressure_svd_calls},{"support_pivot_calls",stats.support_pivot_calls},{"supplemental_solves",stats.supplemental_solves},{"supplemental_svd_calls",stats.supplemental_svd_calls},{"supplemental_iteration_steps",stats.supplemental_iteration_steps},{"supplemental_pressure_svd_calls",stats.supplemental_pressure_svd_calls},{"supplemental_pivot_calls",stats.supplemental_pivot_calls},{"supplemental_projector_calls",stats.supplemental_projector_calls},{"supplemental_restarts",stats.supplemental_restarts},{"null_pressure_solves",stats.null_pressure.solves},{"null_pressure_svd_calls",stats.null_pressure.svd_calls},{"null_pressure_boundary_moves",stats.null_pressure.null_steps},{"active_solves",stats.active_solves},{"active_passes",stats.active.passes},{"active_svd_calls",stats.active.search.svd_calls},{"active_pressure_svd_calls",stats.active.search.pressure_svd_calls},{"active_mode_guesses",stats.active.mode_guesses},{"pressure_solves",stats.pressure_solves},{"pressure_svd_calls",stats.pressure.svd_calls},{"iteration_sweeps_total",stats.iteration_sweeps_total},{"solves",stats.solves},{"sweeps_max",stats.sweeps_max},{"newton_steps",stats.newton_steps},
                         {"polish_solves",stats.polish_solves},{"polish_steps",stats.polish_steps},{"gauge_restarts",stats.gauge_restarts},
                         {"continuation_solves",stats.continuation_solves},{"continuation_svd_calls",stats.continuation.svd_calls},{"continuation_stages",stats.continuation.stages},{"continuation_budget_rejections",stats.continuation.budget_rejections},{"rank_restarts",stats.rank_restarts},{"opposing_restarts",stats.opposing_restarts},{"cold_restarts",stats.cold_restarts},{"polish_svd_calls",stats.polish_svd_calls},
                         {"polish_budget_rejections",stats.polish_budget_rejections},{"polish_svd_rejections",stats.polish_svd_rejections},{"residual_m_s",stats.last_residual}}}};
  output["projection_tail_policy"]={{"stage","after_all_existing_pipeline_failure"},
   {"compiled",coulombLapackRecoveryEnabled()},{"max_rows",64},{"max_svd_calls",2048},{"max_iteration_steps",2048},
   {"attempts",stats.projection_attempts},{"solves",stats.projection_solves},{"declines",stats.projection_declines},
   {"svd_calls",stats.projection_svd_calls},{"iteration_steps",stats.projection_iteration_steps},{"newton_steps",stats.projection_newton_steps}};
#ifdef SPATIAL_LAPACK_RECOVERY
  const auto& t=stats.null_seed;output["null_traction_seed_policy"]={{"attempts",stats.null_seed_attempts},{"solves",stats.null_seed_solves},{"declines",stats.null_seed_declines},{"components",t.components},{"largest_component_rows",t.largest_rows},{"component_cap_rejections",t.cap_rejections},{"seed_attempts",t.seed_attempts},{"null_svd_calls",t.null_svd_calls},{"seed_svd_calls",t.seed_svd_calls},{"iteration_steps",t.iteration_steps},{"svd_calls",t.svd_calls},{"newton_steps",t.newton_steps},{"seed_response_change_max_m_s",t.seed_response_change_max}};
#endif
  output["experimental_component_cap_rows"]=192;
  std::cout<<output.dump(2)<<'\n';return solver_ok&&law_ok&&energy_ok?0:2;
 }catch(const std::exception& error){std::cerr<<json({{"accepted",false},{"error",error.what()}}).dump()<<'\n';return 3;}
}
