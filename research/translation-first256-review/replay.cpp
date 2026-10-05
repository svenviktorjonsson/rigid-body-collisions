// Isolated captured-system test; final law independently checked outside helper.
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "coulomb.h"
#include "../translation-native-review/v3/support_restart.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <chrono>
using json=nlohmann::json;
int main(int argc,char** argv){
 for(int file=1;file<argc;file++){
  std::ifstream input(argv[file]);json d;input>>d;const int n=d["b"].size();
  btMatrixXu A(n,n);btVectorXu b(n),p(n),hi(n);btAlignedObjectArray<int>dep;dep.resize(n);
  for(int i=0;i<n;i++){b[i]=d["b"][i];p[i]=d["p"][i];hi[i]=d["hi"][i];dep[i]=d["dependencies"][i];for(int j=0;j<n;j++)A.setElem(i,j,d["A"][i][j]);}
  const double tolerance=d.at("tolerance_m_s");
  const auto original=p;btVectorXu lo(n);for(int i=0;i<n;i++)lo[i]=d["lo"][i];
  for(int repeat=0;repeat<3;repeat++)for(int order=0;order<2;order++){
  const bool early=((repeat%2)==0 ? order==1 : order==0);
  p=original;support_restart_v3::Stats stats,tail_stats;CoulombStats base,first;
  std::vector<double> final_rejected,first_rejected;bool accepted=false;
  std::string lane;double residual=0;
  const auto started=std::chrono::steady_clock::now();
  if(early){
   accepted=coulombIterate(A,b,p,lo,hi,dep,256,tolerance,first,&first_rejected);
   lane="first256";residual=first.last_residual;
   if(!accepted){
    if(first_rejected.size()!=static_cast<size_t>(n))throw std::runtime_error("Missing first256 seed");
    for(int i=0;i<n;i++)p[i]=first_rejected[i];
    accepted=support_restart_v3::solve(A,b,p,hi,dep,tolerance,stats);
    lane="first256_v3";residual=stats.residual;
   }
  }
  if(!accepted){
   p=original;accepted=coulombSolve(A,b,p,lo,hi,dep,4096,tolerance,base,&final_rejected);
   lane=early ? "early_decline_frozen52" : "baseline_frozen52";residual=base.last_residual;
   if(!accepted){
    if(final_rejected.size()!=static_cast<size_t>(n))throw std::runtime_error("Missing final rejected seed");
    for(int i=0;i<n;i++)p[i]=final_rejected[i];
    support_restart_v3::Stats& tail=tail_stats;
    accepted=support_restart_v3::solve(A,b,p,hi,dep,tolerance,tail);
    // Keep early and tail work separate, rather than resetting consumed counters.
    lane=early ? "early_decline_frozen52_tail_v3" : "baseline_tail_v3";residual=tail.residual;
    if(!early)stats=tail;
   }
  }
  const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
  std::vector<double>x(n),w(n);double energy=0,scale=1,impulseScale=1;bool finite=true;
  for(int i=0;i<n;i++){x[i]=p[i];w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*p[j];
   finite&=std::isfinite(x[i])&&std::isfinite(w[i]);energy+=.5*x[i]*(w[i]-b[i]);scale+=std::abs(x[i]*b[i]);impulseScale=std::max(impulseScale,std::abs(x[i]));}
  // These are independent physical inequalities/support-function identities,
  // not the projection/FB numerical merit or optimizer convergence flag.
  double normalNegative=0,normalImpulseNegative=0,complementWork=0,coneViolation=0,supportGap=0,positiveFrictionWork=0;
  for(int k=0;k<n;k++)if(dep[k]<0){
   std::vector<int>t;for(int j=0;j<n;j++)if(dep[j]==k)t.push_back(j);if(t.size()!=2)throw std::runtime_error("Invalid captured contact");
   const int u=t[0],v=t[1];const double mu=hi[u],cap=mu*std::max(0.,x[k]);
   const double wt=std::hypot(w[u],w[v]),pt=std::hypot(x[u],x[v]),work=x[u]*w[u]+x[v]*w[v];
   const double eigen=.5*(A(u,u)+A(v,v)+std::hypot(A(u,u)-A(v,v),2*A(u,v)));
   normalNegative=std::max(normalNegative,-w[k]);normalImpulseNegative=std::max(normalImpulseNegative,-x[k]*A(k,k));
   complementWork=std::max(complementWork,std::abs(x[k]*w[k]));coneViolation=std::max(coneViolation,(pt-cap)*eigen);
   supportGap=std::max(supportGap,std::abs(work+cap*wt));positiveFrictionWork=std::max(positiveFrictionWork,work);
  }
  const bool law=finite&&normalNegative<=tolerance&&normalImpulseNegative<=tolerance&&
      complementWork<=tolerance*impulseScale&&coneViolation<=tolerance&&supportGap<=tolerance*impulseScale&&positiveFrictionWork<=tolerance*impulseScale;
  const bool passive=std::isfinite(energy)&&std::isfinite(scale)&&energy<=tolerance*scale;
  json result={{"capture",argv[file]},{"solver_accepted",accepted},{"independent_law_accepted",law},{"independent_passivity_accepted",passive},{"accepted",accepted&&law&&passive},{"residual",residual},{"lane",lane},{"svd_calls",stats.svd_calls},{"iteration_steps",stats.iteration_steps},{"helper_calls",stats.helper_calls},{"skipped_components",stats.skipped_components},{"component_sizes",stats.component_sizes},{"component_cap_rejections",stats.component_cap_rejections},{"passes",stats.passes},{"largest_reduced_rows",stats.largest_reduced_rows},{"row_counts",stats.row_counts},{"release_contacts",stats.release_contacts},{"expanded_contacts",stats.expanded_contacts},{"pressure_svd_calls",stats.pressure_svd_calls},{"pressure_attempts",stats.pressure_attempts},{"pivot_attempts",stats.pivot_attempts},{"passive_change_bound_J",energy},{"p",x},{"w",w}};
  auto raw=[](const CoulombStats&s){return json{{"iteration_sweeps_total",s.iteration_sweeps_total},{"newton_steps",s.newton_steps},{"polish_steps",s.polish_steps},{"polish_svd_calls",s.polish_svd_calls},{"active_svd_calls",s.active.search.svd_calls},{"continuation_svd_calls",s.continuation.svd_calls},{"normal_null_svd_calls",s.null_pressure.svd_calls},{"pressure_svd_calls",s.pressure.svd_calls},{"supplemental_svd_calls",s.supplemental_svd_calls},{"supplemental_pressure_svd_calls",s.supplemental_pressure_svd_calls},{"supplemental_pivot_calls",s.supplemental_pivot_calls},{"last_residual",s.last_residual}};};
  result["repeat"]=repeat;result["order"]=order;result["strategy"]=early?"early":"baseline";result["solve_seconds"]=seconds;
  std::vector<double> initial(n);for(int i=0;i<n;i++)initial[i]=original[i];result["initial_p"]=initial;
  result["first256_rejected_p"]=first_rejected;result["first256_counters"]=raw(first);result["baseline_counters"]=raw(base);
  result["early_fallback_tail_counters"]={{"svd_calls",early?tail_stats.svd_calls:0},{"helper_calls",early?tail_stats.helper_calls:0},{"passes",early?tail_stats.passes:0},{"pressure_svd_calls",early?tail_stats.pressure_svd_calls:0},{"pivot_attempts",early?tail_stats.pivot_attempts:0}};
  std::cout<<result.dump()<<std::endl;
  }
 }
}
