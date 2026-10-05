// Isolated captured-system test; final law independently checked outside helper.
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "active_face_v4.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
using json=nlohmann::json;
int main(int argc,char** argv){
 for(int file=1;file<argc;file++){
  std::ifstream input(argv[file]);json d;input>>d;const int n=d["b"].size();
  btMatrixXu A(n,n);btVectorXu b(n),p(n),hi(n);btAlignedObjectArray<int>dep;dep.resize(n);
  for(int i=0;i<n;i++){b[i]=d["b"][i];p[i]=d["p"][i];hi[i]=d["hi"][i];dep[i]=d["dependencies"][i];for(int j=0;j<n;j++)A.setElem(i,j,d["A"][i][j]);}
  const double tolerance=d.at("tolerance_m_s");
  active_face_v4::Stats all;auto& stats=all.search;bool accepted=active_face_v4::solve(A,b,p,hi,dep,tolerance,all);
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
  json result={{"capture",argv[file]},{"solver_accepted",accepted},{"independent_law_accepted",law},{"independent_passivity_accepted",passive},{"accepted",accepted&&law&&passive},{"passive_change_bound_J",energy},{"passivity_scale",scale},{"normal_negative_m_s",normalNegative},{"normal_impulse_negative_m_s",normalImpulseNegative},{"normal_complementarity_work_J",complementWork},{"cone_violation_m_s",coneViolation},{"maximum_dissipation_support_gap_J",supportGap},{"residual",all.residual},{"passes",all.passes},{"mode_guesses",all.mode_guesses},{"active_contacts",all.active_contacts},{"expanded_contacts",all.expanded_contacts},{"pressure_guides",stats.pressure_guides},{"pressure_attempts",stats.pressure_attempts},{"guide_residual",stats.guide_residual},{"normal_qp_guides",stats.normal_qp_guides},{"normal_pivot_attempts",stats.normal_pivot_attempts},{"normal_pivot_guides",stats.normal_pivot_guides},{"attempts",stats.attempts},{"stages",stats.stages},{"newton_steps",stats.newton_steps},{"svd_calls",stats.svd_calls},{"damped_steps",stats.damped_steps},{"budget_rejections",stats.budget_rejections},{"p",x},{"w",w}};
  std::cout<<result.dump()<<"\n";
 }
}
