#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "normal_null.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
using json=nlohmann::json;
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("capture path required");json d;std::ifstream(argv[1])>>d;
 if(d.at("phase")!="position_translation")throw std::runtime_error("normal-only position capture required");
 int n=d.at("b").size();btMatrixXu A(n,n);btVectorXu b(n),hi(n),seed(n),out(n);
 for(int i=0;i<n;i++){b[i]=d["b"][i];hi[i]=d["hi"][i];seed[i]=d["p"][i];if(d["lo"][i]!=0||d["dependencies"][i]!=-1)throw std::runtime_error("bounds/dependency mismatch");for(int j=0;j<n;j++)A.setElem(i,j,d["A"][i][j]);}out=seed;
 double tol=d.at("tolerance_m_s");normal_null::Stats stats;bool found=normal_null::solve(A,b,hi,seed,out,tol,stats,128);
 std::vector<double>p(n),w(n);double residual=0,energy=0,scale=1;bool bounds=true,finite=true;
 for(int i=0;i<n;i++){p[i]=out[i];w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*out[j];residual=std::max(residual,std::abs(out[i]-std::max(0.,out[i]-w[i]/A(i,i)))*A(i,i));bounds &= out[i]>=0&&out[i]<=hi[i];finite &= std::isfinite(out[i])&&std::isfinite(w[i]);energy+=.5*out[i]*(w[i]-b[i]);scale+=std::abs(out[i]*b[i]);}
 bool accepted=found&&finite&&bounds&&residual<=tol&&energy<=tol*scale;
 json result={{"max_full_rows",POSITION_MAX_ROWS},{"found",found},{"accepted",accepted},{"p",p},{"w",w},{"independent_residual_m_s",residual},{"passive_change_bound_J",energy},{"passivity_scale",scale},{"stats",{{"states",stats.states},{"svd_calls",stats.svd_calls},{"null_steps",stats.null_steps},{"range_steps",stats.range_steps},{"svd_rejections",stats.svd_rejections},{"budget_rejections",stats.budget_rejections}}}};
 std::cout<<result.dump(2)<<'\n';return accepted?0:2;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 3;}}
