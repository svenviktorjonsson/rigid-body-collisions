// Independent research solver harness: no world or production build.
#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include <nlohmann/json.hpp>
#include "projection_more.h"
#include <fstream>
#include <iostream>
using J=nlohmann::json;
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("Usage: replay capture scaling projection");
 J d;std::ifstream file(argv[1]);file>>d;int n=d.at("b").size();btMatrixXu A(n,n);btVectorXu b(n),p(n),hi(n);btAlignedObjectArray<int>dep;dep.resize(n);
 for(int i=0;i<n;i++){b[i]=d["b"][i];p[i]=d["p"][i];hi[i]=d["hi"][i];dep[i]=d["dependencies"][i];for(int j=0;j<n;j++)A.setElem(i,j,d["A"][i][j]);}
 auto before=p;projection_recovery::Stats stats;bool ok=projection_recovery::solve(A,b,p,hi,dep,d.at("tolerance_m_s"),stats,1024,1024,std::stoi(argv[2])!=0,std::stoi(argv[3])!=0);
 bool unchanged=true;std::vector<double>candidate(n),returned(n);for(int i=0;i<n;i++){unchanged&=p[i]==before[i];returned[i]=p[i];candidate[i]=stats.failed_candidate.size()==static_cast<size_t>(n)?stats.failed_candidate[i]:p[i];}
 J out={{"accepted",ok},{"returned_impulse",returned},{"candidate_impulse",candidate},{"decline_preserves_input",ok||unchanged},{"svd_calls",stats.svd_calls},{"iteration_steps",stats.iteration_steps},{"accepted_steps",stats.newton_steps},{"reported_original_residual_m_s",stats.residual}};std::cout<<out.dump()<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 3;}}
