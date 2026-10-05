#include "pressure_release.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
using json=nlohmann::json;
int main(int argc,char** argv){for(int file=1;file<argc;file++){
 std::ifstream input(argv[file]);json d;input>>d;std::vector<int>ks;for(int i=0;i<static_cast<int>(d["b"].size());i++)if(d["dependencies"][i].get<int>()<0)ks.push_back(i);int n=ks.size();
 btMatrixXu A(n,n);btVectorXu b(n),upper(n),seed(n),p(n);for(int i=0;i<n;i++){b[i]=d["b"][ks[i]];upper[i]=d["hi"][ks[i]];seed[i]=d["p"][ks[i]];p[i]=seed[i];for(int j=0;j<n;j++)A.setElem(i,j,d["A"][ks[i]][ks[j]]);}
 normal_pressure::Stats stats;bool success=normal_pressure::solve(A,b,upper,seed,p,1e-8,stats);std::vector<double>x(n),w(n);for(int i=0;i<n;i++){x[i]=p[i];w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*p[j];}
 double residual=0,energy=0,energy_scale=1,negative=0,upper_bad=0;for(int i=0;i<n;i++){
 residual=std::max(residual,std::abs(x[i]-std::max(0.,x[i]-w[i]/A(i,i)))*A(i,i));
 energy+=.5*x[i]*(w[i]-b[i]);energy_scale+=std::abs(x[i]*b[i]);negative=std::max(negative,-x[i]);upper_bad=std::max(upper_bad,x[i]-static_cast<double>(upper[i]));}
 const bool independent=std::isfinite(residual)&&std::isfinite(energy)&&residual<=1e-8&&negative==0&&upper_bad==0&&energy<=1e-8*energy_scale;
 std::cout<<json({{"independent_normal_accepted",independent},{"independent_residual",residual},{"energy_bound_J",energy},{"energy_scale",energy_scale},{"capture",argv[file]},{"accepted",success},{"residual",stats.residual},{"attempts",stats.attempts},{"svd_calls",stats.svd_calls},{"released",stats.released},{"p_normal",x},{"w_normal",w}}).dump()<<"\n";
}}
