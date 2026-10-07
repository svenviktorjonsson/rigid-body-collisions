#include "normal_null.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
using json=nlohmann::json;
int main(int argc,char**argv){bool all=true;for(int f=1;f<argc;f++){
 std::ifstream input(argv[f]);json d;input>>d;std::vector<int>ks;int full=d["b"].size();for(int i=0;i<full;i++)if(d["dependencies"][i].get<int>()<0)ks.push_back(i);int n=ks.size();btMatrixXu A(n,n);btVectorXu b(n),upper(n),seed(n),p(n);for(int i=0;i<n;i++){b[i]=d["b"][ks[i]];upper[i]=d["hi"][ks[i]];seed[i]=p[i]=d["p"][ks[i]];for(int j=0;j<n;j++)A.setElem(i,j,d["A"][ks[i]][ks[j]]);}
 normal_null::Stats stats;double tol=d["tolerance_m_s"];bool accepted=normal_null::solve(A,b,upper,seed,p,tol,stats);std::vector<double>x(full,0),w(full),normal(n);for(int i=0;i<n;i++)x[ks[i]]=normal[i]=p[i];double error=0,energy=0,scale=1;bool bounds=true;
 for(int i=0;i<full;i++){w[i]=-d["b"][i].get<double>();for(int j=0;j<full;j++)w[i]+=d["A"][i][j].get<double>()*x[j];energy+=.5*x[i]*(w[i]-d["b"][i].get<double>());scale+=std::abs(x[i]*d["b"][i].get<double>());}
 for(int k:ks){double diag=d["A"][k][k];error=std::max(error,std::abs(x[k]-std::max(0.,x[k]-w[k]/diag))*diag);bounds&=x[k]>=0&&x[k]<=d["hi"][k].get<double>();std::vector<int>ts;for(int j=0;j<full;j++)if(d["dependencies"][j].get<int>()==k){ts.push_back(j);if(d["hi"][j].get<double>()!=0)throw std::runtime_error("normal-only capacity required");}if(ts.size()!=2)throw std::runtime_error("normal and two tangent rows required");error=std::max(error,std::hypot(x[ts[0]],x[ts[1]]));}
 bool independent=bounds&&std::isfinite(error)&&std::isfinite(energy)&&std::isfinite(scale)&&error<=tol&&energy<=tol*scale;all&=accepted&&independent;
 std::cout<<json({{"capture",argv[f]},{"accepted",accepted},{"independent_full_accepted",independent},{"full_residual_m_s",error},{"passive_change_bound_J",energy},{"states",stats.states},{"svd_calls",stats.svd_calls},{"null_steps",stats.null_steps},{"range_steps",stats.range_steps},{"released",stats.released},{"entered",stats.entered},{"svd_rejections",stats.svd_rejections},{"budget_rejections",stats.budget_rejections},{"maximum_null_velocity_change_m_s",stats.maximum_null_velocity_change},{"normal_impulse",normal}}).dump()<<'\n';
 }return all?0:1;}
