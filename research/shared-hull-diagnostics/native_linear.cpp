// Independently exported projection Jacobian: numerical subproblem only.
#include "newton_linear_source.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=2)return 2;std::ifstream file(argv[1]);nlohmann::json data;file>>data;
 int n=data.at("rhs").size();auto matrix=data.at("matrix").get<std::vector<double>>();auto rhs=data.at("rhs").get<std::vector<double>>();
 auto result=minimumNormNewton(matrix,rhs,n);double norm=0;for(double x:result.step)norm+=x*x;
 std::cout<<nlohmann::json({{"rank",result.rank},{"converged",result.converged},{"column_correlation",result.column_correlation},{"step_norm",std::sqrt(norm)},{"step",result.step}}).dump()<<'\n';
}
