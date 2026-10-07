#include "spectral_more_lapack.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
using json=nlohmann::json;
int main(int argc,char**argv){std::ifstream f(argv[1]);json d;f>>d;std::vector<double>rhs=d["rhs"],J;for(auto row:d["J"])for(auto v:row)J.push_back(v);auto s=spectralMoreStep(J,rhs,rhs.size(),d["radius"],-1);std::cout<<json({{"step",s.step},{"lambda",s.lambda},{"norm",s.norm},{"converged",s.converged}}).dump()<<"\n";}
