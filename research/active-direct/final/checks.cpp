#include "coulomb_restart.h"
#include <iostream>
#include <stdexcept>
void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
int main(){
 auto step=restart_spectral::spectralMoreStep({1,0,0,0},{1,0},2,.5,-1);require(step.converged&&std::abs(step.norm-.5)<1e-14&&step.step[0]>.49,"rank deficient trust step");
 auto weak=restart_spectral::spectralMoreStep({1,0,0,1e-12},{0,1e-12},2,2,-1);require(weak.converged&&std::abs(weak.step[1]-1)<1e-12,"weak retained singular direction");
 require(!restart_spectral::spectralMoreStep({1,0},{1},2,1,-1).converged,"shape rejection");
 require(!restart_spectral::spectralMoreStep({1},{1},1,0,-1).converged,"radius rejection");
 require(!restart_spectral::spectralMoreStep({std::numeric_limits<double>::quiet_NaN()},{1},1,1,-1).converged,"nonfinite matrix rejection");
 btMatrixXu A(3,3);A.setZero();for(int i=0;i<3;i++)A.setElem(i,i,1);
 btVectorXu b(3),p(3),hi(3);b[0]=1;b[1]=.1;b[2]=.2;p.setZero();hi[0]=1e10;hi[1]=hi[2]=.4;
 btAlignedObjectArray<int>dep;dep.resize(3);dep[0]=-1;dep[1]=dep[2]=0;
 circular_restart::Stats stats;require(circular_restart::solve(A,b,p,hi,dep,1e-8,stats),"analytical sticking contact");require(std::abs(p[0]-1)<1e-8&&std::abs(p[1]-.1)<1e-8&&std::abs(p[2]-.2)<1e-8,"analytical sticking impulse");
 const auto saved=p;restart_neutral::Stats ns;require(!restart_neutral::solve(A,b,p,hi,dep,1e-8,ns,0),"zero budget rejection");require(ns.svd_calls==0&&p[0]==saved[0]&&p[1]==saved[1]&&p[2]==saved[2],"zero budget unchanged output");
 dep[1]=5;require(!restart_neutral::solve(A,b,p,hi,dep,1e-8,ns),"dependency bounds rejection");require(p[0]==saved[0]&&p[1]==saved[1]&&p[2]==saved[2],"invalid dependency unchanged output");dep[1]=0;
 b[1]=std::numeric_limits<double>::quiet_NaN();require(!restart_neutral::solve(A,b,p,hi,dep,1e-8,ns),"nonfinite rhs rejection");require(p[0]==saved[0]&&p[1]==saved[1]&&p[2]==saved[2],"nonfinite unchanged output");b[1]=.1;
 circular_restart::Stats zero;require(!circular_restart::solve(A,b,p,hi,dep,1e-8,zero,0)&&zero.svd_calls==0,"wrapper zero budget rejection");
 std::cout<<"PASS: rank-deficient and weak-direction TRF; invalid shape/radius/nonfinite; analytical contact; budget/dependency/output preservation\n";
}
