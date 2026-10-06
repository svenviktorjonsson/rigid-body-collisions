#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
// Standard two-phase primal tableau simplex, with explicit shared pivot budget.
// Algorithm reference: Stanford Notebook / KACTL Simplex.h (MIT), revision
// acb548f9e3f4aa792243a4f9929793c56d76d903. See simplex-LICENSE.txt.
namespace bounded_face_lp {
class Solver {
 int constraints,variables;std::vector<int>basis,nonbasis;std::vector<std::vector<double>>table;
 int&work;const int cap;const double epsilon=1e-12;
 bool pivot(int row,int column){
  if(work>=cap||!std::isfinite(table[row][column])||std::abs(table[row][column])<=epsilon)return false;
  ++work;double inverse=1/table[row][column];
  for(int i=0;i<constraints+2;i++)if(i!=row){
   double multiplier=table[i][column]*inverse;
   for(int j=0;j<variables+2;j++)if(j!=column)table[i][j]-=table[row][j]*multiplier;
   table[i][column]=-multiplier;
  }
  for(int j=0;j<variables+2;j++)if(j!=column)table[row][j]*=inverse;
  table[row][column]=inverse;std::swap(basis[row],nonbasis[column]);return true;
 }
 bool optimize(bool artificial){
  int objective=constraints+(artificial?1:0);
  for(;;){
   int entering=-1;
   for(int j=0;j<=variables;j++){
    if(!artificial&&nonbasis[j]<0)continue;
    if(!std::isfinite(table[objective][j]))return false;
    if(entering<0||table[objective][j]<table[objective][entering]-epsilon||
       (std::abs(table[objective][j]-table[objective][entering])<=epsilon&&nonbasis[j]<nonbasis[entering]))entering=j;
   }
   if(entering<0||table[objective][entering]>=-epsilon)return true;
   int leaving=-1;double best=0;
   for(int i=0;i<constraints;i++)if(table[i][entering]>epsilon){
    double ratio=table[i][variables+1]/table[i][entering];if(!std::isfinite(ratio))return false;
    if(leaving<0||ratio<best-epsilon||(std::abs(ratio-best)<=epsilon&&basis[i]<basis[leaving])){leaving=i;best=ratio;}
   }
   if(leaving<0||!pivot(leaving,entering))return false;
  }
 }
public:
 Solver(const std::vector<std::vector<double>>&A,const std::vector<double>&b,const std::vector<double>&objective,int&used,int maximum):constraints(b.size()),variables(objective.size()),basis(constraints),nonbasis(variables+1),table(constraints+2,std::vector<double>(variables+2)),work(used),cap(maximum){
  for(int i=0;i<constraints;i++){double scale=1;for(double value:A[i])scale=std::max(scale,std::abs(value));for(int j=0;j<variables;j++)table[i][j]=A[i][j]/scale;table[i][variables]=-1;table[i][variables+1]=b[i]/scale;basis[i]=variables+i;}
  for(int j=0;j<variables;j++){nonbasis[j]=j;table[constraints][j]=-objective[j];}
  nonbasis[variables]=-1;table[constraints+1][variables]=1;
 }
 bool solve(std::vector<double>&solution){
  if(constraints<=0||constraints>4096||variables<=0||variables>384)return false;
  int row=0;for(int i=1;i<constraints;i++)if(table[i][variables+1]<table[row][variables+1])row=i;
  if(table[row][variables+1]<-epsilon){
   if(!pivot(row,variables)||!optimize(true)||std::abs(table[constraints+1][variables+1])>16*epsilon)return false;
   for(int i=0;i<constraints;i++)if(basis[i]<0){
    int column=-1;for(int j=0;j<=variables;j++)if(std::abs(table[i][j])>epsilon&&(column<0||nonbasis[j]<nonbasis[column]))column=j;
    if(column>=0&&!pivot(i,column))return false;
   }
  }
  if(!optimize(false))return false;solution.assign(variables,0);
  for(int i=0;i<constraints;i++)if(basis[i]>=0&&basis[i]<variables)solution[basis[i]]=table[i][variables+1];
  for(double value:solution)if(!std::isfinite(value)||value<-16*epsilon)return false;
  return true;
 }
};
}
