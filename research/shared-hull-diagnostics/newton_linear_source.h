// Numerical Newton subproblem only: never regularize the physical mobility.
#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
struct NewtonLinearSolution {std::vector<double> step;std::vector<std::vector<double>> nullspace;int rank=0;bool converged=false;double column_correlation=0;};
inline NewtonLinearSolution minimumNormNewton(const std::vector<double>& matrix,const std::vector<double>& rhs,int n){
 auto C=matrix;std::vector<double> V(n*n,0);for(int i=0;i<n;i++)V[i*n+i]=1;
 double frobenius=0;for(double value:C)frobenius+=value*value;
 // One-sided Jacobi orthogonalizes columns without squaring the condition
 // number. The right rotations expose null directions for pressure relocation.
 for(int sweep=0;sweep<64;sweep++){
  bool changed=false;
  for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){
   double a=0,d=0,g=0;for(int i=0;i<n;i++){a+=C[i*n+p]*C[i*n+p];d+=C[i*n+q]*C[i*n+q];g+=C[i*n+p]*C[i*n+q];}
   if(std::min(a,d)<=1e-28*frobenius||std::abs(g)<=2e-14*std::sqrt(a*d))continue;
   double tau=(d-a)/(2*g),t=std::copysign(1.,tau)/(std::abs(tau)+std::hypot(1.,tau));
   double c=1/std::hypot(1.,t),s=c*t;
   for(int i=0;i<n;i++){double x=C[i*n+p],y=C[i*n+q];C[i*n+p]=c*x-s*y;C[i*n+q]=s*x+c*y;x=V[i*n+p];y=V[i*n+q];V[i*n+p]=c*x-s*y;V[i*n+q]=s*x+c*y;}
   changed=true;
  }
  if(!changed)break;
 }
 std::vector<double> square(n,0);double largest=0;for(int j=0;j<n;j++){for(int i=0;i<n;i++)square[j]+=C[i*n+j]*C[i*n+j];largest=std::max(largest,square[j]);}
 NewtonLinearSolution result;result.step.assign(n,0);
 for(int j=0;j<n;j++){
  if(square[j]>1e-24*largest){
   double product=0;for(int i=0;i<n;i++)product+=C[i*n+j]*rhs[i];
   for(int i=0;i<n;i++)result.step[i]+=V[i*n+j]*product/square[j];
   result.rank++;
  }else{std::vector<double> direction(n);for(int i=0;i<n;i++)direction[i]=V[i*n+j];result.nullspace.push_back(direction);}
 }
 for(int p=0;p<n;p++)for(int q=p+1;q<n;q++)if(std::min(square[p],square[q])>1e-24*largest){
  double g=0;for(int i=0;i<n;i++)g+=C[i*n+p]*C[i*n+q];result.column_correlation=std::max(result.column_correlation,std::abs(g)/std::sqrt(square[p]*square[q]));
 }
 result.converged=std::isfinite(result.column_correlation)&&result.column_correlation<=1e-10;
 for(double value:result.step)result.converged&=std::isfinite(value);
 return result;
}
