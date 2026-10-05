#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
struct SpectralStep {std::vector<double>step;bool converged=false;double norm=0,correlation=0,lambda=0;};
inline SpectralStep spectralTrustStep(const std::vector<double>&matrix,const std::vector<double>&rhs,int n,double radius){
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
 std::vector<double>square(n,0),product(n,0);double largest=0;for(int j=0;j<n;j++){for(int i=0;i<n;i++){square[j]+=C[i*n+j]*C[i*n+j];product[j]+=C[i*n+j]*rhs[i];}largest=std::max(largest,square[j]);}
 const double cutoff=1e-26*largest;auto norm2=[&](double lambda){double norm=0;for(int j=0;j<n;j++)if(square[j]>cutoff){double value=product[j]/(square[j]+lambda);norm+=value*value;}return norm;};
 double lambda=0;if(norm2(0)>radius*radius){double low=0,high=std::max(1.,largest);for(int i=0;i<32&&norm2(high)>radius*radius;i++)high*=4;for(int i=0;i<64;i++){double middle=.5*(low+high);if(norm2(middle)>radius*radius)low=middle;else high=middle;}lambda=high;}
 SpectralStep out;out.lambda=lambda;out.step.assign(n,0);for(int j=0;j<n;j++)if(square[j]>cutoff){double coefficient=product[j]/(square[j]+lambda);for(int i=0;i<n;i++)out.step[i]+=V[i*n+j]*coefficient;}
 for(int p=0;p<n;p++)for(int q=p+1;q<n;q++)if(std::min(square[p],square[q])>cutoff){double g=0;for(int i=0;i<n;i++)g+=C[i*n+p]*C[i*n+q];out.correlation=std::max(out.correlation,std::abs(g)/std::sqrt(square[p]*square[q]));}
 for(double v:out.step)out.norm+=v*v;out.norm=std::sqrt(out.norm);out.converged=std::isfinite(out.norm)&&std::isfinite(out.correlation)&&out.correlation<=1e-10;return out;
}
