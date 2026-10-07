#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
struct MoreStep {std::vector<double>step;bool converged=false;double norm=0,lambda=0;};
inline MoreStep spectralMoreStep(const std::vector<double>&matrix,const std::vector<double>&rhs,int n,double radius,double initialAlpha){
 MoreStep out;out.step.assign(n,0);auto C=matrix;std::vector<double>V(n*n,0),s(n),uf(n),suf(n),VT(n*n);for(int i=0;i<n;i++)V[i*n+i]=1;
 double frobenius=0;for(double value:C)frobenius+=value*value;
 for(int sweep=0;sweep<64;sweep++){
  bool changed=false;for(int p=0;p<n;p++)for(int q=p+1;q<n;q++){
   double a=0,d=0,g=0;for(int i=0;i<n;i++){a+=C[i*n+p]*C[i*n+p];d+=C[i*n+q]*C[i*n+q];g+=C[i*n+p]*C[i*n+q];}
   if(std::min(a,d)<=1e-28*frobenius||std::abs(g)<=2e-14*std::sqrt(a*d))continue;
   double tau=(d-a)/(2*g),t=std::copysign(1.,tau)/(std::abs(tau)+std::hypot(1.,tau));double c=1/std::hypot(1.,t),ss=c*t;
   for(int i=0;i<n;i++){double x=C[i*n+p],y=C[i*n+q];C[i*n+p]=c*x-ss*y;C[i*n+q]=ss*x+c*y;x=V[i*n+p];y=V[i*n+q];V[i*n+p]=c*x-ss*y;V[i*n+q]=ss*x+c*y;}changed=true;
  }if(!changed)break;
 }
 std::vector<double>norm(n,0);std::vector<int>order(n);for(int j=0;j<n;j++){order[j]=j;for(int i=0;i<n;i++)norm[j]+=C[i*n+j]*C[i*n+j];norm[j]=std::sqrt(norm[j]);}
 std::stable_sort(order.begin(),order.end(),[&](int a,int b){return norm[a]>norm[b];});
 double correlation=0;for(int a=0;a<n;a++)for(int b=a+1;b<n;b++)if(norm[a]>1e-13*norm[order[0]]&&norm[b]>1e-13*norm[order[0]]){double product=0;for(int i=0;i<n;i++)product+=C[i*n+a]*C[i*n+b];correlation=std::max(correlation,std::abs(product)/(norm[a]*norm[b]));}if(!std::isfinite(correlation)||correlation>1e-10)return out;
 for(int j=0;j<n;j++){int source=order[j];s[j]=norm[source];for(int i=0;i<n;i++){if(s[j]>0)uf[j]+=C[i*n+source]*rhs[i]/s[j];VT[j+i*n]=V[i*n+source];}suf[j]=s[j]*uf[j];}
 bool fullRank=s[n-1]>std::numeric_limits<double>::epsilon()*n*s[0];double alpha=0;
 auto reconstruct=[&](double lambda,bool gauss){std::fill(out.step.begin(),out.step.end(),0);for(int j=0;j<n;j++){double coeff=gauss?uf[j]/s[j]:suf[j]/(s[j]*s[j]+lambda);for(int i=0;i<n;i++)out.step[i]+=VT[j+i*n]*coeff;}double norm=0;for(double v:out.step)norm+=v*v;return std::sqrt(norm);};
 if(fullRank){out.norm=reconstruct(0,true);if(out.norm<=radius){out.converged=std::isfinite(out.norm);return out;}}
 auto phi=[&](double lambda){double norm2=0,numerator=0;for(int j=0;j<n;j++){double denom=s[j]*s[j]+lambda,value=suf[j]/denom;norm2+=value*value;numerator+=value*value/denom;}double norm=std::sqrt(norm2);return std::pair<double,double>{norm-radius,-numerator/norm};};
 double gradient=0;for(double v:suf)gradient+=v*v;double upper=std::sqrt(gradient)/radius,lower=0;if(!(upper>0&&std::isfinite(upper)))return out;
 if(fullRank){auto v=phi(0);lower=-v.first/v.second;}
 alpha=initialAlpha<0||(!fullRank&&initialAlpha==0)?std::max(.001*upper,std::sqrt(std::max(0.,lower*upper))):initialAlpha;
 for(int it=0;it<10;it++){
  if(alpha<lower||alpha>upper)alpha=std::max(.001*upper,std::sqrt(std::max(0.,lower*upper)));
  auto v=phi(alpha);if(!std::isfinite(v.first)||!std::isfinite(v.second)||v.second==0)return out;if(v.first<0)upper=alpha;double ratio=v.first/v.second;lower=std::max(lower,alpha-ratio);alpha-=(v.first+radius)*ratio/radius;if(std::abs(v.first)<.01*radius)break;
 }
 out.norm=reconstruct(alpha,false);if(!(out.norm>0&&std::isfinite(out.norm)))return out;for(double&v:out.step)v*=radius/out.norm;out.norm=radius;out.lambda=alpha;out.converged=std::isfinite(alpha);for(double v:out.step)out.converged&=std::isfinite(v);return out;
}
