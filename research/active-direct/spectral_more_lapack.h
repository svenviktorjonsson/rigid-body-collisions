#pragma once
#include <vector>
#include <cmath>
#include <algorithm>
#include <limits>
extern "C" void dgesdd_(const char*,const int*,const int*,double*,const int*,double*,double*,const int*,double*,const int*,double*,const int*,int*,int*);
struct MoreStep {std::vector<double>step;bool converged=false;double norm=0,lambda=0;};
inline MoreStep spectralMoreStep(const std::vector<double>&matrix,const std::vector<double>&rhs,int n,double radius,double initialAlpha){
 MoreStep out;
 if(n<=0||n>384||matrix.size()!=static_cast<size_t>(n*n)||rhs.size()!=static_cast<size_t>(n)||!(radius>0)||!std::isfinite(radius)||!std::isfinite(initialAlpha))return out;
 for(double value:matrix)if(!std::isfinite(value))return out;
 for(double value:rhs)if(!std::isfinite(value))return out;
 out.step.assign(n,0);std::vector<double>C(n*n),s(n),U(n*n),VT(n*n),uf(n),suf(n);std::vector<int>iwork(8*n);for(int i=0;i<n;i++)for(int j=0;j<n;j++)C[i+j*n]=matrix[i*n+j];char job='S';int lwork=-1,info=0;double query=0;
 dgesdd_(&job,&n,&n,C.data(),&n,s.data(),U.data(),&n,VT.data(),&n,&query,&lwork,iwork.data(),&info);if(info||!std::isfinite(query)||query<1||query>std::numeric_limits<int>::max())return out;lwork=static_cast<int>(query);std::vector<double>work(lwork);dgesdd_(&job,&n,&n,C.data(),&n,s.data(),U.data(),&n,VT.data(),&n,work.data(),&lwork,iwork.data(),&info);if(info)return out;
 for(int j=0;j<n;j++){for(int i=0;i<n;i++)uf[j]+=U[i+j*n]*rhs[i];suf[j]=s[j]*uf[j];}
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
