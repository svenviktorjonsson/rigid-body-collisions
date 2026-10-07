#pragma once
#include <cmath>
namespace restart_validation {
inline bool valid(const btMatrixXu&A,const btVectorXu&b,const btVectorXu&x,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tolerance,int maximum=384){
 const int n=b.rows();if(n<=0||n>maximum||A.rows()!=n||A.cols()!=n||x.rows()!=n||hi.rows()!=n||dep.size()!=n||!(tolerance>0)||!std::isfinite(tolerance))return false;
 for(int i=0;i<n;i++){
  if(!std::isfinite(b[i])||!std::isfinite(x[i])||!std::isfinite(hi[i])||hi[i]<0||dep[i]<-1||dep[i]>=n)return false;
  if(dep[i]>=0&&dep[dep[i]]!=-1)return false;
  for(int j=0;j<n;j++)if(!std::isfinite(A(i,j)))return false;
 }
 for(int k=0;k<n;k++)if(dep[k]==-1){int count=0;double mu=-1;for(int j=0;j<n;j++)if(dep[j]==k){count++;if(mu<0)mu=hi[j];else if(mu!=hi[j])return false;}if(count!=2||!(A(k,k)>0))return false;}
 return true;
}
}
