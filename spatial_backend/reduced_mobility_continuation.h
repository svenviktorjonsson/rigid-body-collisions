// Search-only support restriction. Full original gate accepts, never a reduced gate alone.
#pragma once
// The caller supplies the original authoritative production gate.
#ifdef SPATIAL_LAPACK_RECOVERY
#include "projection_more.h"
#endif
namespace reduced_mobility_continuation {
struct Stats {int components=0,largest_rows=0,stage_attempts=0,stage_accepts=0,iteration_steps=0,svd_calls=0,support_passes=0,reduced_rows_max=0;};
#ifdef SPATIAL_LAPACK_RECOVERY
template<class OriginalGate>
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&p,const btVectorXu&lo,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats,OriginalGate&&original_gate){
 int n=b.rows();if(n<=0||n>4096)return false;std::vector<bool>visited(n,false);btVectorXu candidate=p;
 auto gate=[&](const btMatrixXu&M,const btVectorXu&rhs,btVectorXu&q,const btVectorXu&lower,const btVectorXu&upper,const btAlignedObjectArray<int>&d){return original_gate(M,rhs,q,lower,upper,d,tol);};
 for(int first=0;first<n;first++)if(!visited[first]){
  std::vector<int>ids{first};visited[first]=true;
  for(size_t at=0;at<ids.size();at++)for(int j=0;j<n;j++)if(!visited[j]&&(A(ids[at],j)!=0||A(j,ids[at])!=0||dep[j]==ids[at]||dep[ids[at]]==j)){visited[j]=true;ids.push_back(j);}
  int m=ids.size();stats.components++;stats.largest_rows=std::max(stats.largest_rows,m);std::vector<int>inverse(n,-1);for(int i=0;i<m;i++)inverse[ids[i]]=i;
  btMatrixXu M(m,m);btVectorXu rhs(m),q(m),lower(m),upper(m);btAlignedObjectArray<int>d;d.resize(m);
  for(int i=0;i<m;i++){rhs[i]=b[ids[i]];q[i]=candidate[ids[i]];lower[i]=lo[ids[i]];upper[i]=hi[ids[i]];d[i]=dep[ids[i]]<0?-1:inverse[dep[ids[i]]];for(int j=0;j<m;j++)M.setElem(i,j,A(ids[i],ids[j]));}
  btVectorXu checked=q;if(gate(M,rhs,checked,lower,upper,d)){for(int i=0;i<m;i++)candidate[ids[i]]=checked[i];continue;}
  if(m<=192)return false; // Only new large-component support reduction.
  std::vector<bool>active(m,false);std::vector<int>normals;for(int k=0;k<m;k++)if(d[k]<0){normals.push_back(k);double w=-rhs[k];for(int j=0;j<m;j++)w+=M(k,j)*q[j];if(q[k]>tol/M(k,k)||w<-tol)active[k]=true;}
  bool found=false;
  for(int pass=0;pass<8&&!found;pass++){
   stats.support_passes++;std::vector<int>rows;
   for(int i=0;i<m;i++)if(d[i]<0?active[i]:active[d[i]])rows.push_back(i);
   int r=rows.size();stats.reduced_rows_max=std::max(stats.reduced_rows_max,r);if(r<=0||r>192)return false;
   std::vector<int>inv(m,-1);for(int i=0;i<r;i++)inv[rows[i]]=i;
   btMatrixXu R(r,r);btVectorXu rr(r),rp(r),rl(r),rh(r);btAlignedObjectArray<int>rd;rd.resize(r);
   for(int i=0;i<r;i++){rr[i]=rhs[rows[i]];rp[i]=q[rows[i]];rl[i]=lower[rows[i]];rh[i]=upper[rows[i]];rd[i]=d[rows[i]]<0?-1:inv[d[rows[i]]];for(int j=0;j<r;j++)R.setElem(i,j,M(rows[i],rows[j]));}
   for(double alpha:{.1,.03,.01,.003,.001,.0003,.0001,.00003,.00001,1e-6,1e-7,1e-8,1e-9,1e-10,1e-11,1e-12,0.}){
    btMatrixXu search=R;for(int i=0;i<r;i++)search.setElem(i,i,R(i,i)*(1+alpha));projection_recovery_v2::Stats s;stats.stage_attempts++;bool ok=projection_recovery_v2::solve(search,rr,rp,rh,rd,tol,s,2048,2048,true,true,true,192);stats.stage_accepts+=ok;stats.iteration_steps+=s.iteration_steps;stats.svd_calls+=s.svd_calls;if(!ok&&s.failed_candidate.size()==static_cast<size_t>(r))for(int i=0;i<r;i++)rp[i]=s.failed_candidate[i];
    for(int k=0;k<r;k++)if(rd[k]<0){rp[k]=std::max(0.,static_cast<double>(rp[k]));std::vector<int>ts;for(int j=0;j<r;j++)if(rd[j]==k)ts.push_back(j);if(ts.size()!=2)return false;double length=std::hypot(rp[ts[0]],rp[ts[1]]),cap=rh[ts[0]]*rp[k];if(length>cap){rp[ts[0]]*=cap/length;rp[ts[1]]*=cap/length;}}
   }
   q.setZero();for(int i=0;i<r;i++)q[rows[i]]=rp[i];found=gate(M,rhs,q,lower,upper,d);
   if(found)break;bool grew=false;for(int k:normals)if(!active[k]){double w=-rhs[k];for(int j=0;j<m;j++)w+=M(k,j)*q[j];if(w<-tol){active[k]=true;grew=true;}}
   if(!grew)return false;
  }
  if(!found)return false;for(int i=0;i<m;i++)candidate[ids[i]]=q[i];
 }
 if(!gate(A,b,candidate,lo,hi,dep))return false;p=candidate;return true;
}
#endif
}
