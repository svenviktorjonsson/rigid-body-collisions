// Search-only support restriction. Full original gate accepts, never a reduced gate alone.
#pragma once
#include "coulomb.h"
#include "projection_more.h"
inline bool researchLongPGS(const btMatrixXu& A,const btVectorXu& b,btVectorXu& x,
 const btVectorXu& lo,const btVectorXu& hi,const btAlignedObjectArray<int>& dep,
 int budget,double tolerance,CoulombStats& stats,std::vector<double>* rejected_impulses=nullptr){
 const int n=b.rows();
 struct Contact {int normal;std::vector<int> tangent;double mu=0;};
 std::vector<Contact> contacts;std::vector<int> map(n,-1);
 for(int i=0;i<n;i++)if(dep[i]<0){
  if(lo[i]!=0||hi[i]<1e9||!(A(i,i)>0))throw std::runtime_error("Unsupported Coulomb normal row");
  map[i]=static_cast<int>(contacts.size());contacts.push_back({i,{},0});
 }
 for(int i=0;i<n;i++)if(dep[i]>=0){
  if(dep[i]>=n||map[dep[i]]<0||lo[i]!=-hi[i]||hi[i]<0)throw std::runtime_error("Unsupported Coulomb tangent row");
  auto& c=contacts[map[dep[i]]];
  if(!c.tangent.empty()&&hi[i]!=c.mu)throw std::runtime_error("Anisotropic coefficients are not supported");
  c.mu=hi[i];c.tangent.push_back(i);
 }
 for(auto& c:contacts)if(c.tangent.size()!=2)throw std::runtime_error("Two tangents per contact required");
 // Exact zeros only: retain every normal/tangent and inter-contact coupling.
 // Dense Bullet assembly is still used; iterative propagation uses sparse columns.
 std::vector<std::vector<std::pair<int,double>>> columns(n);
 for(int i=0;i<n;i++)for(int j=0;j<n;j++)if(A(i,j)!=0)columns[j].push_back({i,A(i,j)});
 std::vector<double> p(n),w(n);
 for(auto& c:contacts){
  p[c.normal]=std::max(0.,static_cast<double>(x[c.normal]));
  int t=c.tangent[0],s=c.tangent[1];p[t]=x[t];p[s]=x[s];
  double cap=c.mu*p[c.normal],length=std::hypot(p[t],p[s]);
  if(length>cap){p[t]*=cap/length;p[s]*=cap/length;}
 }
 auto recompute=[&](){for(int i=0;i<n;i++)w[i]=-b[i];for(int j=0;j<n;j++)for(auto e:columns[j])w[e.first]+=e.second*p[j];};
 auto update=[&](int j,double value){double delta=value-p[j];p[j]=value;for(auto e:columns[j])w[e.first]+=e.second*delta;};
 auto residual=[&](){
  double maximum=0;
  for(int i=0;i<n;i++)if(!std::isfinite(p[i])||!std::isfinite(w[i]))return std::numeric_limits<double>::infinity();
  for(auto& c:contacts){
   int k=c.normal,t=c.tangent[0],s=c.tangent[1];double a=A(t,t),d=A(s,s),off=A(t,s);
   double eigen=.5*(a+d+std::hypot(a-d,2*off));
   if(!(eigen>0))throw std::runtime_error("Degenerate Coulomb tangent mobility");
   maximum=std::max(maximum,std::abs(p[k]-std::max(0.,p[k]-w[k]/A(k,k)))*A(k,k));
   double zt=p[t]-w[t]/eigen,zs=p[s]-w[s]/eigen;
   double length=std::hypot(zt,zs),cap=c.mu*p[k],factor=length>cap?cap/length:1.;
   maximum=std::max(maximum,std::hypot(p[t]-zt*factor,p[s]-zs*factor)*eigen);
  }
  return maximum;
 };
 // Semismooth Newton acceleration uses the same circular-contact equations.
 // Free impulse coordinates are set to zero at rank-deficient faces; no compliance is added.
 auto newton=[&](){
  if(n>512)return false;
  std::vector<double> F(n,0),J(n*n,0);
  for(auto& c:contacts){
   int k=c.normal,t=c.tangent[0],s=c.tangent[1];double rho=1/A(k,k),zn=p[k]-rho*w[k];
   F[k]=(p[k]-std::max(0.,zn))/rho;
   for(int j=0;j<n;j++)J[k*n+j]=zn>0?A(k,j):(j==k?1/rho:0);
   double a=A(t,t),d=A(s,s),off=A(t,s);rho=2/(a+d+std::hypot(a-d,2*off));
   double z[2]={p[t]-rho*w[t],p[s]-rho*w[s]},length=std::hypot(z[0],z[1]),cap=c.mu*std::max(0.,p[k]);int rows[2]={t,s};
   if(length<=cap&&cap>0){
    for(int r:rows){F[r]=w[r];for(int j=0;j<n;j++)J[r*n+j]=A(r,j);}
   }else{
    double direction[2]={length>0?z[0]/length:0,length>0?z[1]/length:0};
    for(int u=0;u<2;u++){
     int r=rows[u];F[r]=(p[r]-cap*direction[u])/rho;
     for(int j=0;j<n;j++){
      double v=j==r?1.:0.;
      for(int h=0;h<2;h++){
       double dp=length>0?cap/length*((u==h?1.:0.)-direction[u]*direction[h]):0;
       v-=dp*((j==rows[h]?1.:0.)-rho*A(rows[h],j));
      }
      if(j==k&&p[k]>0)v-=c.mu*direction[u];
      J[r*n+j]=v/rho;
     }
    }
   }
  }
  double merit=0,scale=0;for(double f:F)merit+=f*f;for(double v:J)scale=std::max(scale,std::abs(v));
  auto rhs=F;for(int i=0;i<n;i++){rhs[i]=-F[i];for(int j=0;j<n;j++)rhs[i]+=J[i*n+j]*p[j];}
  std::vector<int> pivots;int rank=0;
  for(int col=0;col<n;col++){
   int row=rank;for(int r=rank;r<n;r++)if(std::abs(J[r*n+col])>std::abs(J[row*n+col]))row=r;
   if(std::abs(J[row*n+col])<=1e-12*scale)continue;
   for(int j=col;j<n;j++)std::swap(J[rank*n+j],J[row*n+j]);
   std::swap(rhs[rank],rhs[row]);
   for(int r=rank+1;r<n;r++){
    double factor=J[r*n+col]/J[rank*n+col];J[r*n+col]=0;
    for(int j=col+1;j<n;j++)J[r*n+j]-=factor*J[rank*n+j];
    rhs[r]-=factor*rhs[rank];
   }
   pivots.push_back(col);if(++rank==n)break;
  }
  std::vector<double> step(n,0),old=p;
  for(int i=rank-1;i>=0;i--){int col=pivots[i];double value=rhs[i];for(int j=col+1;j<n;j++)value-=J[i*n+j]*step[j];step[col]=value/J[i*n+col];}
  for(int i=0;i<n;i++)step[i]-=old[i];
  for(int line=0;line<24;line++){
   double alpha=std::ldexp(1.,-line);for(int i=0;i<n;i++)p[i]=old[i]+alpha*step[i];
   for(auto& c:contacts)p[c.normal]=std::max(0.,p[c.normal]);
   recompute();double trial=0;
   for(auto& c:contacts){
    int k=c.normal,t=c.tangent[0],ss=c.tangent[1];double a=A(t,t),d=A(ss,ss),off=A(t,ss),eigen=.5*(a+d+std::hypot(a-d,2*off));
    double fn=(p[k]-std::max(0.,p[k]-w[k]/A(k,k)))*A(k,k);trial+=fn*fn;
    double zt=p[t]-w[t]/eigen,zs=p[ss]-w[ss]/eigen,length=std::hypot(zt,zs),cap=c.mu*p[k],factor=length>cap?cap/length:1.;
    double ft=(p[t]-zt*factor)*eigen,fs=(p[ss]-zs*factor)*eigen;trial+=ft*ft+fs*fs;
   }
   if(std::isfinite(trial)&&trial<(1-1e-4*alpha)*merit){stats.newton_steps++;return true;}
  }
  p=old;recompute();return false;
 };
 recompute();
 for(int sweep=0;sweep<=budget;sweep++){
  if(sweep%8==0||sweep==budget){
   recompute();double error=residual();stats.last_residual=error;
   if(error<=tolerance){
    double change=0,scale=1;
    for(int i=0;i<n;i++){change+=.5*p[i]*(w[i]-b[i]);scale+=std::abs(p[i]*b[i]);}
    // With e=0 and separate position correction this is an upper bound on
    // E_after-E_before-W_wall. Positive-gap targets add a conservative term.
    if(!std::isfinite(change)||!std::isfinite(scale)||change>tolerance*scale)throw std::runtime_error("Coulomb passivity gate failed");
    for(auto& c:contacts)if(p[c.normal]>hi[c.normal]){if(rejected_impulses)*rejected_impulses=p;return false;}
    for(int i=0;i<n;i++)x[i]=p[i];
    stats.solves++;stats.sweeps_max=std::max(stats.sweeps_max,sweep);
    stats.fast_solves+=sweep<=8;stats.residual_max=std::max(stats.residual_max,error);
    stats.passive_change_max=std::max(stats.passive_change_max,change);return true;
   }
   if(sweep==budget){if(rejected_impulses)*rejected_impulses=p;return false;}
  }
  stats.iteration_sweeps_total++;
  // Pure bounded PGS diagnostic: no Newton acceleration.
  // Coupled block Gauss-Seidel: solve a normal, then its circular tangent block.
  for(auto& c:contacts){
   int k=c.normal,t=c.tangent[0],s=c.tangent[1];
   update(k,std::max(0.,p[k]-w[k]/A(k,k)));
   double cap=c.mu*p[k];
   if(cap==0){update(t,0);update(s,0);continue;}
   double a=A(t,t),d=A(s,s),off=A(t,s);
   double qt=w[t]-a*p[t]-off*p[s],qs=w[s]-off*p[t]-d*p[s];
   auto solve=[&](double lambda){double aa=a+lambda,dd=d+lambda,det=aa*dd-off*off;
    if(!(det>0))throw std::runtime_error("Singular tangent block");
    return std::pair<double,double>{(-dd*qt+off*qs)/det,(off*qt-aa*qs)/det};};
   auto answer=solve(0);
   if(std::hypot(answer.first,answer.second)>cap){
    double low=0,high=std::hypot(qt,qs)/cap;
    // high bounds the Lagrange multiplier for the circular trust region.
    for(int j=0;j<48;j++){double mid=.5*(low+high);auto trial=solve(mid);if(std::hypot(trial.first,trial.second)>cap)low=mid;else high=mid;}
    answer=solve(high);
   }
   update(t,answer.first);update(s,answer.second);
  }
 }
 return false;
}

namespace long_pgs_research {
struct Stats {int components=0,largest_rows=0,stage_attempts=0,stage_accepts=0,iteration_steps=0,svd_calls=0,support_passes=0,reduced_rows_max=0;};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&p,const btVectorXu&lo,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats){
 int n=b.rows();if(n<=0||n>4096)return false;std::vector<bool>visited(n,false);btVectorXu candidate=p;
 auto gate=[&](const btMatrixXu&M,const btVectorXu&rhs,btVectorXu&q,const btVectorXu&lower,const btVectorXu&upper,const btAlignedObjectArray<int>&d){CoulombStats s;return coulombIterate(M,rhs,q,lower,upper,d,0,tol,s);};
 for(int first=0;first<n;first++)if(!visited[first]){
  std::vector<int>ids{first};visited[first]=true;
  for(size_t at=0;at<ids.size();at++)for(int j=0;j<n;j++)if(!visited[j]&&(A(ids[at],j)!=0||A(j,ids[at])!=0||dep[j]==ids[at]||dep[ids[at]]==j)){visited[j]=true;ids.push_back(j);}
  int m=ids.size();stats.components++;stats.largest_rows=std::max(stats.largest_rows,m);std::vector<int>inverse(n,-1);for(int i=0;i<m;i++)inverse[ids[i]]=i;
  btMatrixXu M(m,m);btVectorXu rhs(m),q(m),lower(m),upper(m);btAlignedObjectArray<int>d;d.resize(m);
  for(int i=0;i<m;i++){rhs[i]=b[ids[i]];q[i]=candidate[ids[i]];lower[i]=lo[ids[i]];upper[i]=hi[ids[i]];d[i]=dep[ids[i]]<0?-1:inverse[dep[ids[i]]];for(int j=0;j<m;j++)M.setElem(i,j,A(ids[i],ids[j]));}
  btVectorXu checked=q;if(gate(M,rhs,checked,lower,upper,d)){for(int i=0;i<m;i++)candidate[ids[i]]=checked[i];continue;}
  std::vector<bool>active(m,m<=192);std::vector<int>normals;for(int k=0;k<m;k++)if(d[k]<0){normals.push_back(k);double w=-rhs[k];for(int j=0;j<m;j++)w+=M(k,j)*q[j];if(q[k]>tol/M(k,k)||w<-tol)active[k]=true;}
  bool found=false;
  for(int pass=0;pass<8&&!found;pass++){
   stats.support_passes++;std::vector<int>rows;
   for(int i=0;i<m;i++)if(d[i]<0?active[i]:active[d[i]])rows.push_back(i);
   int r=rows.size();stats.reduced_rows_max=std::max(stats.reduced_rows_max,r);if(r<=0||r>192)return false;
   std::vector<int>inv(m,-1);for(int i=0;i<r;i++)inv[rows[i]]=i;
   btMatrixXu R(r,r);btVectorXu rr(r),rp(r),rl(r),rh(r);btAlignedObjectArray<int>rd;rd.resize(r);
   for(int i=0;i<r;i++){rr[i]=rhs[rows[i]];rp[i]=q[rows[i]];rl[i]=lower[rows[i]];rh[i]=upper[rows[i]];rd[i]=d[rows[i]]<0?-1:inv[d[rows[i]]];for(int j=0;j<r;j++)R.setElem(i,j,M(rows[i],rows[j]));}
   CoulombStats trial;std::vector<double>last;stats.stage_attempts++;bool ok=researchLongPGS(R,rr,rp,rl,rh,rd,100000,tol,trial,&last);stats.stage_accepts+=ok;stats.iteration_steps+=trial.iteration_sweeps_total;
   if(!ok&&last.size()==static_cast<size_t>(r))for(int i=0;i<r;i++)rp[i]=last[i];
   q.setZero();for(int i=0;i<r;i++)q[rows[i]]=rp[i];found=gate(M,rhs,q,lower,upper,d);
   if(found)break;bool grew=false;for(int k:normals)if(!active[k]){double w=-rhs[k];for(int j=0;j<m;j++)w+=M(k,j)*q[j];if(w<-tol){active[k]=true;grew=true;}}
   if(!grew)return false;
  }
  if(!found)return false;for(int i=0;i<m;i++)candidate[ids[i]]=q[i];
 }
 if(!gate(A,b,candidate,lo,hi,dep))return false;p=candidate;return true;
}
}
