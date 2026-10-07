// Isolated CMINPACK numerical search; original physical system/gate unchanged.
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include "coulomb.h"
#include "spectral_more.h"
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
using json=nlohmann::json;
struct Context{
 btMatrixXu A;btVectorXu b,lo,hi;btAlignedObjectArray<int>dep;double tolerance;int calls=0,limit=2048;std::vector<double>best,solution;double best_score=BT_LARGE_FLOAT;
 struct Contact{int k,t,s;double eig,mu;};std::vector<Contact>contacts;
 Context(const json&d):A(d["b"].size(),d["b"].size()),b(d["b"].size()),lo(d["b"].size()),hi(d["b"].size()),tolerance(d["tolerance_m_s"]){const int n=b.rows();dep.resize(n);for(int i=0;i<n;i++){b[i]=d["b"][i];lo[i]=d["lo"][i];hi[i]=d["hi"][i];dep[i]=d["dependencies"][i];for(int j=0;j<n;j++)A.setElem(i,j,d["A"][i][j]);}for(int k=0;k<n;k++)if(dep[k]<0){std::vector<int>ts;for(int j=0;j<n;j++)if(dep[j]==k)ts.push_back(j);if(ts.size()!=2)throw std::runtime_error("Two tangents required");const int t=ts[0],s=ts[1];contacts.push_back({k,t,s,.5*(A(t,t)+A(s,s)+std::hypot(A(t,t)-A(s,s),2*A(t,s))),static_cast<double>(hi[t])});}}
};
int evaluate(void*opaque,int m,int n,const double*x,double*F,int){
 auto&c=*static_cast<Context*>(opaque);if(c.calls>=c.limit)return -2;c.calls++;std::vector<double>w(n);for(int i=0;i<n;i++){w[i]=0;for(int j=0;j<n;j++)w[i]+=c.A(i,j)*x[j];w[i]-=c.b[i];}
 double residual=0;for(auto t:c.contacts){F[t.k]=(x[t.k]-std::max(0.,x[t.k]-w[t.k]/c.A(t.k,t.k)))*c.A(t.k,t.k);double zt=x[t.t]-w[t.t]/t.eig,zs=x[t.s]-w[t.s]/t.eig,cap=t.mu*std::max(0.,x[t.k]),length=std::hypot(zt,zs),factor=length>cap?cap/length:1.;F[t.t]=(x[t.t]-factor*zt)*t.eig;F[t.s]=(x[t.s]-factor*zs)*t.eig;residual=std::max(residual,std::max(std::abs(F[t.k]),std::hypot(F[t.t],F[t.s])));}
 for(int i=0;i<n;i++)if(!std::isfinite(x[i])||!std::isfinite(F[i]))return -3;
 if(residual<c.best_score){c.best_score=residual;c.best.assign(x,x+n);}
 if(residual<=c.tolerance){btVectorXu q(n);for(int i=0;i<n;i++)q[i]=x[i];CoulombStats stats;try{if(coulombIterate(c.A,c.b,q,c.lo,c.hi,c.dep,0,c.tolerance,stats)){c.solution.resize(n);for(int i=0;i<n;i++)c.solution[i]=q[i];return -1;}}catch(const std::exception&){}}
 return 0;
}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("Expected original component JSON");std::ifstream input(argv[1]);json d;input>>d;Context c(d);int n=c.b.rows();if(n<=0||n>64)throw std::runtime_error("Component cap64");std::vector<double>x=d["p"].get<std::vector<double>>(),F(n),diag(n,0),J(n*n);int nfev=0,status=0;double radius=0,alpha=0;bool initialized=false;
 auto jacobian=[&](){for(int j=0;j<n;j++){auto q=x;double h=std::sqrt(std::numeric_limits<double>::epsilon())*(x[j]<0?-1.:1.)*std::max(1.,std::abs(x[j]));q[j]+=h;h=q[j]-x[j];std::vector<double>next(n);int result=evaluate(&c,n,n,q.data(),next.data(),2);if(result<0)return result;for(int i=0;i<n;i++)J[i*n+j]=(next[i]-F[i])/h;}for(int j=0;j<n;j++){double norm=0;for(int i=0;i<n;i++)norm=std::hypot(norm,J[i*n+j]);diag[j]=std::max(diag[j],norm);if(diag[j]==0)diag[j]=1;}return 0;};
 status=evaluate(&c,n,n,x.data(),F.data(),1);if(status==0)status=jacobian();
 while(status==0&&c.calls<c.limit){
  if(!initialized){for(int j=0;j<n;j++)radius=std::hypot(radius,diag[j]*x[j]);if(radius==0)radius=1;initialized=true;}
  auto scaled=J;for(int i=0;i<n;i++)for(int j=0;j<n;j++)scaled[i*n+j]/=diag[j];auto rhs=F;for(double&v:rhs)v=-v;
  auto linear=restart_spectral::spectralMoreStep(scaled,rhs,n,radius,alpha);if(!linear.converged){status=-4;break;}
  auto step=linear.step;for(int j=0;j<n;j++)step[j]/=diag[j];auto trial=x;for(int j=0;j<n;j++)trial[j]+=step[j];std::vector<double>next(n);status=evaluate(&c,n,n,trial.data(),next.data(),1);if(status<0)break;
  double old_cost=0,new_cost=0,predicted_cost=0;for(int i=0;i<n;i++){old_cost+=F[i]*F[i];new_cost+=next[i]*next[i];double value=F[i];for(int j=0;j<n;j++)value+=J[i*n+j]*step[j];predicted_cost+=value*value;}
  double actual=old_cost-new_cost,predicted=old_cost-predicted_cost,ratio=predicted>0?actual/predicted:0,old_radius=radius;
  alpha=linear.lambda;if(ratio<.25)radius=.25*linear.norm;else if(ratio>.75&&linear.norm>=.95*radius)radius*=2;if(!(radius>0&&std::isfinite(radius))){status=-5;break;}alpha*=old_radius/radius;
  if(actual>0){x=trial;F=next;status=jacobian();}
 }
 nfev=c.calls;
 const bool found=!c.solution.empty();std::cout<<json({{"accepted",found},{"raw_evaluations",c.calls},{"trust_status",status},{"reported_residual_calls",nfev},{"p",found?c.solution:c.best},{"best_original_projection_m_s",c.best_score}}).dump(2)<<'\n';return found?0:2;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 3;}}
