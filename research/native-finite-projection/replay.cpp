// Isolated CMINPACK numerical search; original physical system/gate unchanged.
#include "coulomb.h"
#include <cminpack-1/cminpack.h>
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
 auto&c=*static_cast<Context*>(opaque);if(c.calls>=c.limit)return -2;c.calls++;std::vector<double>w(n);for(int i=0;i<n;i++){w[i]=-c.b[i];for(int j=0;j<n;j++)w[i]+=c.A(i,j)*x[j];}
 double residual=0;for(auto t:c.contacts){F[t.k]=(x[t.k]-std::max(0.,x[t.k]-w[t.k]/c.A(t.k,t.k)))*c.A(t.k,t.k);double zt=x[t.t]-w[t.t]/t.eig,zs=x[t.s]-w[t.s]/t.eig,cap=t.mu*std::max(0.,x[t.k]),length=std::hypot(zt,zs),factor=length>cap?cap/length:1.;F[t.t]=(x[t.t]-factor*zt)*t.eig;F[t.s]=(x[t.s]-factor*zs)*t.eig;residual=std::max(residual,std::max(std::abs(F[t.k]),std::hypot(F[t.t],F[t.s])));}
 if(residual<c.best_score){c.best_score=residual;c.best.assign(x,x+n);}
 if(residual<=c.tolerance){btVectorXu q(n);for(int i=0;i<n;i++)q[i]=x[i];CoulombStats stats;try{if(coulombIterate(c.A,c.b,q,c.lo,c.hi,c.dep,0,c.tolerance,stats)){c.solution.resize(n);for(int i=0;i<n;i++)c.solution[i]=q[i];return -1;}}catch(const std::exception&){}}
 return 0;
}
int main(int argc,char**argv){try{
 if(argc!=2)throw std::runtime_error("Expected original component JSON");std::ifstream input(argv[1]);json d;input>>d;Context c(d);int n=c.b.rows();if(n<=0||n>64)throw std::runtime_error("Component cap64");std::vector<double>x=d["p"].get<std::vector<double>>(),F(n),diag(n),J(n*n),qtf(n),wa1(n),wa2(n),wa3(n),wa4(n);std::vector<int>piv(n);int nfev=0;int status=lmdif(evaluate,&c,n,n,x.data(),F.data(),1e-14,1e-14,1e-14,2048,0,diag.data(),1,100,0,&nfev,J.data(),n,piv.data(),qtf.data(),wa1.data(),wa2.data(),wa3.data(),wa4.data());const bool found=!c.solution.empty();std::cout<<json({{"accepted",found},{"raw_evaluations",c.calls},{"minpack_status",status},{"minpack_nfev",nfev},{"p",found?c.solution:c.best},{"best_original_projection_m_s",c.best_score}}).dump(2)<<'\n';return found?0:2;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 3;}}
