// Independent research solver harness: no world or production build.
#include <btBulletDynamicsCommon.h>
#include <BulletDynamics/MLCPSolvers/btDantzigSolver.h>
#include <nlohmann/json.hpp>
#include "projection_more_v2.h"
#include <fstream>
#include <iostream>
using J=nlohmann::json;
int main(int argc,char**argv){try{
 if(argc!=4)throw std::runtime_error("Usage: replay capture component_mode reserved");
 J d;std::ifstream file(argv[1]);file>>d;int n=d.at("b").size();btMatrixXu A(n,n);btVectorXu b(n),p(n),hi(n);btAlignedObjectArray<int>dep;dep.resize(n);
 for(int i=0;i<n;i++){b[i]=d["b"][i];p[i]=d["p"][i];hi[i]=d["hi"][i];dep[i]=d["dependencies"][i];for(int j=0;j<n;j++)A.setElem(i,j,d["A"][i][j]);}

 auto before=p;auto candidate=p;projection_recovery_v2::Stats stats;bool ok=false;std::vector<int>component_rows;int passes=0;
 if(std::stoi(argv[2])==0){ok=projection_recovery_v2::solve(A,b,candidate,hi,dep,d.at("tolerance_m_s"),stats,2048,2048,true,true);if(ok)p=candidate;}
 else{
  std::vector<int>parent(n);for(int i=0;i<n;i++)parent[i]=i;auto find=[&](int i){while(parent[i]!=i){parent[i]=parent[parent[i]];i=parent[i];}return i;};auto unite=[&](int i,int j){int a=find(i),b=find(j);if(a!=b)parent[b]=a;};
  for(int i=0;i<n;i++){if(dep[i]>=0)unite(i,dep[i]);for(int j=0;j<i;j++)if(A(i,j)!=0||A(j,i)!=0)unite(i,j);}
  std::vector<std::vector<int>>groups;for(int i=0;i<n;i++){int root=find(i),group=-1;for(int g=0;g<static_cast<int>(groups.size());g++)if(find(groups[g][0])==root)group=g;if(group<0){groups.push_back({});group=groups.size()-1;}groups[group].push_back(i);}
  bool success=true;for(auto indices:groups){int m=indices.size();component_rows.push_back(m);btMatrixXu C(m,m);btVectorXu cb(m),cx(m),ch(m);btAlignedObjectArray<int>cd;cd.resize(m);for(int i=0;i<m;i++){int row=indices[i];cb[i]=b[row];cx[i]=candidate[row];ch[i]=hi[row];cd[i]=dep[row]<0?-1:std::find(indices.begin(),indices.end(),dep[row])-indices.begin();for(int j=0;j<m;j++)C.setElem(i,j,A(row,indices[j]));}
   projection_recovery_v2::Stats part;bool accepted=projection_recovery_v2::solve(C,cb,cx,ch,cd,d.at("tolerance_m_s"),part,1024-stats.iteration_steps,1024-stats.svd_calls,true,true);stats.iteration_steps+=part.iteration_steps;stats.svd_calls+=part.svd_calls;stats.newton_steps+=part.newton_steps;stats.residual=std::max(stats.residual,part.residual);passes++;
   for(int i=0;i<m;i++)candidate[indices[i]]=accepted?cx[i]:part.failed_candidate.size()==static_cast<size_t>(m)?part.failed_candidate[i]:cx[i];
   if(!accepted){success=false;break;}
  }
  if(success){
   double error=0,energy=0,scale=1;std::vector<double>w(n);bool finite=true;
   for(int i=0;i<n;i++){w[i]=-b[i];for(int j=0;j<n;j++)w[i]+=A(i,j)*candidate[j];energy+=.5*candidate[i]*(w[i]-b[i]);scale+=std::abs(candidate[i]*b[i]);finite&=std::isfinite(candidate[i])&&std::isfinite(w[i]);}
   for(int k=0;k<n;k++)if(dep[k]<0){std::vector<int>t;for(int j=0;j<n;j++)if(dep[j]==k)t.push_back(j);if(t.size()!=2){finite=false;break;}double eig=.5*(A(t[0],t[0])+A(t[1],t[1])+std::hypot(A(t[0],t[0])-A(t[1],t[1]),2*A(t[0],t[1])));double z0=candidate[t[0]]-w[t[0]]/eig,z1=candidate[t[1]]-w[t[1]]/eig,length=std::hypot(z0,z1),cap=hi[t[0]]*std::max(0.,static_cast<double>(candidate[k])),factor=length>cap&&length>0?cap/length:1;error=std::max({error,std::abs(candidate[k]-std::max(0.,static_cast<double>(candidate[k])-w[k]/A(k,k)))*A(k,k),std::hypot(candidate[t[0]]-factor*z0,candidate[t[1]]-factor*z1)*eig});finite&=candidate[k]>=0&&candidate[k]<=hi[k];}
   success=finite&&std::isfinite(energy)&&std::isfinite(scale)&&error<=d.at("tolerance_m_s").get<double>()&&energy<=d.at("tolerance_m_s").get<double>()*scale;
  }
  if(success){p=candidate;ok=true;}else {stats.failed_candidate.resize(n);for(int i=0;i<n;i++)stats.failed_candidate[i]=candidate[i];}
 }
 bool unchanged=true;std::vector<double>last(n),returned(n);for(int i=0;i<n;i++){unchanged&=p[i]==before[i];returned[i]=p[i];last[i]=stats.failed_candidate.size()==static_cast<size_t>(n)?stats.failed_candidate[i]:p[i];}
 J out={{"accepted",ok},{"component_rows",component_rows},{"component_passes",passes},{"returned_impulse",returned},{"candidate_impulse",last},{"decline_preserves_input",ok||unchanged},{"svd_calls",stats.svd_calls},{"iteration_steps",stats.iteration_steps},{"accepted_steps",stats.newton_steps},{"reported_original_residual_m_s",stats.residual}};std::cout<<out.dump()<<'\n';return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 3;}}
