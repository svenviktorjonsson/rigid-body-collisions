#pragma once
#include "normal_qp.h"
#include "mixed_face_simplex.h"
namespace mixed_face_traction {
struct Stats {unsigned long long components=0,mode_attempts=0,lp_calls=0,pivots=0,direction_steps=0;int largest_rows=0,search_rows_max=0,weak_contacts_max=0;double accepted_residual=0;};
#ifdef SPATIAL_LAPACK_RECOVERY
template<class OriginalGate,class OriginalResidual>
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&p,const btVectorXu&lo,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tol,Stats&stats,OriginalGate original_gate,OriginalResidual original_residual){
 const int n=b.rows();if(n<=0||n>4096)return false;btVectorXu candidate=p;std::vector<bool>visited(n,false);
 for(int first=0;first<n;first++)if(!visited[first]){
  std::vector<int>ids{first};visited[first]=true;for(size_t at=0;at<ids.size();at++)for(int j=0;j<n;j++)if(!visited[j]&&(A(ids[at],j)!=0||A(j,ids[at])!=0||dep[j]==ids[at]||dep[ids[at]]==j)){visited[j]=true;ids.push_back(j);}
  int m=ids.size();stats.components++;stats.largest_rows=std::max(stats.largest_rows,m);std::vector<int>inverse(n,-1);for(int i=0;i<m;i++)inverse[ids[i]]=i;
  btMatrixXu M(m,m);btVectorXu rhs(m),q(m),lower(m),upper(m);btAlignedObjectArray<int>d;d.resize(m);
  for(int i=0;i<m;i++){rhs[i]=b[ids[i]];q[i]=candidate[ids[i]];lower[i]=lo[ids[i]];upper[i]=hi[ids[i]];d[i]=dep[ids[i]]<0?-1:inverse[dep[ids[i]]];for(int j=0;j<m;j++)M.setElem(i,j,A(ids[i],ids[j]));}
  btVectorXu checked=q;if(original_gate(M,rhs,checked,lower,upper,d,tol)){for(int i=0;i<m;i++)candidate[ids[i]]=checked[i];continue;}
  std::vector<double>w(m);for(int i=0;i<m;i++){w[i]=-rhs[i];for(int j=0;j<m;j++)w[i]+=M(i,j)*q[j];}
  struct Contact {int normal,t,s;double mu;bool active,weak;};std::vector<Contact>contacts;std::vector<int>weak,strong;
  std::vector<bool>selected(m,false);
  for(int k=0;k<m;k++)if(d[k]<0){std::vector<int>ts;for(int j=0;j<m;j++)if(d[j]==k)ts.push_back(j);if(ts.size()!=2)return false;bool active=q[k]>tol/M(k,k)||w[k]<-tol;bool isweak=active&&std::hypot(w[ts[0]],w[ts[1]])<=1e-5;int index=contacts.size();contacts.push_back({k,ts[0],ts[1],upper[ts[0]],active,isweak});if(active){selected[k]=selected[ts[0]]=selected[ts[1]]=true;(isweak?weak:strong).push_back(index);}}
  std::vector<int>rows;for(int i=0;i<m;i++)if(selected[i])rows.push_back(i);int r=rows.size();stats.search_rows_max=std::max(stats.search_rows_max,r);stats.weak_contacts_max=std::max(stats.weak_contacts_max,static_cast<int>(weak.size()));if(r<=0||r>192||weak.size()>12)return false;
  // Normal impulses have one nonnegative variable; free tangents use +/- parts.
  std::vector<int>positive(m,-1),negative(m,-1);int vars=0;
  for(int row:rows){positive[row]=vars++;if(d[row]>=0)negative[row]=vars++;}if(vars>384)return false;
  int faces=1;for(size_t i=0;i<weak.size();i++)faces*=3;bool found=false;
  for(int face=0;face<faces&&!found&&stats.lp_calls<4096&&stats.pivots<100000;face++){
   stats.mode_attempts++;std::vector<int>mode(contacts.size(),-1);for(int i:strong)mode[i]=2;int number=face;
   for(int i=static_cast<int>(weak.size())-1;i>=0;i--){int digit=number%3;number/=3;mode[weak[i]]=digit==0?1:(digit==1?0:2);}
   std::vector<std::pair<double,double>>direction(contacts.size());for(size_t i=0;i<contacts.size();i++)if(mode[i]==2){auto&c=contacts[i];double length=std::hypot(w[c.t],w[c.s]);direction[i]=length>0?std::make_pair(w[c.t]/length,w[c.s]/length):std::make_pair(1.,0.);}
   for(int iteration=0;iteration<128&&stats.lp_calls<4096&&stats.pivots<100000;iteration++){
    stats.direction_steps++;std::vector<std::vector<double>>constraints;std::vector<double>limits,cost(vars,0);double allowance=.5*tol;
    auto rowVector=[&](int row,double sign){std::vector<double>v(vars,0);for(int j:rows){double value=sign*M(row,j);v[positive[j]]=value;if(negative[j]>=0)v[negative[j]]=-value;}return v;};
    auto add=[&](std::vector<double>v,double bound){constraints.push_back(std::move(v));limits.push_back(bound);};
    auto equality=[&](std::vector<double>v,double target,double slack){add(v,target+slack);for(double&value:v)value=-value;add(std::move(v),-target+slack);};
    for(size_t i=0;i<contacts.size();i++){auto&c=contacts[i];if(mode[i]>=1)equality(rowVector(c.normal,1),rhs[c.normal],allowance);else add(rowVector(c.normal,-1),-rhs[c.normal]+allowance);
     if(!c.active)continue;cost[positive[c.normal]]=-1;
     if(mode[i]==0){for(int j:{c.normal,c.t,c.s}){std::vector<double>v(vars,0);v[positive[j]]=1;add(v,0);if(negative[j]>=0){v[positive[j]]=0;v[negative[j]]=1;add(v,0);}}}
     else if(mode[i]==1){equality(rowVector(c.t,1),rhs[c.t],allowance);equality(rowVector(c.s,1),rhs[c.s],allowance);for(int panel=0;panel<32;panel++){double angle=2*3.14159265358979323846*panel/32;std::vector<double>v(vars,0);v[positive[c.t]]=std::cos(angle);v[negative[c.t]]=-v[positive[c.t]];v[positive[c.s]]=std::sin(angle);v[negative[c.s]]=-v[positive[c.s]];v[positive[c.normal]]=-c.mu*std::cos(3.14159265358979323846/32);add(std::move(v),0);}}
     else {for(int axis=0;axis<2;axis++){int j=axis==0?c.t:c.s;double value=axis==0?direction[i].first:direction[i].second;std::vector<double>v(vars,0);v[positive[j]]=1;v[negative[j]]=-1;v[positive[c.normal]]=c.mu*value;equality(std::move(v),0,0);}}
    }
    if(constraints.size()>4096)return false;stats.lp_calls++;mixed_face_simplex::Solver lp(constraints,limits,cost,stats.pivots,100000);std::vector<double>solution;if(!lp.solve(solution))break;
    btVectorXu trial(m);trial.setZero();for(int j:rows)trial[j]=solution[positive[j]]-(negative[j]>=0?solution[negative[j]]:0);
    for(const auto&c:contacts){trial[c.normal]=std::max(0.,static_cast<double>(trial[c.normal]));double length=std::hypot(trial[c.t],trial[c.s]),cap=c.mu*trial[c.normal];if(length>cap){trial[c.t]*=cap/length;trial[c.s]*=cap/length;}}
    if(original_gate(M,rhs,trial,lower,upper,d,tol)){q=trial;found=true;stats.accepted_residual=std::max(stats.accepted_residual,original_residual(M,rhs,q,upper,d));break;}
    std::vector<double>response(m);for(int i=0;i<m;i++){response[i]=-rhs[i];for(int j=0;j<m;j++)response[i]+=M(i,j)*trial[j];}
    bool invalid=false;for(size_t i=0;i<contacts.size();i++)if(mode[i]==2){auto&c=contacts[i];double length=std::hypot(response[c.t],response[c.s]);if(length<=1e-12){invalid=true;break;}double x=.5*direction[i].first+.5*response[c.t]/length,y=.5*direction[i].second+.5*response[c.s]/length;double norm=std::hypot(x,y);if(norm<=1e-12){invalid=true;break;}direction[i]={x/norm,y/norm};}
    if(invalid)break;
   }
  }
  if(!found)return false;for(int i=0;i<m;i++)candidate[ids[i]]=q[i];
 }
 if(!original_gate(A,b,candidate,lo,hi,dep,tol))return false;p=candidate;return true;
}
#endif
}
