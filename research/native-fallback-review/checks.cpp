#include "../active-direct/final/coulomb_restart.h"
#include <stdexcept>
#include <iostream>
void expect(bool ok,const char*why){if(!ok)throw std::runtime_error(why);}
struct Case{btMatrixXu A;btVectorXu b,x,hi;btAlignedObjectArray<int>dep;Case():A(3,3),b(3),x(3),hi(3){dep.resize(3);for(int i=0;i<3;i++){b[i]=0;x[i]=.125;hi[i]=i==0?1e30:.5;dep[i]=i==0?-1:0;for(int j=0;j<3;j++)A.setElem(i,j,i==j?1:0);}b[0]=1;b[1]=.2;}};
void rejected(Case&c,double tol=1e-8,int budget=1024){std::vector<double>before(c.x.rows());for(int i=0;i<c.x.rows();i++)before[i]=c.x[i];circular_restart::Stats stats;expect(!circular_restart::solve(c.A,c.b,c.x,c.hi,c.dep,tol,stats,budget),"invalid/unsupported call accepted");for(int i=0;i<c.x.rows();i++)expect(c.x[i]==before[i],"failure mutated caller");}
int main(){
 {Case c;circular_restart::Stats stats;expect(circular_restart::solve(c.A,c.b,c.x,c.hi,c.dep,1e-8,stats),"simple stick failed");expect(std::abs(c.x[0]-1)<1e-8&&std::abs(c.x[1]-.2)<1e-8&&std::abs(c.x[2])<1e-8,"wrong simple stick");expect(stats.svd_calls<=1024,"SVD cap");}
 {Case c;c.b[1]=2;circular_restart::Stats stats;expect(circular_restart::solve(c.A,c.b,c.x,c.hi,c.dep,1e-8,stats),"simple slide failed");expect(std::abs(c.x[0]-1)<1e-8&&std::abs(c.x[1]-.5)<1e-8&&std::abs(c.x[2])<1e-8,"wrong isotropic slide");}
 {Case c;c.A.resize(2,2);rejected(c);}
 {Case c;c.x.resize(2);rejected(c);}
 {Case c;c.dep.resize(2);rejected(c);}
 {Case c;c.dep[2]=4;rejected(c);}
 {Case c;c.dep[2]=1;rejected(c);}
 {Case c;c.dep[0]=-2;rejected(c);}
 {Case c;c.hi[2]=.6;rejected(c);}
 {Case c;c.hi[0]=-1;rejected(c);}
 {Case c;c.A.setElem(1,1,std::numeric_limits<double>::quiet_NaN());rejected(c);}
 {Case c;c.b[1]=std::numeric_limits<double>::infinity();rejected(c);}
 {Case c;rejected(c,0);rejected(c,std::numeric_limits<double>::quiet_NaN());}
 {Case c;rejected(c,1e-8,255);}
 {Case c;c.hi[0]=0;rejected(c);}
 {Case c;std::vector<double> before(3);for(int i=0;i<3;i++)before[i]=c.x[i];restart_more::Stats more;expect(!restart_more::solve(c.A,c.b,c.x,c.hi,c.dep,1e-8,more,0,0),"More zero budget");restart_guide::Stats guide;expect(!restart_guide::solve(c.A,c.b,c.x,c.hi,c.dep,1e-8,guide,0,0),"guide zero budget");for(int i=0;i<3;i++)expect(c.x[i]==before[i],"helper budget mutated output");}
 std::cout<<"16 independent native fallback checks passed\n";
}
