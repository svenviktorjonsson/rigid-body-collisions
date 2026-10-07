#include "qr_linear.h"
#include <stdexcept>
#include <iostream>
void require(bool value,const char* text){if(!value)throw std::runtime_error(text);}
int main(){
 auto identity=trial_qr::direction({1,0,0,1},{2,-3},2);
 require(identity.converged&&identity.rank==2&&std::abs(identity.step[0]-2)<1e-13&&std::abs(identity.step[1]+3)<1e-13,"Full-rank QR direction");
 auto redundant=trial_qr::direction({1,2,2,4},{3,6},2);
 require(redundant.converged&&redundant.rank==1&&redundant.model_residual_square<1e-25,"Consistent redundant equations");
 auto weak=trial_qr::direction({1,0,0,1e-8},{1,1e-8},2);
 require(weak.converged&&weak.rank==2&&std::abs(weak.step[1]-1)<1e-13,"Weak retained numerical direction");
 auto zero=trial_qr::direction({0,0,0,0},{1,1},2);
 require(!zero.converged&&zero.rank==0,"Zero matrix must fall back");
 auto inconsistent=trial_qr::direction({1,2,2,4},{1,-2},2);
 require(inconsistent.converged&&inconsistent.model_residual_square<5,"Inconsistent linearized face gives only a merit-decreasing search candidate");
 trial_qr::stats.calls=1024;auto bounded=trial_qr::direction({1},{1},1);
 require(!bounded.converged&&trial_qr::stats.budget_rejections==1,"Bounded QR calls");
 std::cout<<"QR numerical checks PASS; no contact-law acceptance inferred\n";
}
