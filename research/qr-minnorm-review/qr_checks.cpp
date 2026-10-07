#include "qr_linear.h"
#include <iostream>
#include <stdexcept>
void require(bool x,const char* label){if(!x)throw std::runtime_error(label);}
int main(){
 auto redundant=trial_qr::direction({1,2,2,4},{3,6},2);
 require(redundant.converged&&redundant.rank==1&&std::abs(redundant.step[0]-.6)<1e-13&&std::abs(redundant.step[1]-1.2)<1e-13,"Exact minimum-norm redundant direction");
 auto weak=trial_qr::direction({1,0,0,1e-8},{1,1e-8},2);
 require(weak.converged&&weak.rank==2&&std::abs(weak.step[1]-1)<1e-13,"Retained weak direction");
 auto flat=trial_qr::direction({1,0,0,0},{1e-5,1},2);
 require(!flat.converged&&trial_qr::stats.model_rejections>0,"Tiny predicted reduction must use SVD fallback");
 auto scaled=trial_qr::direction({1e200,2e200,2e200,4e200},{3e200,6e200},2);
 require(!scaled.converged,"Overflowing original model norm must fall back");
 auto small=trial_qr::direction({1e-100,2e-100,2e-100,4e-100},{3e-100,6e-100},2);
 require(small.converged&&std::abs(small.step[0]-.6)<1e-13,"Uniformly small numerical scaling");
 trial_qr::stats.calls=1024;require(!trial_qr::direction({1},{1},1).converged,"Bounded QR calls");
 std::cout<<"Minimum-norm QR numerical checks PASS\n";
}
