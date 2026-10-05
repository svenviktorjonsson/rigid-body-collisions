// Independent analytic fixtures for both complete-minimum-norm Gram branches.
#include "qr_linear.h"
#include <iostream>
#include <stdexcept>

void fixture(const std::vector<double>& A,const std::vector<double>& b,
             const std::vector<double>& expected,int rank) {
    auto x=trial_qr::direction(A,b,static_cast<int>(b.size()));
    if(!x.converged || x.rank!=rank) throw std::runtime_error("Unexpected QR rejection or rank");
    for(size_t k=0;k<b.size();k++)
        if(std::abs(x.step[k]-expected[k])>1e-12)
            throw std::runtime_error("Minimum-norm result differs from analytic solution");
}

int main() {
    try {
        // Two retained coordinates, one discarded: smaller free-coordinate Gram.
        fixture({1,0,1,0,1,1,0,0,0},{2,3,0},{1./3,4./3,5./3},2);
        // One retained coordinate, three discarded: smaller retained-coordinate Gram.
        const std::vector<double>u{1,2,0,-1},v{4,-2,1,3};
        std::vector<double>A(16);for(int i=0;i<4;i++)for(int j=0;j<4;j++)A[i*4+j]=u[i]*v[j];
        fixture(A,{3,6,0,-3},{.4,-.2,.1,.3},1);
        // Uniformly very small numerical scaling leaves the completed direction unchanged.
        for(double& value:A)value*=1e-100;
        fixture(A,{3e-100,6e-100,0,-3e-100},{.4,-.2,.1,.3},1);
        // Full rank includes no Gram correction.
        fixture({1,1,0,0,2,1,1,0,3},{3,7,10},{1,2,3},3);
        std::cout << "Both complete minimum-norm Gram branches PASS\n";
    } catch(const std::exception& error) {
        std::cerr<<error.what()<<"\n";return 1;
    }
}
