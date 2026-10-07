#include "supported_kernel.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    std::ifstream file(argv[1]);int count=0;file>>count;
    if(count!=400)return 3;
    std::vector<supported::Input> inputs(count);std::vector<std::array<double,11>> expected(count);double max_error=0;
    for(int k=0;k<count;++k) {
        auto& a=inputs[k];file>>a.m>>a.I>>a.R>>a.N>>a.drive>>a.v>>a.w>>a.spin>>a.h>>a.mu_s>>a.mu_d>>a.mu_r>>a.a_r>>a.mu_n>>a.a_n;
        for(double& x:expected[k])file>>x;
        if(!file)return 4;
        const auto result=supported::advance(a);
        const double vscale=std::max({std::abs(a.v),a.R*std::abs(a.w),a.R*std::abs(a.spin),std::abs(a.drive/a.m)*a.h,1e-30});
        const std::array<double,11> scales={vscale,vscale/a.R,vscale/a.R,vscale*a.h,a.m*vscale,a.m*a.R*vscale,a.m*a.R*vscale,a.m*vscale*vscale,a.m*vscale*vscale,a.m*vscale*vscale,a.m*vscale*vscale};
        for(int j=0;j<11;++j)max_error=std::max(max_error,std::abs(result.fields[j]-expected[k][j])/scales[j]);
    }
    if(max_error>1e-10)return 5;
    std::cout<<std::setprecision(17)<<"{\"native_reference_controls\":"<<count<<",\"max_scaled_error\":"<<max_error<<",\"batches\":[";
    bool first=true;
    for(std::size_t size:{100,10000,100000,1000000}) {
        // Flat SoA body index k; allocation is outside timing. Every update loads
        // fifteen inputs and writes all eleven physical response channels.
        std::array<std::vector<double>,15> in;std::array<std::vector<double>,11> out;
        for(auto& f:in)f.resize(size);
        for(auto& f:out)f.resize(size);
        for(std::size_t k=0;k<size;++k) {
            const auto& a=inputs[k%inputs.size()];
            const std::array<double,15> fields={a.m,a.I,a.R,a.N,a.drive,a.v,a.w,a.spin,a.h,a.mu_s,a.mu_d,a.mu_r,a.a_r,a.mu_n,a.a_n};
            for(int j=0;j<15;++j)in[j][k]=fields[j];
        }
        std::array<double,5> elapsed{};
        for(int repeat=-1;repeat<5;++repeat) {
            const auto start=std::chrono::steady_clock::now();
            for(std::size_t k=0;k<size;++k) {
                const supported::Input a{in[0][k],in[1][k],in[2][k],in[3][k],in[4][k],in[5][k],in[6][k],in[7][k],in[8][k],in[9][k],in[10][k],in[11][k],in[12][k],in[13][k],in[14][k]};
                const auto response=supported::advance(a);
                for(int j=0;j<11;++j)out[j][k]=response.fields[j];
            }
            const double ns=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
            if(repeat>=0)elapsed[repeat]=ns;
        }
        std::sort(elapsed.begin(),elapsed.end());double checksum=0;
        for(const auto& field:out)checksum+=std::accumulate(field.begin(),field.end(),0.);
        if(!first)std::cout<<",";
        first=false;
        std::cout<<"{\"responses\":"<<size<<",\"median_ns_per_response\":"<<elapsed[2]/size<<",\"min_ns_per_response\":"<<elapsed[0]/size<<",\"max_ns_per_response\":"<<elapsed[4]/size<<",\"array_bytes\":"<<26*size*sizeof(double)<<",\"checksum\":"<<checksum<<"}";
    }
    std::cout<<"]}\n";
}
