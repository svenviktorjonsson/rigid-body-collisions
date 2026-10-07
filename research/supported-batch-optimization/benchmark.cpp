#include "../../supported_backend/batch.h"
#include "../contact-gap-fix/supported_kernel.h"
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <stdexcept>

// Identical field-major data movement and failure checks for the frozen law.
void baseline(const double* in,double* out,std::size_t count) {
    for(std::size_t k=0;k<count;++k) {
        const supported::Input x{in[k],in[count+k],in[2*count+k],in[3*count+k],in[4*count+k],in[5*count+k],in[6*count+k],in[7*count+k],in[8*count+k],in[9*count+k],in[10*count+k],in[11*count+k],in[12*count+k],in[13*count+k],in[14*count+k]};
        auto r=supported::advance(x);
        for(int j=0;j<11;++j)out[j*count+k]=r.fields[j];
    }
}
double elapsed(const double* in,double* out,std::size_t count,int threads) {
    auto start=std::chrono::steady_clock::now();
    if(threads==0)baseline(in,out,count);
    else if(supported_batch(in,out,count,threads)!=std::numeric_limits<std::size_t>::max())throw std::runtime_error("batch failure");
    return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
int main(int argc,char** argv) {
    if(argc!=2)return 2;
    std::ifstream file(argv[1]);int n=0;file>>n;if(n!=400)return 3;
    std::vector<std::array<double,26>> fixtures(n);
    for(auto& row:fixtures)for(double& v:row)file>>v;
    if(!file)return 4;
    double max_scaled=0;
    std::vector<double> controls(15*n),check(11*n);
    for(int k=0;k<n;++k)for(int f=0;f<15;++f)controls[f*n+k]=fixtures[k][f];
    if(supported_validate(controls.data(),n)!=std::numeric_limits<std::size_t>::max())return 5;
    if(supported_batch(controls.data(),check.data(),n,1)!=std::numeric_limits<std::size_t>::max())return 6;
    for(int k=0;k<n;++k) {
        const auto& x=fixtures[k];const double speed=std::max({std::abs(x[5]),x[2]*std::abs(x[6]),x[2]*std::abs(x[7]),std::abs(x[4]/x[0])*x[8],1e-30});
        const std::array<double,11> scale{speed,speed/x[2],speed/x[2],speed*x[8],x[0]*speed,x[0]*x[2]*speed,x[0]*x[2]*speed,x[0]*speed*speed,x[0]*speed*speed,x[0]*speed*speed,x[0]*speed*speed};
        for(int f=0;f<11;++f)max_scaled=std::max(max_scaled,std::abs(check[f*n+k]-x[15+f])/scale[f]);
    }
    if(max_scaled>1e-10)return 7;
    std::cout<<std::setprecision(17)<<"{\"controls\":400,\"max_python_scaled_error\":"<<max_scaled<<",\"batches\":[";
    bool first=true;
    for(std::size_t count:{100,10000,100000,1000000}) {
        std::vector<double> in(15*count),ref(11*count),out(11*count),single(11*count);
        for(std::size_t k=0;k<count;++k)for(int f=0;f<15;++f)in[f*count+k]=fixtures[k%n][f];
        if(supported_validate(in.data(),count)!=std::numeric_limits<std::size_t>::max())return 8;
        baseline(in.data(),ref.data(),count);
        if(supported_batch(in.data(),single.data(),count,1)!=std::numeric_limits<std::size_t>::max())return 9;
        double max_difference=0;
        for(std::size_t k=0;k<count;++k)for(int f=0;f<11;++f)max_difference=std::max(max_difference,std::abs(ref[f*count+k]-single[f*count+k]));
        for(int threads:{1,4,8}) {
            elapsed(in.data(),out.data(),count,threads);std::array<double,7> old_times{},new_times{};
            for(int repeat=0;repeat<7;++repeat) {
                if(repeat%2==0) {old_times[repeat]=elapsed(in.data(),ref.data(),count,0);new_times[repeat]=elapsed(in.data(),out.data(),count,threads);}
                else {new_times[repeat]=elapsed(in.data(),out.data(),count,threads);old_times[repeat]=elapsed(in.data(),ref.data(),count,0);}
                if(out!=single)return 10; // Every response/field, not just checksum.
            }
            std::sort(old_times.begin(),old_times.end());std::sort(new_times.begin(),new_times.end());
            double checksum=std::accumulate(out.begin(),out.end(),0.);
            if(!first)std::cout<<",";
            first=false;
            std::cout<<"{\"responses\":"<<count<<",\"threads\":"<<threads<<",\"baseline_median_ms\":"<<old_times[3]<<",\"candidate_median_ms\":"<<new_times[3]<<",\"candidate_min_ms\":"<<new_times[0]<<",\"candidate_max_ms\":"<<new_times[6]<<",\"speedup\":"<<old_times[3]/new_times[3]<<",\"working_arrays_bytes\":"<<48*count*sizeof(double)<<",\"native_reference_max_abs_difference\":"<<max_difference<<",\"checksum\":"<<checksum<<",\"samples_ms\":[";
            for(int k=0;k<7;++k) {if(k)std::cout<<",";std::cout<<new_times[k];}
            std::cout<<"]}";
        }
    }
    std::cout<<"]}\n";
}
