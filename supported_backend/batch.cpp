#include "kernel.h"
#include "batch.h"
#include <cstddef>
#include <limits>
#include <omp.h>

// Field-major Float64 storage, single owner index k. Calls allocate no memory.
// Failures return the earliest failed body; callers must discard that batch.
extern "C" std::size_t supported_batch(const double* input, double* output,
                                      std::size_t count, int threads) {
    std::size_t failed=std::numeric_limits<std::size_t>::max();
    if(threads<1||threads>omp_get_num_procs()||count>std::numeric_limits<std::size_t>::max()/(15*sizeof(double))||
       (count&&(!input||!output)))return SUPPORTED_INVALID_ARGUMENT;
    #pragma omp parallel for num_threads(threads) schedule(static) reduction(min:failed) if(threads>1)
    for(std::size_t k=0;k<count;++k) {
        const supported_fast::Input state{
            input[k],input[count+k],input[2*count+k],input[3*count+k],
            input[4*count+k],input[5*count+k],input[6*count+k],input[7*count+k],
            input[8*count+k],input[9*count+k],input[10*count+k],input[11*count+k],
            input[12*count+k],input[13*count+k],input[14*count+k]};
        try {
            const auto result=supported_fast::advance(state);
            for(int field=0;field<11;++field)output[field*count+k]=result.fields[field];
        } catch(...) {
            failed=std::min(failed,k);
        }
    }
    return failed;
}

extern "C" int supported_max_threads() {return omp_get_num_procs();}
extern "C" uint32_t supported_abi_version() {return 1;}
extern "C" uint32_t supported_input_fields() {return 15;}
extern "C" uint32_t supported_output_fields() {return 11;}
extern "C" std::size_t supported_validate(const double* input,std::size_t count) {
    if(count>std::numeric_limits<std::size_t>::max()/(15*sizeof(double))||(count&&!input))return SUPPORTED_INVALID_ARGUMENT;
    for(std::size_t k=0;k<count;++k) {
        for(int field=0;field<15;++field)if(!std::isfinite(input[field*count+k]))return k;
        for(int field:{0,1,2,8})if(input[field*count+k]<=0)return k;
        for(int field:{3,9,10,11,12,13,14})if(input[field*count+k]<0)return k;
        if(input[10*count+k]>input[9*count+k]||
           (input[11*count+k]>0&&input[12*count+k]==0)||
           (input[13*count+k]>0&&input[14*count+k]==0))return k;
    }
    return std::numeric_limits<std::size_t>::max();
}
