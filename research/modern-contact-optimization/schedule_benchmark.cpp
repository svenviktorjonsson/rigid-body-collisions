#include "../../contact_backend/schedule.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

using contact_schedule::Index;
using Clock=std::chrono::steady_clock;
struct Graph{std::vector<Index>a,b;std::vector<std::uint8_t>movable;};
Graph graph(const std::string& kind,std::size_t count){
    Graph result;result.a.resize(count);result.b.resize(count);
    if(kind=="independent_floor"){
        result.movable.assign(count+1,1);result.movable.back()=0;
        for(std::size_t k=0;k<count;++k){result.a[k]=static_cast<Index>(k);result.b[k]=static_cast<Index>(count);}
    }else if(kind=="chain"){
        result.movable.assign(count+1,1);
        for(std::size_t k=0;k<count;++k){result.a[k]=static_cast<Index>(k);result.b[k]=static_cast<Index>(k+1);}
    }else if(kind=="hub"){
        result.movable.assign(count+1,1);
        for(std::size_t k=0;k<count;++k){result.a[k]=0;result.b[k]=static_cast<Index>(k+1);}
    }else{
        // Different degrees and repeated edges in eight-body components. This
        // is graph topology, not generated irregular-shape collision geometry.
        result.movable.assign(((count+23)/24)*8,1);
        for(std::size_t k=0;k<count;++k){auto base=(k/24)*8,t=k%24;
            auto u=t%8,v=(u+1+(t/8))%8;if(v==u)v=(v+1)%8;
            result.a[k]=static_cast<Index>(base+u);result.b[k]=static_cast<Index>(base+v);}
    }
    return result;
}
double elapsed(Clock::time_point start){return std::chrono::duration<double,std::milli>(Clock::now()-start).count();}
void array(const std::vector<double>& values){std::cout<<'[';for(std::size_t i=0;i<values.size();++i){if(i)std::cout<<',';std::cout<<values[i];}std::cout<<']';}
double median(std::vector<double> values){std::sort(values.begin(),values.end());return values[values.size()/2];}
int main(){try{
    std::cout<<std::setprecision(15)<<"{\"scope\":\"single-worker topology planning/cache checking only; no collision discovery or physical solve\",\"cases\":[";
    bool first=true;std::size_t checksum=0;
    for(const std::string kind:{"independent_floor","chain","hub","heterogeneous_degree"})
        for(std::size_t count:{100u,1000u,10000u,100000u,1000000u}){
            auto input=graph(kind,count);contact_schedule::Cache cache;
            auto start=Clock::now();const auto& plan=cache.prepare(input.a,input.b,input.movable);const double preparation=elapsed(start);
            std::vector<double> builds,hits;
            // Warmup and then alternate fresh-build and exact-cache-hit timing.
            auto warm=contact_schedule::build(input.a,input.b,input.movable);checksum+=warm.contact_count;
            cache.prepare(input.a,input.b,input.movable);
            auto build=[&](){auto begin=Clock::now();auto result=contact_schedule::build(input.a,input.b,input.movable);const auto ms=elapsed(begin);checksum+=result.island_order.back();return ms;};
            auto hit=[&](){auto begin=Clock::now();const auto& result=cache.prepare(input.a,input.b,input.movable);const auto ms=elapsed(begin);checksum+=result.color_order.back();return ms;};
            for(int repeat=0;repeat<7;++repeat){if(repeat%2==0){builds.push_back(build());hits.push_back(hit());}else{hits.push_back(hit());builds.push_back(build());}}
            if(!first)std::cout<<',';
            first=false;
            std::cout<<"{\"kind\":\""<<kind<<"\",\"contacts\":"<<count<<",\"bodies\":"<<input.movable.size()
                     <<",\"islands\":"<<plan.island_offsets.size()-1<<",\"colors\":"<<plan.color_offsets.size()-1
                     <<",\"serial_contacts\":"<<plan.serial_order.size()<<",\"cache_preparation_ms\":"<<preparation
                     <<",\"build_median_ms\":"<<median(builds)<<",\"cache_hit_median_ms\":"<<median(hits)<<",\"build_samples_ms\":";
            array(builds);std::cout<<",\"cache_hit_samples_ms\":";array(hits);std::cout<<'}'<<std::flush;
        }
    std::cout<<"],\"checksum\":"<<checksum<<"}\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
