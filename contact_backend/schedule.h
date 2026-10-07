// Numerical scheduling only: no material, restitution or contact-law decisions.
#pragma once
#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace contact_schedule {
using Index=std::uint32_t;
struct Plan {
    // Each original contact appears once in island_order and once in the union
    // of color_order/serial_order. A contact is the entire force/couple block.
    std::vector<Index> island_order,color_order,serial_order;
    std::vector<std::size_t> island_offsets,color_offsets;
    std::size_t body_count=0,contact_count=0;
};

inline unsigned first_set(std::uint64_t mask) noexcept {
#if defined(__GNUC__) || defined(__clang__)
    return static_cast<unsigned>(__builtin_ctzll(mask)); // caller ensures nonzero
#else
    unsigned result=0;while((mask&1)==0){mask>>=1;++result;}return result;
#endif
}

inline Plan build(const std::vector<Index>& a,const std::vector<Index>& b,
                  const std::vector<std::uint8_t>& movable,unsigned colors=32) {
    const auto bodies=movable.size(),contacts=a.size();
    if(b.size()!=contacts||bodies>std::numeric_limits<Index>::max()||
       contacts>std::numeric_limits<Index>::max()||colors<1||colors>64)
        throw std::invalid_argument("Invalid indexed contact graph or color budget");
    for(auto flag:movable)if(flag>1)throw std::invalid_argument("Mutable body mask must contain 0/1");
    for(std::size_t k=0;k<contacts;++k)
        if(a[k]>=bodies||b[k]>=bodies||a[k]==b[k]||(!movable[a[k]]&&!movable[b[k]]))
            throw std::invalid_argument("Contact needs distinct valid endpoints and a mutable body");
    std::vector<Index> parent(bodies),size(bodies,1);
    for(std::size_t k=0;k<bodies;++k)parent[k]=static_cast<Index>(k);
    auto find=[&](Index k){while(parent[k]!=k){parent[k]=parent[parent[k]];k=parent[k];}return k;};
    for(std::size_t k=0;k<contacts;++k)if(movable[a[k]]&&movable[b[k]]) {
        auto u=find(a[k]),v=find(b[k]);
        if(u!=v){if(size[u]<size[v])std::swap(u,v);parent[v]=u;size[u]+=size[v];}
    }
    Plan plan;plan.body_count=bodies;plan.contact_count=contacts;
    const auto absent=std::numeric_limits<Index>::max();
    std::vector<Index> root_island(bodies,absent),island(contacts);
    std::vector<std::size_t> island_counts;
    for(std::size_t k=0;k<contacts;++k) {
        const auto root=find(movable[a[k]]?a[k]:b[k]);
        if(root_island[root]==absent){root_island[root]=static_cast<Index>(island_counts.size());island_counts.push_back(0);}
        island[k]=root_island[root];++island_counts[island[k]];
    }
    plan.island_offsets.push_back(0);
    for(auto count:island_counts)plan.island_offsets.push_back(plan.island_offsets.back()+count);
    plan.island_order.resize(contacts);auto next=plan.island_offsets;
    for(std::size_t k=0;k<contacts;++k)plan.island_order[next[island[k]]++]=static_cast<Index>(k);

    // Bounded greedy coloring. High-degree contacts enter a serial tail rather
    // than an unbounded color search. Never silently parallelize that tail.
    std::vector<std::uint64_t> used(bodies,0);
    std::vector<unsigned> color(contacts,colors);
    std::vector<std::size_t> color_counts(colors,0);
    const auto mask=colors==64?std::numeric_limits<std::uint64_t>::max():((std::uint64_t{1}<<colors)-1);
    unsigned count=0;
    for(std::size_t k=0;k<contacts;++k) {
        const auto occupied=(movable[a[k]]?used[a[k]]:0)|(movable[b[k]]?used[b[k]]:0);
        const auto available=mask&~occupied;
        if(!available){plan.serial_order.push_back(static_cast<Index>(k));continue;}
        const unsigned c=first_set(available);color[k]=c;++color_counts[c];count=std::max(count,c+1);
        const auto bit=std::uint64_t{1}<<c;
        if(movable[a[k]])used[a[k]]|=bit;
        if(movable[b[k]])used[b[k]]|=bit;
    }
    plan.color_offsets.push_back(0);
    for(unsigned c=0;c<count;++c)plan.color_offsets.push_back(plan.color_offsets.back()+color_counts[c]);
    plan.color_order.resize(plan.color_offsets.back());next=plan.color_offsets;
    for(std::size_t k=0;k<contacts;++k)if(color[k]<colors)plan.color_order[next[color[k]]++]=static_cast<Index>(k);
    return plan;
}

class Cache {
    struct Entry {std::vector<Index>a,b;std::vector<std::uint8_t>movable;unsigned colors;Plan plan;};
    std::unique_ptr<Entry> entry;
public:
    std::size_t builds=0,hits=0;
    const Plan& prepare(const std::vector<Index>& a,const std::vector<Index>& b,
                        const std::vector<std::uint8_t>& movable,unsigned colors=32) {
        if(entry&&entry->colors==colors&&entry->a==a&&entry->b==b&&entry->movable==movable){++hits;return entry->plan;}
        // Build/copy before publication: failed input never corrupts old state.
        auto fresh=std::make_unique<Entry>(Entry{a,b,movable,colors,build(a,b,movable,colors)});
        entry.swap(fresh);++builds;return entry->plan;
    }
};

// The callback applies one complete already-validated contact block. It must
// write only its declared mutable endpoints and contact-local exclusive state;
// shared reductions/read-write aliases need separate handling. Physics solving
// may instead consume the plan in its own gated trial buffers.
template<class Apply>
inline void visit_colors(const Plan& plan,int threads,Apply&& apply) {
    static_assert(noexcept(apply(Index{})),"Contact application must be noexcept");
    if(threads<1)throw std::invalid_argument("Positive worker count required");
#ifdef _OPENMP
    if(threads>omp_get_num_procs())throw std::invalid_argument("Worker count exceeds host processors");
    #pragma omp parallel num_threads(threads) if(threads>1)
    {
        for(std::size_t c=0;c+1<plan.color_offsets.size();++c) {
            #pragma omp for schedule(static)
            for(std::size_t i=plan.color_offsets[c];i<plan.color_offsets[c+1];++i)apply(plan.color_order[i]);
            // Implicit barrier prevents two colors writing the same body.
        }
    }
#else
    if(threads!=1)throw std::invalid_argument("Parallel schedule requires OpenMP build");
    for(auto k:plan.color_order)apply(k);
#endif
    for(auto k:plan.serial_order)apply(k);
}
} // namespace contact_schedule
