#include "../../contact_backend/schedule.h"
#include <array>
#include <cmath>
#include <iostream>
#include <random>

using contact_schedule::Index;
using Vec=std::array<double,3>;
inline void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
inline Vec cross(const Vec& a,const Vec& b) noexcept {return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
inline Vec sub(const Vec& a,const Vec& b) noexcept{return {a[0]-b[0],a[1]-b[1],a[2]-b[2]};}

void check(const std::vector<Index>& a,const std::vector<Index>& b,
           const std::vector<std::uint8_t>& mutable_body,const contact_schedule::Plan& plan) {
    const std::size_t n=a.size();
    require(plan.contact_count==n&&plan.body_count==mutable_body.size(),"Plan extent");
    std::vector<int> seen(n,0);
    for(auto k:plan.island_order){require(k<n,"Island index");++seen[k];}
    for(auto count:seen)require(count==1,"Island lost/duplicated contact");
    std::fill(seen.begin(),seen.end(),0);
    for(std::size_t c=0;c+1<plan.color_offsets.size();++c){
        std::vector<bool> written(mutable_body.size(),false);
        for(std::size_t i=plan.color_offsets[c];i<plan.color_offsets[c+1];++i){
            const auto k=plan.color_order[i];require(k<n,"Color index");++seen[k];
            for(auto body:{a[k],b[k]})if(mutable_body[body]){require(!written[body],"Parallel write conflict");written[body]=true;}
        }
    }
    for(auto k:plan.serial_order){require(k<n,"Serial index");++seen[k];}
    for(auto count:seen)require(count==1,"Color lost/duplicated contact");
    // Independent BFS on contacts linked through mutable-body incidence.
    std::vector<std::vector<Index>> incidence(mutable_body.size());
    for(std::size_t k=0;k<n;++k)for(auto body:{a[k],b[k]})if(mutable_body[body])incidence[body].push_back(static_cast<Index>(k));
    std::vector<int> label(n,-1);int groups=0;
    for(std::size_t k=0;k<n;++k)if(label[k]<0){
        label[k]=groups;std::vector<Index> queue{static_cast<Index>(k)};
        for(std::size_t at=0;at<queue.size();++at)for(auto body:{a[queue[at]],b[queue[at]]})if(mutable_body[body])
            for(auto neighbor:incidence[body])if(label[neighbor]<0){label[neighbor]=groups;queue.push_back(neighbor);}
        ++groups;
    }
    require(plan.island_offsets.size()==static_cast<std::size_t>(groups)+1,"Island count differs from BFS");
    std::vector<bool> used(static_cast<std::size_t>(groups),false);
    for(std::size_t island=0;island+1<plan.island_offsets.size();++island){
        require(plan.island_offsets[island]<plan.island_offsets[island+1],"Empty island");
        const auto component=label[plan.island_order[plan.island_offsets[island]]];
        require(!used[component],"BFS component split");used[component]=true;
        for(std::size_t i=plan.island_offsets[island];i<plan.island_offsets[island+1];++i)
            require(label[plan.island_order[i]]==component,"Different BFS components joined");
    }
}

struct Body {Vec position{},p{},L{};};
struct Impulse {Vec point{},force{},couple{};};

void momentum_control(int dimension,int workers) {
    std::vector<Index> a,b;std::vector<std::uint8_t> movable(80,1);movable.back()=0;
    for(Index k=0;k<79;++k){a.push_back(k);b.push_back(79);}
    for(Index k=0;k<79;++k){a.push_back(k);b.push_back((k+1)%79);}
    // High-degree mutable hub, repeated contacts and independent pure couples.
    for(Index k=1;k<79;++k){a.push_back(0);b.push_back(k);}
    auto plan=contact_schedule::build(a,b,movable,8);check(a,b,movable,plan);
    require(!plan.serial_order.empty(),"Overflow control did not overflow");
    std::vector<Body> initial(movable.size());
    for(std::size_t k=0;k<initial.size();++k)initial[k].position={.01*k,.02*k,dimension==3?.03*k:0};
    std::vector<Impulse> impulses(a.size());
    for(std::size_t k=0;k<a.size();++k){
        impulses[k].point={.005*k,.004*k,dimension==3?.003*k:0};
        if(k%5!=0)impulses[k].force={.002*(k+1),-.003*(k+1),dimension==3?.004*(k+1):0};
        impulses[k].couple={dimension==3?.0002*(k+1):0,dimension==3?-.0001*(k+1):0,.0003*(k+1)};
    }
    auto apply=[&](std::vector<Body>& bodies,Index k) noexcept {
        for(int side=0;side<2;++side){const auto body=side==0?a[k]:b[k];if(!movable[body])continue;
            const double sign=side==0?1.:-1.;const auto lever=cross(sub(impulses[k].point,bodies[body].position),impulses[k].force);
            for(int axis=0;axis<3;++axis){bodies[body].p[axis]+=sign*impulses[k].force[axis];bodies[body].L[axis]+=sign*(lever[axis]+impulses[k].couple[axis]);}
        }
    };
    auto serial=initial,parallel=initial,original=initial;
    contact_schedule::visit_colors(plan,1,[&](Index k) noexcept {apply(serial,k);});
    contact_schedule::visit_colors(plan,workers,[&](Index k) noexcept {apply(parallel,k);});
    for(std::size_t k=0;k<a.size();++k)apply(original,static_cast<Index>(k));
    for(std::size_t k=0;k<initial.size();++k){
        require(serial[k].p==parallel[k].p&&serial[k].L==parallel[k].L,"Worker output differs");
        for(int axis=0;axis<3;++axis){require(std::abs(serial[k].p[axis]-original[k].p[axis])<1e-12,"Linear impulse mismatch");
            require(std::abs(serial[k].L[axis]-original[k].L[axis])<1e-12,"Lever plus independent couple mismatch");}
    }
    Vec total_p{},total_L{},wall_p{},wall_L{};
    for(std::size_t k=0;k<initial.size();++k){auto orbital=cross(initial[k].position,parallel[k].p);
        for(int axis=0;axis<3;++axis){total_p[axis]+=parallel[k].p[axis];total_L[axis]+=parallel[k].L[axis]+orbital[axis];}}
    for(std::size_t k=0;k<a.size();++k)for(int side=0;side<2;++side)if(!movable[side==0?a[k]:b[k]]){
        const auto angular=cross(impulses[k].point,impulses[k].force);const double sign=side==0?1.:-1.;
        for(int axis=0;axis<3;++axis){wall_p[axis]+=sign*impulses[k].force[axis];wall_L[axis]+=sign*(angular[axis]+impulses[k].couple[axis]);}}
    for(int axis=0;axis<3;++axis){require(std::abs(total_p[axis]+wall_p[axis])<1e-11,"Global linear ledger");
        require(std::abs(total_L[axis]+wall_L[axis])<1e-11,"Global angular ledger");}
}

int main(){try {
    std::mt19937 random(20261007);
    for(int trial=0;trial<300;++trial){
        std::vector<std::uint8_t> movable(10+random()%80,1);movable.back()=0;
        std::vector<Index>a,b;
        for(int k=0;k<200;++k){Index u=random()%(movable.size()-1),v=random()%movable.size();if(u==v)v=static_cast<Index>((v+1)%movable.size());a.push_back(u);b.push_back(v);}
        auto plan=contact_schedule::build(a,b,movable,1+random()%64);check(a,b,movable,plan);
    }
    std::vector<Index>a{0,1,2},b{3,3,3};std::vector<std::uint8_t>movable{1,1,1,0};
    contact_schedule::Cache cache;const auto& initial=cache.prepare(a,b,movable);
    require(initial.island_offsets.size()==4&&initial.color_offsets.size()==2,"Static floor merges independent bodies");
    const auto* pointer=&initial;require(&cache.prepare(a,b,movable)==pointer&&cache.hits==1,"Topology cache did not reuse");
    b[0]=1;auto coupled=cache.prepare(a,b,movable);require(coupled.island_offsets.size()==3&&cache.builds==2,"Endpoint edit did not invalidate");
    b[0]=3;movable[3]=1;require(cache.prepare(a,b,movable).island_offsets.size()==2,"Body type did not invalidate");
    require(cache.prepare(a,b,movable,1).serial_order.size()==2,"Color budget did not invalidate");
    const auto builds=cache.builds;auto bad=b;bad[0]=100;
    try{cache.prepare(a,bad,movable,1);throw std::runtime_error("Invalid graph accepted");}catch(const std::invalid_argument&){}
    cache.prepare(a,b,movable,1);require(cache.builds==builds&&cache.hits==2,"Failed cache request damaged old plan");
    for(unsigned budget:{0u,65u}){try{contact_schedule::build(a,b,movable,budget);throw std::runtime_error("Invalid budget accepted");}catch(const std::invalid_argument&){}}
    try{contact_schedule::build({0},{1},{0,0});throw std::runtime_error("Read-only contact accepted");}catch(const std::invalid_argument&){}
    try{contact_schedule::build({0},{0},{1});throw std::runtime_error("Self contact accepted");}catch(const std::invalid_argument&){}
    try{contact_schedule::build({0},{1},{1,2});throw std::runtime_error("Invalid mask accepted");}catch(const std::invalid_argument&){}
    try{contact_schedule::build({0},{},{1});throw std::runtime_error("Unequal endpoints accepted");}catch(const std::invalid_argument&){}
    auto empty=contact_schedule::build({},{},{});check({},{},{},empty);
    contact_schedule::visit_colors(empty,1,[](Index) noexcept {});
    try{contact_schedule::visit_colors(empty,0,[](Index) noexcept {});throw std::runtime_error("Invalid workers accepted");}catch(const std::invalid_argument&){}
    int workers=1;
#ifdef _OPENMP
    workers=std::min(8,omp_get_num_procs());
#else
    try{contact_schedule::visit_colors(empty,2,[](Index) noexcept {});throw std::runtime_error("Parallel non-OpenMP accepted");}catch(const std::invalid_argument&){}
#endif
    momentum_control(2,workers);momentum_control(3,workers);
    std::cout<<"PASS: 300 graph/BFS/conflict controls; cache invalidation/atomicity; 2D/3D force and independent couple momentum; worker bit identity\n";
    return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
