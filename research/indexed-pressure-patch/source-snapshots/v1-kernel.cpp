// Ordered CPU indexed gather -> local patch -> body scatter, no fast-math.
// Frozen planar footprints, synthetic inputs. Not a scene/contact-network solve.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
using V = std::array<double,3>;
V cross(const V& a,const V& b) {return {a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]};}
double dot(const V&a,const V&b) {return a[0]*b[0]+a[1]*b[1]+a[2]*b[2];}
struct Wrench {V f{},m{};};
struct Patch {
    std::vector<double> x,y,w;
    std::array<double,9> gram{};
    double xmin=0,xmax=0,ymin=0,ymax=0;
};
struct Fields {
    std::array<std::vector<double>,3> v,w,force,moment;
    explicit Fields(size_t n) {for(auto* a:{&v,&w,&force,&moment}) for(auto &x:*a) x.resize(n);}
    V get(const std::array<std::vector<double>,3>& a,size_t k)const {return {a[0][k],a[1][k],a[2][k]};}
    void set(std::array<std::vector<double>,3>& a,size_t k,const V&x) {for(int j=0;j<3;++j) a[j][k]=x[j];}
};
// Body owner k is gathered from incidence; site j belongs to shared footprint.
struct Contacts {
    std::array<std::vector<double>,3> compression;
    std::vector<double> K,C,mu;
    std::vector<size_t> start,owner;
    std::vector<double> sign;
    std::vector<V> arm;
    explicit Contacts(size_t n):K(n),C(n),mu(n),start(n+1),owner(2*n),sign(2*n),arm(2*n) {
        for(auto& x:compression) x.resize(n);
    }
};

Wrench reference(const Patch&p,const V&d,const V&u,const V&omega,double K,double C,double mu) {
    Wrench result;
    const V n{0,0,1};
    for(size_t j=0;j<p.x.size();++j) {
        const V offset{p.x[j],p.y[j],0};
        const V rotational=cross(omega,offset);
        V local{};for(int a=0;a<3;++a) local[a]=u[a]+rotational[a];
        const double delta=d[0]+offset[1]*d[1]-offset[0]*d[2];
        const double pressure=delta>0?std::max(0.,K*delta-C*dot(n,local))*p.w[j]:0;
        V tangent=local;tangent[2]=0;
        const double speed=std::sqrt(dot(tangent,tangent));
        V force{};for(int a=0;a<3;++a) force[a]=pressure*n[a]-(speed>0?mu*pressure*tangent[a]/speed:0);
        const V moment=cross(offset,force);
        for(int a=0;a<3;++a) {result.f[a]+=force[a];result.m[a]+=moment[a];}
    }
    return result;
}

double lower(const Patch&p,const V&a) {
    return a[0]+std::min(a[1]*p.ymin,a[1]*p.ymax)+std::min(-a[2]*p.xmin,-a[2]*p.xmax);
}
Wrench compact(const Patch&p,const V&d,const V&u,const V&omega,double K,double C,double mu) {
    const V normal_speed{u[2],omega[0],omega[1]};
    V trial{};for(int a=0;a<3;++a) trial[a]=K*d[a]-C*normal_speed[a];
    // Conservative box test: may decline caching, never overlooks an open site.
    const bool loaded=lower(p,d)>0 && lower(p,trial)>0;
    Wrench result;
    if(loaded) {
        V integrated{};
        for(int a=0;a<3;++a) for(int b=0;b<3;++b) integrated[a]+=p.gram[3*a+b]*trial[b];
        result.f[2]=integrated[0];result.m[0]=integrated[1];result.m[1]=integrated[2];
        // Zero axial twist gives uniform tangential slip at every planar site.
        if(omega[2]==0 || mu==0) {
            const double speed=std::hypot(u[0],u[1]);
            const double tx=speed>0?-mu*u[0]/speed:0,ty=speed>0?-mu*u[1]/speed:0;
            result.f[0]=tx*result.f[2];result.f[1]=ty*result.f[2];
            result.m[2]=-ty*integrated[2]-tx*integrated[1];
            return result;
        }
    }
    for(size_t j=0;j<p.x.size();++j) {
        const double delta=d[0]+p.y[j]*d[1]-p.x[j]*d[2];
        const double un=u[2]+p.y[j]*omega[0]-p.x[j]*omega[1];
        const double fn=delta>0?p.w[j]*std::max(0.,K*delta-C*un):0;
        const double vx=u[0]-omega[2]*p.y[j],vy=u[1]+omega[2]*p.x[j];
        const double speed=std::sqrt(vx*vx+vy*vy);
        const double scale=speed>0?-mu*fn/speed:0;
        const double fx=scale*vx,fy=scale*vy;
        result.f[0]+=fx;result.f[1]+=fy;result.m[2]+=p.x[j]*fy-p.y[j]*fx;
        if(!loaded) {result.f[2]+=fn;result.m[0]+=p.y[j]*fn;result.m[1]-=p.x[j]*fn;}
    }
    return result;
}

[[gnu::noinline]] void run(const Patch&p,Contacts&c,Fields&b,std::vector<Wrench>&out,bool fast) {
    for(auto* a:{&b.force,&b.moment}) for(auto& x:*a) std::fill(x.begin(),x.end(),0.);
    for(size_t i=0;i<c.K.size();++i) {
        V u{},omega{},d{};
        for(size_t z=c.start[i];z<c.start[i+1];++z) {
            const size_t k=c.owner[z];const V v=b.get(b.v,k),w=b.get(b.w,k),rot=cross(w,c.arm[z]);
            for(int a=0;a<3;++a) {u[a]+=c.sign[z]*(v[a]+rot[a]);omega[a]+=c.sign[z]*w[a];}
        }
        for(int a=0;a<3;++a) d[a]=c.compression[a][i];
        out[i]=fast?compact(p,d,u,omega,c.K[i],c.C[i],c.mu[i]):reference(p,d,u,omega,c.K[i],c.C[i],c.mu[i]);
        for(size_t z=c.start[i];z<c.start[i+1];++z) {
            const size_t k=c.owner[z];const V lever=cross(c.arm[z],out[i].f);
            for(int a=0;a<3;++a) {b.force[a][k]+=c.sign[z]*out[i].f[a];b.moment[a][k]+=c.sign[z]*(lever[a]+out[i].m[a]);}
        }
    }
}

int main(int argc,char**argv) {
    if(argc!=2) return 2;
    std::ifstream input(argv[1]);size_t count;input>>count;
    std::vector<Patch> patches(count);std::vector<std::string> names(count);
    for(size_t i=0;i<count;++i) {
        size_t nodes;input>>names[i]>>nodes;auto&p=patches[i];p.x.resize(nodes);p.y.resize(nodes);p.w.resize(nodes);
        for(size_t j=0;j<nodes;++j) {
            input>>p.x[j]>>p.y[j]>>p.w[j];
            p.xmin=std::min(p.xmin,p.x[j]);p.xmax=std::max(p.xmax,p.x[j]);
            p.ymin=std::min(p.ymin,p.y[j]);p.ymax=std::max(p.ymax,p.y[j]);
            const V basis{1,p.y[j],-p.x[j]};
            for(int a=0;a<3;++a) for(int b=0;b<3;++b) p.gram[3*a+b]+=p.w[j]*basis[a]*basis[b];
        }
    }
    if(!input) return 3;
    std::cout<<std::setprecision(17)<<"{\"batches\":[";
    bool first=true;
    // Modes cover loaded sliding/rolling, loaded mixed twist, and opening/clipping.
    for(size_t shape=0;shape<count;++shape) for(int mode=0;mode<3;++mode) {
        if(shape==0 && mode==1) continue; // 2D has no axial twist.
        for(size_t n:{100ul,10000ul,100000ul,1000000ul}) {
            Contacts c(n);Fields b(n+1);std::vector<Wrench> base(n),fast(n);
            for(size_t i=0;i<n;++i) {
                const double f=double((i*7919)%10007)/10007;
                c.start[i]=2*i;c.owner[2*i]=i+1;c.owner[2*i+1]=0;c.sign[2*i]=1;c.sign[2*i+1]=-1;
                // Same world contact point for both incidences; wall center is zero.
                c.arm[2*i]={0,0,-.05};c.arm[2*i+1]={.001*f,.002*f,0};
                const V omega=shape==0?V{0,1+f,0}:V{.5+f,1+f,mode==1?10+20*f:0};
                const V target{.1+.4*f,shape==0?0:.03+.2*f,-.03+.02*f};
                const V lever=cross(omega,c.arm[2*i]);V body{};for(int a=0;a<3;++a) body[a]=target[a]-lever[a];
                b.set(b.v,i+1,body);b.set(b.w,i+1,omega);
                c.compression[0][i]=mode==2?-.001+.002*f:.002+.001*f;
                c.compression[1][i]=shape==0?0:mode==2?.2:.01;
                c.compression[2][i]=mode==2?-.1:.01;
                c.K[i]=10000+2000*f;c.C[i]=20;c.mu[i]=.4;
            }
            c.start[n]=2*n;
            run(patches[shape],c,b,base,false);
            const auto saved_force=b.force,saved_moment=b.moment;
            run(patches[shape],c,b,fast,true);
            double error=0,scatter_error=0;
            for(size_t i=0;i<n;++i) for(int a=0;a<3;++a) {
                error=std::max(error,std::abs(base[i].f[a]-fast[i].f[a])/std::max(1.,std::abs(base[i].f[a])));
                error=std::max(error,std::abs(base[i].m[a]-fast[i].m[a])/std::max(1.,std::abs(base[i].m[a])));
            }
            for(size_t k=0;k<n+1;++k) for(int a=0;a<3;++a) {
                scatter_error=std::max(scatter_error,std::abs(saved_force[a][k]-b.force[a][k])/std::max(1.,std::abs(saved_force[a][k])));
                scatter_error=std::max(scatter_error,std::abs(saved_moment[a][k]-b.moment[a][k])/std::max(1.,std::abs(saved_moment[a][k])));
            }
            if(error>1e-11 || scatter_error>1e-10) throw std::runtime_error("native equivalence");
            std::array<double,2> seconds{};
            const size_t repeats=std::max(1ul,100000ul/n);
            for(int which=0;which<2;++which) {
                std::vector<double> samples;
                for(int trial=0;trial<5;++trial) {
                    const auto start=std::chrono::steady_clock::now();
                    for(size_t r=0;r<repeats;++r) run(patches[shape],c,b,which?fast:base,which);
                    samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats);
                }
                std::sort(samples.begin(),samples.end());seconds[which]=samples[2];
            }
            double checksum=0;for(const auto&w:fast) for(int a=0;a<3;++a) checksum+=w.f[a]+w.m[a];
            if(!std::isfinite(checksum)) return 4;
            if(!first) std::cout<<",";
            first=false;
            std::cout<<"{\"shape\":\""<<names[shape]<<"\",\"mode\":"<<mode<<",\"count\":"<<n
                     <<",\"sites\":"<<patches[shape].x.size()<<",\"reference_seconds\":"<<seconds[0]
                     <<",\"compact_seconds\":"<<seconds[1]<<",\"speedup\":"<<seconds[0]/seconds[1]
                     <<",\"maximum_scaled_wrench_error\":"<<error<<",\"maximum_scaled_scatter_error\":"<<scatter_error
                     <<",\"checksum\":"<<checksum<<",\"array_bytes\":"
                     <<(n*9*sizeof(double)+(n+1)*12*sizeof(double)+(n+1)*sizeof(size_t)+2*n*(sizeof(size_t)+4*sizeof(double))+2*n*sizeof(Wrench))<<"}";
            if(n==100) {
                // Auditable controls in raw output, not only checksums.
                std::cout.flush();
            }
        }
    }
    std::cout<<"],\"controls\":[";first=true;
    for(size_t shape=0;shape<count;++shape) for(int i=0;i<12;++i) {
        const V d{-.001+.0004*i,.04,-.02},u{.1+.02*i,shape==0?0:.2,-.03},omega=shape==0?V{0,2,0}:V{2,3,double(i)};
        const auto r=compact(patches[shape],d,u,omega,10000,20,.4);
        if(!first) std::cout<<",";
        first=false;
        std::cout<<"{\"shape\":\""<<names[shape]<<"\",\"compression\":["<<d[0]<<","<<d[1]<<","<<d[2]
                 <<"],\"velocity\":["<<u[0]<<","<<u[1]<<","<<u[2]<<"],\"omega\":["<<omega[0]<<","<<omega[1]<<","<<omega[2]
                 <<"],\"wrench\":["<<r.f[0]<<","<<r.f[1]<<","<<r.f[2]<<","<<r.m[0]<<","<<r.m[1]<<","<<r.m[2]<<"]}";
    }
    std::cout<<"]}\n";
}
