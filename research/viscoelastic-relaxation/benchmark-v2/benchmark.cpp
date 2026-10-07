// Local isolated response kernels. No collision detection or contact network.
// Compile without fast-math; exact rolling torque includes its static reaction.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

struct Input { double omega, radius, mass, alpha, relaxation, dt, factor; };
struct Output { double omega, linear_impulse, angular_impulse, removed_energy; };
using Polynomial = std::array<double,4>;

double lookup(double beta, const std::vector<Polynomial>& table) {
    if (!std::isfinite(beta) || beta<0 || beta>8) throw std::runtime_error("table range");
    const int bin=std::min(static_cast<int>(beta*64),511);
    const double x=beta-static_cast<double>(bin)/64;
    const auto &c=table[bin];
    return ((c[0]*x+c[1])*x+c[2])*x+c[3];
}

[[gnu::noinline]] void rolling(const std::vector<Input>& inputs, std::vector<Output>& outputs, int mode) {
    for (size_t i=0;i<inputs.size();++i) {
        const auto &p=inputs[i];
        const double before=p.omega;
        double after;
        if (mode==0) after=std::max(0.,before-.02*9.81*p.dt/((1+p.alpha)*p.radius));
        else if (mode==1) after=before*std::exp(-p.relaxation*9.81*p.dt/((1+p.alpha)*p.radius));
        else after=before*p.factor;
        const double inertia=p.alpha*p.mass*p.radius*p.radius;
        const double effective=inertia+p.mass*p.radius*p.radius;
        outputs[i]={after,p.mass*p.radius*(after-before),effective*(after-before),
                    .5*effective*(before*before-after*after)};
    }
}

[[gnu::noinline]] void normal(const std::vector<Input>& inputs,std::vector<Output>& outputs,
                              const std::vector<Polynomial>& table) {
    for(size_t i=0;i<inputs.size();++i) {
        const auto &p=inputs[i];const double speed=p.omega;
        // Fixed synthetic effective parameter; size/speed conversion included.
        const double beta=.25*.05/p.radius*std::pow(speed,.2);
        const double restitution=lookup(beta,table);
        outputs[i]={restitution*speed,p.mass*(1+restitution)*speed,0.,
                    .5*p.mass*speed*speed*(1-restitution*restitution)};
    }
}

template<class Function> double timed(Function function,size_t repeats) {
    auto start=std::chrono::steady_clock::now();
    for(size_t i=0;i<repeats;++i) function();
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()/repeats;
}

int main(int argc,char **argv) {
    if(argc!=2) return 2;
    std::ifstream stream(argv[1]);std::vector<Polynomial> table(512);
    for(auto &c:table) for(double &x:c) if(!(stream>>x)) return 3;
    std::cout<<std::setprecision(17);
    std::cout<<"{\"scope\":\"Synthetic local response kernels, not full scene or native engine validation\",\"batches\":[";
    bool first=true;
    for(const size_t size:{100ul,10000ul,1000000ul}) {
        std::vector<Input> inputs(size);std::vector<Output> outputs(size);
        for(size_t i=0;i<size;++i) {
            const double f=static_cast<double>((i*7919)%10007)/10007;
            const double R=.01+.09*f,omega=.1+50*f,A=.0001+.005*f;
            const double dt=.0001+.001*(1-f);
            inputs[i]={omega,R,.01+f,.4,A,dt,std::exp(-A*9.81*dt/(1.4*R))};
        }
        if(!first) std::cout<<",";
        first=false;
        std::cout<<"{\"count\":"<<size<<",\"input_output_array_bytes\":"<<size*(sizeof(Input)+sizeof(Output))<<",\"kernels\":[";
        for(int mode=0;mode<4;++mode) {
            auto run=[&](){if(mode<3) rolling(inputs,outputs,mode);else normal(inputs,outputs,table);};
            run();const size_t repeats=std::max(2ul,2000000ul/size);
            std::vector<double> durations;
            for(int trial=0;trial<5;++trial) durations.push_back(timed(run,repeats));
            std::sort(durations.begin(),durations.end());
            double checksum=0;for(const auto &p:outputs) checksum+=p.omega+p.linear_impulse+p.angular_impulse+p.removed_energy;
            if(!std::isfinite(checksum)) return 4;
            if(mode) std::cout<<",";
            const char *label=mode==0?"constant_rolling":mode==1?"relaxation_exp":"relaxation_cached_factor";
            if(mode==3) label="normal_size_speed_and_table";
            std::cout<<"{\"kernel\":\""<<label<<"\",\"median_seconds\":"<<durations[2]
                     <<",\"nanoseconds_per_response\":"<<durations[2]*1e9/size<<",\"checksum\":"<<checksum<<"}";
        }
        std::cout<<"]}";
    }
    std::cout<<"],\"controls\":[";
    for(int i=0;i<16;++i) {
        const double b=.001+i*.45;
        Input p{.5+i,.01+i*.003,1.+i*.1,.4,.002,.001,0.};
        std::vector<Input> input{p};std::vector<Output> out(1);rolling(input,out,1);
        if(i) std::cout<<",";
        std::cout<<"{\"beta\":"<<b<<",\"restitution\":"<<lookup(b,table)<<",\"rolling_input\":["
                 <<p.mass<<","<<p.radius<<","<<p.alpha<<","<<p.omega<<","<<p.dt<<","<<p.relaxation
                 <<"],\"rolling_output\":["<<out[0].omega<<","<<out[0].linear_impulse<<","<<out[0].angular_impulse<<","<<out[0].removed_energy<<"]}";
    }
    std::cout<<"]}\n";
}
