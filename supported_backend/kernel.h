#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace supported_fast {
struct Input {double m,I,R,N,drive,v,w,spin,h,mu_s,mu_d,mu_r,a_r,mu_n,a_n;};
struct Output {std::array<double,11> fields;int phases;};
inline double sign(double x) {return (x>0)-(x<0);}

// Normalize rotation to rim speed q=R*w and couple to force b=M/R.
// Both mobility rows now have units 1/mass. No candidate arrays or dimension-
// mixed tolerances; precompute reciprocals/capacities once per update.
inline Output advance(const Input& x) {
    constexpr double eps=std::numeric_limits<double>::epsilon();
    const double inverse_m=1/x.m,inverse_rot=x.R*x.R/x.I,inverse_R=1/x.R;
    const double mobility=inverse_m+inverse_rot;
    const double cap=x.mu_r*x.a_r*inverse_R*x.N,static_cap=x.mu_s*x.N,dynamic_cap=x.mu_d*x.N;
    const double force_tolerance=64*eps*std::max({static_cap,dynamic_cap,std::abs(x.drive),cap,1e-300});
    const double slip_tolerance=force_tolerance*(inverse_m+2*inverse_rot);
    const double rolling_tolerance=2*force_tolerance*inverse_rot;
    const double inverse_mobility=1/mobility;
    double v=x.v,q=x.R*x.w,time=0,dx=0,p=0,L=0,loss_f=0,loss_r=0;int phases=0;
    while(time<x.h) {
        if(++phases>32)throw std::runtime_error("branch budget");
        const double tolerance=std::min(1e-12,64*eps*std::max(std::abs(v),std::abs(q)));
        double u=v-q;
        if(std::abs(u)<=tolerance) {v=q;u=0;}
        if(std::abs(q)<=tolerance)q=0;
        const bool sliding=std::abs(u)>tolerance,rolling=std::abs(q)>tolerance;
        double f=0,b=0,dv=0,dq=0,du=0;int su=0,sq=0;bool found=false;
        // Established nonzero directions need exactly one candidate. At zero,
        // retain the reference order 0,-1,+1 and solve the static constraints.
        if(sliding&&rolling) {
            // Continued sliding/rolling has known directions: no active-set
            // search or static/onset checks can change this branch.
            su=static_cast<int>(sign(u));sq=static_cast<int>(sign(q));
            f=-dynamic_cap*su;b=-cap*sq;
            dv=(x.drive+f)*inverse_m;dq=(-f+b)*inverse_rot;du=dv-dq;found=true;
        }
        const int nu=sliding?1:3,nq=rolling?1:3;
        for(int i=0;i<nu&&!found;++i)for(int j=0;j<nq&&!found;++j) {
            su=sliding?static_cast<int>(sign(u)):(i==0?0:(i==1?-1:1));
            sq=rolling?static_cast<int>(sign(q)):(j==0?0:(j==1?-1:1));
            if(su==0&&sq==0) {f=-x.drive;b=f;}
            else if(su==0) {b=-cap*sq;f=(b*inverse_rot-x.drive*inverse_m)*inverse_mobility;}
            else if(sq==0) {f=-dynamic_cap*su;b=f;}
            else {f=-dynamic_cap*su;b=-cap*sq;}
            if(su==0&&std::abs(f)>static_cap+force_tolerance)continue;
            if(sq==0&&std::abs(b)>cap+force_tolerance)continue;
            dv=(x.drive+f)*inverse_m;dq=(-f+b)*inverse_rot;du=dv-dq;
            if(!sliding&&su&&su*du<=slip_tolerance)continue;
            if(!rolling&&sq&&sq*dq<=rolling_tolerance)continue;
            found=true;
        }
        if(!found)throw std::runtime_error("no admissible branch");
        double h=x.h-time,tu=std::numeric_limits<double>::infinity(),tq=tu;
        if(su&&u*du<0)tu=-u/du;
        if(sq&&q*dq<0)tq=-q/dq;
        if(tu>0&&tu<h)h=tu;
        if(tq>0&&tq<h)h=tq;
        const double event_tolerance=16*eps*std::max(x.h,h);
        const bool stop_u=std::abs(tu-h)<=event_tolerance,stop_q=std::abs(tq-h)<=event_tolerance;
        if(h<=0)throw std::runtime_error("nonadvancing event");
        dx+=v*h+.5*dv*h*h;p+=f*h;L+=x.R*b*h;
        loss_f-=f*(u*h+.5*du*h*h);loss_r-=b*(q*h+.5*dq*h*h);
        v+=dv*h;q+=dq*h;time+=h;
        if(stop_q||sq==0)q=0;
        if(stop_u||su==0)v=q;
        if(x.h-time<=8*eps*x.h)time=x.h;
    }
    const double spin_capacity=x.mu_n*x.a_n*x.N*x.h,arrest=x.I*std::abs(x.spin);
    const double Ln=-sign(x.spin)*std::min(spin_capacity,arrest);
    const double spin=spin_capacity>=arrest?0:x.spin+Ln/x.I,w=q*inverse_R;
    const double loss_n=.5*x.I*(x.spin*x.spin-spin*spin);
    const double before=.5*x.m*x.v*x.v+.5*x.I*(x.w*x.w+x.spin*x.spin);
    const double after=.5*x.m*v*v+.5*x.I*(w*w+spin*spin),work=x.drive*dx;
    const double loss=loss_f+loss_r+loss_n,residual=after-before-work+loss;
    const double scale=std::max({before,after,std::abs(work),loss,1e-300});
    // Positive validated m/I and non-fast-math IEEE arithmetic: nonfinite
    // velocities, distance/work or loss channels make this residual nonfinite.
    // Static force/couple integrals can overflow without doing work, so check
    // them independently. This removes redundant per-channel finite tests.
    if(!std::isfinite(residual)||!std::isfinite(p)||!std::isfinite(L))throw std::runtime_error("nonfinite response");
    if(std::min({loss_f,loss_r,loss_n}) < -1e-11*scale||std::abs(residual)>2e-10*scale)throw std::runtime_error("energy gate");
    return {{v,w,spin,dx,p,L,Ln,loss_f,loss_r,loss_n,residual},phases};
}
}
