#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace supported {
struct Input { double m,I,R,N,drive,v,w,spin,h,mu_s,mu_d,mu_r,a_r,mu_n,a_n; };
struct Output { std::array<double,11> fields; int phases; };
inline double sign(double x) { return (x>0)-(x<0); }
inline Output advance(const Input& x) {
    constexpr double eps=std::numeric_limits<double>::epsilon();
    const double A=1/x.m+x.R*x.R/x.I,cap=x.mu_r*x.a_r*x.N;
    const double ftol=64*eps*std::max({x.N,std::abs(x.drive),cap/x.R,1e-300});
    double v=x.v,w=x.w,time=0,dx=0,p=0,L=0,loss_f=0,loss_r=0;int phases=0;
    while(time<x.h) {
        if(++phases>32)throw std::runtime_error("branch budget");
        const double tol=std::min(1e-12,64*eps*std::max(std::abs(v),x.R*std::abs(w)));
        double u=v-x.R*w;
        if(std::abs(u)<=tol) { v=x.R*w;u=0; }
        if(std::abs(w)<=tol/x.R)w=0;
        const std::array<int,3> us=std::abs(u)>tol?std::array<int,3>{static_cast<int>(sign(u)),0,0}:std::array<int,3>{0,-1,1};
        const std::array<int,3> ws=std::abs(w)>tol/x.R?std::array<int,3>{static_cast<int>(sign(w)),0,0}:std::array<int,3>{0,-1,1};
        const int nu=std::abs(u)>tol?1:3,nw=std::abs(w)>tol/x.R?1:3;
        double f=0,M=0,dv=0,dw=0,du=0;int su=0,sw=0;bool found=false;
        for(int i=0;i<nu&&!found;++i)for(int j=0;j<nw&&!found;++j) {
            su=us[i];sw=ws[j];
            if(su==0&&sw==0) { f=-x.drive;M=x.R*f; }
            else if(su==0) { M=-cap*sw;f=(x.R*M/x.I-x.drive/x.m)/A; }
            else if(sw==0) { f=-x.mu_d*x.N*su;M=x.R*f; }
            else { f=-x.mu_d*x.N*su;M=-cap*sw; }
            if(su==0&&std::abs(f)>x.mu_s*x.N+ftol)continue;
            if(sw==0&&std::abs(M)>cap+x.R*ftol)continue;
            dv=(x.drive+f)/x.m;dw=(-x.R*f+M)/x.I;du=dv-x.R*dw;
            const double atol=ftol*std::max(A,x.R/x.I);
            if(std::abs(u)<=tol&&su&&su*du<=atol)continue;
            if(std::abs(w)<=tol/x.R&&sw&&sw*dw<=atol/x.R)continue;
            found=true;
        }
        if(!found)throw std::runtime_error("no admissible branch");
        double h=x.h-time,tu=std::numeric_limits<double>::infinity(),tw=tu;
        if(su&&u*du<0)tu=-u/du;
        if(sw&&w*dw<0)tw=-w/dw;
        if(tu>0&&tu<h)h=tu;
        if(tw>0&&tw<h)h=tw;
        const bool stop_u=std::abs(tu-h)<=16*eps*std::max(x.h,h),stop_w=std::abs(tw-h)<=16*eps*std::max(x.h,h);
        if(h<=0)throw std::runtime_error("nonadvancing event");
        dx+=v*h+.5*dv*h*h;p+=f*h;L+=M*h;
        loss_f-=f*(u*h+.5*du*h*h);loss_r-=M*(w*h+.5*dw*h*h);
        v+=dv*h;w+=dw*h;time+=h;
        if(stop_w||sw==0)w=0;
        if(stop_u||su==0)v=x.R*w;
        if(x.h-time<=8*eps*x.h)time=x.h;
    }
    const double Ln=-sign(x.spin)*std::min(x.mu_n*x.a_n*x.N*x.h,x.I*std::abs(x.spin));
    const double spin=x.spin+Ln/x.I,loss_n=.5*x.I*(x.spin*x.spin-spin*spin);
    const double before=.5*x.m*x.v*x.v+.5*x.I*(x.w*x.w+x.spin*x.spin);
    const double after=.5*x.m*v*v+.5*x.I*(w*w+spin*spin),work=x.drive*dx;
    const double loss=loss_f+loss_r+loss_n,residual=after-before-work+loss;
    const double scale=std::max({before,after,std::abs(work),loss,1e-300});
    if(std::min({loss_f,loss_r,loss_n}) < -1e-11*scale||std::abs(residual)>2e-10*scale)throw std::runtime_error("energy gate");
    return {{v,w,spin,dx,p,L,Ln,loss_f,loss_r,loss_n,residual},phases};
}
}
