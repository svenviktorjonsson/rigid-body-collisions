// Additional numerical searches. Intermediate impulses are never applied.
#pragma once
#include "coulomb_fb_direct.h"
#include "coulomb_neutral.h"
namespace circular_restart {
struct Stats {restart_direct::Stats direct;restart_neutral::Stats neutral;int svd_calls=0,iteration_steps=0;double residual=0;};
inline bool solve(const btMatrixXu&A,const btVectorXu&b,btVectorXu&x,const btVectorXu&hi,const btAlignedObjectArray<int>&dep,double tolerance,Stats&stats,int svd_limit=1024){
 svd_limit=std::min(svd_limit,b.rows()<=64?1024:256);
 if(svd_limit<256)return false;
 auto trial=x;
 bool accepted=restart_direct::solve(A,b,trial,hi,dep,tolerance,stats.direct);
 stats.svd_calls=stats.direct.svd_calls;stats.iteration_steps=stats.direct.iteration_steps;stats.residual=stats.direct.residual;
 if(accepted){x=trial;return true;}
 trial=x;accepted=restart_neutral::solve(A,b,trial,hi,dep,tolerance,stats.neutral,svd_limit-stats.svd_calls);
 stats.svd_calls+=stats.neutral.svd_calls;stats.iteration_steps+=stats.neutral.iteration_steps;stats.residual=stats.neutral.residual;
 if(accepted)x=trial;
 return accepted;
}
}
