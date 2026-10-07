"""Prepare an uncompiled two-build replay using the frozen independent checks."""
from pathlib import Path

root = Path(__file__).resolve().parent
text = (root / "frozen/spatial_backend/coulomb_replay.cpp").read_text()


def change(before, after):
    global text
    assert text.count(before) == 1, before
    text = text.replace(before, after)


change('if(argc<2||argc>3)throw std::runtime_error("Usage: spatial_coulomb_replay dump.json [iteration_budget]");',
       'if(argc<2||argc>4)throw std::runtime_error("Usage: replay_policy dump.json [iteration_budget] [default|early]");')
change('int budget=argc==3?std::stoi(argv[2]):data.value("iteration_budget",4096);',
       'int budget=argc>=3?std::stoi(argv[2]):data.value("iteration_budget",4096);\n'
       '  const std::string schedule=argc==4?argv[3]:"default";\n'
       '  if(schedule!="default"&&schedule!="early")throw std::runtime_error("Unknown recovery schedule");\n'
       '  const bool early=schedule=="early";')
change('  CoulombStats stats;\n  bool solver_ok=coulombSolve(A,b,p,lo,hi,dependencies,budget,tolerance,stats);',
       '''  // Independent diagnostic call from the EXACT input seed. This work is
  // outside the scheduling solve, so this harness is NOT a timing benchmark.
  const auto original=p;auto first_p=p;CoulombStats first;
  std::vector<double> first_rejected;
  const bool first_accepted=coulombIterate(A,b,first_p,lo,hi,dependencies,std::min(budget,256),tolerance,first,&first_rejected);
  CoulombStats stats;p=original;bool solver_ok=false;
#ifdef EARLY_POLICY_CANDIDATE
  solver_ok=coulombSolve(A,b,p,lo,hi,dependencies,budget,tolerance,stats,nullptr,true,early);
#else
  if(early)throw std::runtime_error("Frozen replay has no early policy");
  solver_ok=coulombSolve(A,b,p,lo,hi,dependencies,budget,tolerance,stats);
#endif''')
change("  std::cout<<output.dump(2)<<'\\n';return solver_ok&&law_ok&&energy_ok?0:2;", '''  std::vector<double> initial(n);for(int i=0;i<n;i++)initial[i]=original[i];
  output["initial_p"]=initial;output["schedule"]=schedule;
  output["first256_accepted"]=first_accepted;
  output["first256_rejected_p"]=first_rejected;
  output["first256_sweeps"]=first.iteration_sweeps_total;
  output["scope"]="Prepared schedule replay; historical capture seed, no world trajectory and no timing comparison";
#ifdef EARLY_POLICY_CANDIDATE
  output["early_component_receipt"]={{"attempts",stats.early_component_attempts},
   {"solves",stats.early_component_solves},{"declines",stats.early_component_declines},
   {"helper_calls",stats.early_component_helper_calls},{"passes",stats.early_component_passes},
   {"component_cap_rejections",stats.early_component_cap_rejections},
   {"svd_calls",stats.early_component_svd_calls},{"iteration_steps",stats.early_component_iteration_steps},
   {"pressure_svd_calls",stats.early_component_pressure_svd_calls},
   {"pressure_attempts",stats.early_component_pressure_attempts},
   {"pivot_calls",stats.early_component_pivot_calls}};
  if(!solver_ok){for(int i=0;i<n;i++)if(p[i]!=original[i])throw std::runtime_error("Rejected complete solve changed x");}
  if(stats.early_component_attempts>0&&(first_accepted||first_rejected.size()!=size_t(n)))
   throw std::runtime_error("Early attempt lacks the matched first256 rejection");
  if(stats.early_component_attempts!=stats.early_component_solves+stats.early_component_declines)
   throw std::runtime_error("Early counter receipt mismatch");
  if(stats.early_component_helper_calls>8||stats.early_component_passes>8||
     stats.early_component_svd_calls>1024||stats.early_component_pressure_svd_calls>1024||
     stats.early_component_pivot_calls>8)
   throw std::runtime_error("Fresh early helper call exceeded its original cap");
  const int calls=stats.early_component_attempts?2:1;
  if(stats.support_helper_calls>8*calls||stats.support_passes>8*calls||
     stats.support_svd_calls>1024*calls||stats.support_pressure_svd_calls>1024*calls||
     stats.support_pivot_calls>8*calls)
   throw std::runtime_error("Aggregate early/tail receipt exceeds disclosed two-call caps");
#endif
  std::cout<<output.dump(2)<<'\\n';return solver_ok&&law_ok&&energy_ok?0:2;''')
(root / "replay_policy.cpp").write_text(text)
print("Prepared replay_policy.cpp; no compiler or native execution.")
