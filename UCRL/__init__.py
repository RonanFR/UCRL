# 0.14.dev0:
#   - fixed sigma_r
#   - fixed conditioning number
#   - defined multiple SUCRL algs.
# 0.15.dev0
#   - fixed interface of EVI
#   - updated Bernstein bound
# 0.16.dev0 (Nov 18, 2017)
#   - fixed interface of EVI (.evi -> .run)
#   - added discount factor
#   - added recentering as function parameter
#   - added reset function to reset u1 and u2
#   - added option to perform relative value iteration
#   - added Bias Span Constrained EVI (same structure of EVI)
#   - added test accordingly to (Puterman, 1994) for Evi e SC-EVI
#   - SUCRL and FSUCRL are unchanged
#   - addet test of 2 states domain in article
# 0.18.dev0 (Nov 20, 2017)
#   - changed schema for tie breaking in EVI
#       * use random noise in [0, 1e-4]
# 0.19.dev0 (Nov 21, 2017)
#   - implemented operator N as special case of SCEVI (run parameter)
#   - added test for N as in article
#   - fixed computation of convex combination
#   - added check of policy in test_toy2d1
# 0.20.dev0 (Nov 22, 2017)
#   - fix selection of action in SCEVI
#       * when min and max actions have equal value, pick using noise
#   - added RandomState to EVI and SCEVI initialized every time run is called
#   - SCEVI has span_constraint and relative_vi as class attributes
#       * they can be overloaded through run inputs
#   - implemented Span Regularized UCRL
#   - change simulation of policy in UCRL to account for stochastic policies
#   - added reset function to environment to reset to clear state
#   - added test for UCRL and SC-UCRL using toy 3D domain when span_c = inf
__version__ = '0.20.dev0'