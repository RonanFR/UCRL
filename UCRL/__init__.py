# 0.14dev0:
#   - fixed sigma_r
#   - fixed conditioning number
#   - defined multiple SUCRL algs.
# 0.15dev0
#   - fixed interface of EVI
#   - updated Bernstein bound
# 0.16dev0
#   - fixed interface of EVI (.evi -> .run)
#   - added discount factor
#   - added recentering as function parameter
#   - added reset function to reset u1 and u2
#   - added option to perform relative value iteration
#   - added Bias Span Constrained EVI (same structure of EVI)
#   - added test accordingly to (Puterman, 1994) for Evi e SC-EVI
#   - SUCRL and FSUCRL are unchanged
#   - addet test of 2 states domain in article
# 0.18dev0
#   - changed schema for tie breaking in EVI
#       * use random noise in [0, 1e-4]
# 0.19dev0
#   - implemented operator N as special case of SCEVI (run parameter)
#   - added test for N as in article
#   - fixed computation of convex combination
#   - added check of policy in test_toy2d1
__version__ = '0.19.dev0'