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
# 0.21.dev0 (Nov 25, 2017)
#   - fix SC-EVI
#       * minimum value for (s,a) is computed in a pessimistic way by
#         taking min{R} and min {P u1}
#   - optimized SC-EVI in order to compute the minimum value (s,a)
#     only when required
#       * every iteration for N and at convergence for T
#   - updated test after SC-EVI update
#   - updated max_proba in order to compute min{P u1} as max{P (-u1)}
#       * it simply scan the vector in reverse order
#   - added Dijkstra algorithm for the computation of the shortest path (and diameter)
#   - added test for diameter
# 0.22.dev0 (Nov 26, 2017)
#   - fix error in SC-EVI
#       * added truncation of reward to r_max in pessimistic value computation
#   - adde u3 and u3min as class attributes
#   - added test for EVI using 3S-domain
# 0.24.dev0 (Jan 26, 2018)
#   - renamed span constrained UCRL to SCAL
#   - added augmentation of the reward for SCAL
#   - fixed confidence intervals for bernstein inequality (not in the SMDP case)
#       * added computation of variance of the reward with Welford's method
#   - added Gaussian, Gamma, Beta and Exp reward distributions
#   - modified Toy1 in order to have stochastic reward
# 0.25.dev0 (Jan 30, 2018)
#   - fix error in scopt (scevi.pyx)
#       * checked with valgrind
#   - added river swim and updated navgrid and 4-rooms to use scal
# 0.26.dev0 (Jan 31, 2018)
#   - added value iteration approach to compute diameter
#   - added resource collection domain
# 0.27.dev0 (Feb 1, 2018)
#   - added exception in evi when number of iterations is too big
#     i.e., it > min(1M, ns*na*200)
# 0.28.dev0 (Feb 1, 2018)
#   - fix error in evi and scevi due to random ties breaking
# 0.30.dev0 (Feb 20, 2018)
#   - added Short-Term UCRL
#       * with short-term EVI
#   - redefined structure of UCRL in order to allow different stopping conditions
# 0.31.dev0 (Feb 21, 2018)
#   - fix error in STEVI and in the correspond pytest
# 0.32.dev0 (Feb 21, 2018)
#   - Added Posterior Sampling (PS)
# 0.34.dev0 (Feb 25, 2018)
#   - Added Optimistic Linear Programming (OLP)
# 0.35.dev0 (April 28, 2018)
#   - updated max_proba (l1-norm) to deal with subarray
#   - added quicksort and testing code
#   - added binary search for find element in sorted vector
#   - added Truncated Extend Value iteration and tests
# 0.36.dev0 (May 1, 2018)
#   - added truncated UCRL
# 0.37.dev0 (May 20, 2018)
#   - changed terminal condition of TUCRL
# 0.45.dev0 (Mar 28, 2019)
#   - big refactoring to match thesis
# 0.46.dev0 (Apr 22, 2019)
#   - added Continuous SCAL (SCCAL-PLUS)
# 0.55.dev0 (Apr 22, 2019)
#   - Modified MDPs (PuddleWorld ShipSteering and MountainCar)
__version__ = '0.58.dev0'
