# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

# See free_evi.pyx for details.

import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t, SIZE_t
from ._utils cimport IntVectorStruct, DoubleVectorStruct

cdef class EVI_FSUCRLv1:
    cdef DTYPE_t *u1
    cdef DTYPE_t *u2
    cdef DTYPE_t *mtx_maxprob
    cdef DTYPE_t[:,:] mtx_maxprob_memview
    cdef SIZE_t *sorted_indices
    cdef SIZE_t *sorted_indices_mu
    cdef SIZE_t nb_states
    cdef SIZE_t threshold
    cdef SIZE_t nb_options
    cdef SIZE_t max_reachable_states_per_opt
    cdef IntVectorStruct* actions_per_state
    cdef IntVectorStruct* reachable_states_per_option
    # cdef DoubleVectorStruct* x
    cdef DoubleVectorStruct* mu_opt
    cdef DTYPE_t *cn_opt
    cdef DoubleVectorStruct* r_tilde_opt
    cdef SIZE_t* options_policies
    cdef SIZE_t* options_policies_indices_mdp
    cdef DTYPE_t* options_terminating_conditions
    cdef DTYPE_t* xx
    cdef DTYPE_t* beta_mu_p

    cdef SIZE_t bernstein_bound

    # Methods
    cpdef DTYPE_t run(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     DTYPE_t[:,:,:] p_hat,
                     DTYPE_t[:,:] r_hat_mdp,
                     DTYPE_t[:,:,:] beta_p,
                     DTYPE_t[:,:] beta_r_mdp,
                     DTYPE_t r_max,
                     DTYPE_t epsilon)

    # def compute_mu_and_info(self, environment,
    #                       DTYPE_t[:,:,:] estimated_probabilities_mdp,
    #                       DTYPE_t[:,:] estimated_rewards_mdp,
    #                       DTYPE_t[:,:] beta_r)

    cpdef compute_mu_info(self,
                          DTYPE_t[:,:,:] estimated_probabilities_mdp,
                          DTYPE_t[:,:] estimated_rewards_mdp,
                          DTYPE_t[:,:] beta_r,
                          SIZE_t[:,:] nb_observations_mdp,
                          DTYPE_t range_mu_p,
                          SIZE_t total_time,
                          DTYPE_t delta,
                          SIZE_t max_nb_actions)

    cpdef get_uvectors(self)
    cpdef get_conditioning_numbers(self)
    cpdef get_beta_mu_p(self)
    cpdef get_mu(self)
    cpdef get_r_tilde_opt(self)


cdef class EVI_FSUCRLv2:
    # --------------------------------------------------------------------------
    # Matrix-like structures required by the algorithm
    # --------------------------------------------------------------------------
    cdef DTYPE_t *u1
    cdef DTYPE_t *u2
    cdef DoubleVectorStruct *w1
    cdef DoubleVectorStruct *w2

    cdef SIZE_t *sorted_indices
    cdef DTYPE_t *mtx_maxprob
    cdef DTYPE_t[:,:] mtx_maxprob_memview

    cdef DTYPE_t *max_opt_p # (|O|x|S|)

    # --------------------------------------------------------------------------
    # Model info
    # --------------------------------------------------------------------------
    cdef IntVectorStruct* actions_per_state

    # --------------------------------------------------------------------------
    # Options info
    # --------------------------------------------------------------------------
    cdef IntVectorStruct* reachable_states_per_option
    cdef SIZE_t* options_policies
    cdef SIZE_t* options_policies_indices_mdp # index of an action executed by
                                              # an option wrt the original mdp
    cdef DTYPE_t* options_terminating_conditions

    cdef DoubleVectorStruct* r_tilde_opt

    # --------------------------------------------------------------------------
    # Scalar info
    # --------------------------------------------------------------------------
    cdef SIZE_t nb_states
    cdef SIZE_t nb_options
    cdef SIZE_t threshold

    cdef SIZE_t bernstein_bound

    cdef SIZE_t max_reachable_states_per_opt


