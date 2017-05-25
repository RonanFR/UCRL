# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

# See free_evi.pyx for details.

import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t, SIZE_t
from ._utils cimport IntVectorStruct, DoubleVectorStruct


cdef DTYPE_t check_end_opt_evi(DTYPE_t* x, DTYPE_t* y, SIZE_t dim,
                               DTYPE_t* add_res) nogil

cdef class EVI_FSUCRLv1:
    cdef DTYPE_t *u1
    cdef DTYPE_t *u2
    cdef DTYPE_t *mtx_maxprob
    cdef DTYPE_t[:,:] mtx_maxprob_memview
    cdef SIZE_t *sorted_indices                        # (|S| x 1)
    cdef SIZE_t *sorted_indices_mu                     # (|S| x nb reach states)
    cdef SIZE_t nb_states
    cdef SIZE_t threshold
    cdef SIZE_t nb_options
    cdef SIZE_t max_reachable_states_per_opt
    cdef IntVectorStruct* actions_per_state
    cdef IntVectorStruct* reachable_states_per_option
    # cdef DoubleVectorStruct* x
    cdef DoubleVectorStruct* mu_opt
    cdef DTYPE_t *cn_opt                               # (|O| x 1)
    cdef DoubleVectorStruct* r_tilde_opt               # (|O| x nb reach states)
    cdef SIZE_t* options_policies
    cdef SIZE_t* options_policies_indices_mdp
    cdef DTYPE_t* options_terminating_conditions
    cdef DTYPE_t* xx                                   # (|S| x nb reach states)
    cdef DTYPE_t* beta_mu_p                            # (|O| x 1)

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
                          SIZE_t max_nb_actions,
                          DTYPE_t r_max)

    cpdef compute_mu_info2(self,
                      DTYPE_t[:,:,:] estimated_probabilities_mdp,
                      DTYPE_t[:,:] estimated_rewards_mdp,
                      DTYPE_t[:,:] beta_r,
                      SIZE_t[:,:] nb_observations_mdp,
                      DTYPE_t range_mu_p,
                      SIZE_t total_time,
                      DTYPE_t delta,
                      SIZE_t max_nb_actions,
                           DTYPE_t r_max)

    cpdef get_uvectors(self)
    cpdef get_conditioning_numbers(self)
    cpdef get_beta_mu_p(self)
    cpdef get_mu(self)
    cpdef get_r_tilde_opt(self)


cdef class EVI_FSUCRLv2:
    # --------------------------------------------------------------------------
    # Matrix-like structures required by the algorithm
    # --------------------------------------------------------------------------
    cdef DTYPE_t *u1             # (|S|)
    cdef DTYPE_t *u2             # (|S|)
    cdef DoubleVectorStruct *w1  # (|O|x nb reach states)
    cdef DoubleVectorStruct *w2  # (|O|x nb reach states)
    cdef DTYPE_t* span_w_opt     # (|O|)

    cdef SIZE_t *sorted_indices  # (|S|)
    cdef DTYPE_t *mtx_maxprob    # (|S| x |S|)
    cdef DTYPE_t[:,:] mtx_maxprob_memview


    cdef DTYPE_t *mtx_maxprob_opt # (|O| x |S|)
    cdef DTYPE_t[:,:] mtx_maxprob_opt_memview

    # --------------------------------------------------------------------------
    # Model info
    # --------------------------------------------------------------------------
    cdef IntVectorStruct* actions_per_state

    # --------------------------------------------------------------------------
    # Options info
    # --------------------------------------------------------------------------
    cdef IntVectorStruct* reachable_states_per_option # (|O| x nb reach states)
    cdef SIZE_t* options_policies # (|O| x |S|)
    cdef SIZE_t* options_policies_indices_mdp # index of an action executed by
                                              # an option wrt the original mdp
    cdef DTYPE_t* options_terminating_conditions # (|O| x |S|)

    cdef DoubleVectorStruct* r_tilde_opt #(|O|x nb reach states)
    cdef DoubleVectorStruct* beta_opt_p  #(|O|x (nb reach states)^2 )
    cdef DoubleVectorStruct* p_hat_opt   #(|O|x (nb reach states)^2 )

    cdef IntVectorStruct* sorted_indices_popt #(|O|x nb reach states)

    # --------------------------------------------------------------------------
    # Scalar info
    # --------------------------------------------------------------------------
    cdef SIZE_t nb_states
    cdef SIZE_t nb_options
    cdef SIZE_t threshold

    cdef SIZE_t bernstein_bound

    cdef SIZE_t max_reachable_states_per_opt


    # --------------------------------------------------------------------------
    # METHODS
    # --------------------------------------------------------------------------
    cpdef compute_prerun_info(self, DTYPE_t[:,:,:] estimated_probabilities_mdp,
                          DTYPE_t[:,:] estimated_rewards_mdp,
                          DTYPE_t[:,:] beta_r,
                          SIZE_t[:,:] nb_observations_mdp,
                          SIZE_t total_time,
                          DTYPE_t delta,
                          SIZE_t max_nb_actions,
                          DTYPE_t range_opt_p,
                              DTYPE_t r_max)

    cpdef DTYPE_t run(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     DTYPE_t[:,:,:] p_hat,
                     DTYPE_t[:,:] r_hat_mdp,
                     DTYPE_t[:,:,:] beta_p,
                     DTYPE_t[:,:] beta_r_mdp,
                     DTYPE_t r_max,
                     DTYPE_t epsilon)

    cpdef get_r_tilde_opt(self)
    cpdef get_opt_p_and_beta(self)

