# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

# See free_evi.pyx for details.

import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t, SIZE_t
from ._utils cimport IntVectorStruct, DoubleVectorStruct

cdef class FreeEVIAlg1:
    cdef DTYPE_t *u1
    cdef DTYPE_t *u2
    cdef DTYPE_t *mtx_maxprob
    cdef DTYPE_t[:,:] mtx_maxprob_memview
    cdef SIZE_t *sorted_indices
    cdef SIZE_t nb_states
    cdef SIZE_t threshold
    cdef SIZE_t nb_options
    cdef IntVectorStruct* actions_per_state
    cdef IntVectorStruct* reachable_states_per_option
    cdef DoubleVectorStruct* x
    cdef DoubleVectorStruct* mu_opt
    cdef DTYPE_t *cn_opt
    cdef DTYPE_t *beta_mu_p
    cdef DoubleVectorStruct* r_tilde_opt

    # Methods
    cpdef DTYPE_t run(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     DTYPE_t[:,:,:] p_hat,
                     DTYPE_t[:,:] r_hat_mdp,
                     DTYPE_t[:,:] beta_p,
                     DTYPE_t[:,:] beta_r_mdp,
                     DTYPE_t[:] beta_mu_p,
                     DTYPE_t r_max,
                     DTYPE_t epsilon)

    # def compute_mu_and_info(self, environment,
    #                       DTYPE_t[:,:,:] estimated_probabilities_mdp,
    #                       DTYPE_t[:,:] estimated_rewards_mdp,
    #                       DTYPE_t[:,:] beta_r)

    cpdef get_uvectors(self)
