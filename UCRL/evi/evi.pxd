# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

# See evi.pyx for details.

import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t, SIZE_t
from ._utils cimport IntVectorStruct

cdef class EVI:
    cdef DTYPE_t *u1
    cdef DTYPE_t *u2
    cdef DTYPE_t *mtx_maxprob
    cdef DTYPE_t[:,:] mtx_maxprob_memview
    cdef SIZE_t *sorted_indices
    cdef SIZE_t nb_states
    cdef IntVectorStruct* actions_per_state

    # Methods

    cpdef DTYPE_t evi(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     DTYPE_t[:,:,:] estimated_probabilities,
                     DTYPE_t[:,:] estimated_rewards,
                     DTYPE_t[:,:] estimated_holding_times,
                     DTYPE_t[:,:] beta_r,
                     DTYPE_t[:,:] beta_p,
                     DTYPE_t[:,:] beta_tau,
                     DTYPE_t tau_max,
                     DTYPE_t r_max,
                     DTYPE_t tau,
                     DTYPE_t tau_min,
                     DTYPE_t epsilon)

    cpdef get_uvectors(self)
