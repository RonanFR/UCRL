# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

# See evi.pyx for details.

import numpy as np
cimport numpy as np

ctypedef np.npy_float32 DTYPE_t          # Type of X
ctypedef np.npy_intp SIZE_t              # Type for indices and counters
ctypedef np.npy_int32 INT32_t            # Signed 32 bit integer
ctypedef np.npy_uint32 UINT32_t          # Unsigned 32 bit integer

cdef void max_proba_purec(DTYPE_t[:] p, SIZE_t* asc_sorted_indices,
                          DTYPE_t beta, DTYPE_t[:] new_p) nogil

cdef DTYPE_t dot_prod(DTYPE_t[:] x, DTYPE_t* y, SIZE_t dim) nogil

cdef class EVI:
    cdef DTYPE_t *u1
    cdef DTYPE_t *u2
    cdef DTYPE_t *mtx_maxprob
    cdef DTYPE_t[:,:] mtx_maxprob_memview
    cdef SIZE_t *sorted_indices
    cdef SIZE_t nb_states

    # Methods

    cpdef DTYPE_t evi(self, SIZE_t[:] policy_indices, SIZE_t[:] policy,
                     SIZE_t[:, :] state_actions,
                     SIZE_t[:] num_actions_per_state,
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