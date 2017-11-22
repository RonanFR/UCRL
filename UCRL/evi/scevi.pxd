# Authors: Ronan Fruit <ronan.fruit@inria.fr>
#          Matteo Pirotta <matteo.pirotta@inria.fr>
#
# License: BSD 3 clause

# See scevi.pyx for details.

import numpy as np
cimport numpy as np

from ._utils cimport DTYPE_t, SIZE_t
from ._utils cimport IntVectorStruct

cdef enum BoundType:
    CHERNOFF=0
    CHERNOFF_STATEDIM=1
    BERNSTEIN=2

cdef class SpanConstrainedEVI:
    cdef DTYPE_t *u1
    cdef DTYPE_t *u2
    cdef DTYPE_t *mtx_maxprob
    cdef DTYPE_t[:,:] mtx_maxprob_memview
    cdef SIZE_t *sorted_indices
    cdef SIZE_t nb_states
    cdef SIZE_t max_macroactions_per_state
    cdef IntVectorStruct* actions_per_state
    cdef BoundType bound_type
    cdef SIZE_t random_state
    cdef DTYPE_t gamma
    cdef DTYPE_t span_constraint
    cdef SIZE_t relative_vi

    # Methods

    cpdef DTYPE_t run(self, SIZE_t[:,:] policy_indices, DTYPE_t[:,:] policy,
                     DTYPE_t[:,:,:] estimated_probabilities,
                     DTYPE_t[:,:] estimated_rewards,
                     DTYPE_t[:,:] estimated_holding_times,
                     DTYPE_t[:,:] beta_r,
                     DTYPE_t[:,:,:] beta_p,
                     DTYPE_t[:,:] beta_tau,
                     DTYPE_t tau_max,
                     DTYPE_t r_max,
                     DTYPE_t tau,
                     DTYPE_t tau_min,
                     DTYPE_t epsilon,
                     DTYPE_t span_constraint = *,
                     SIZE_t initial_recenter = *,
                     SIZE_t relative_vi = *,
                     str operator_type = *)

    cpdef get_uvectors(self)

    cpdef get_span_constraint(self)
