from scipy.linalg.cython_lapack cimport dgeev, dgesv
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.math cimport fabs

import numpy as np
cimport numpy as np

#  /* Auxiliary routine: printing eigenvalues */
cdef void print_eigenvalues( char* desc, SIZE_t n, DTYPE_t* wr, DTYPE_t* wi ) nogil:
    cdef SIZE_t j;
    printf( "\n %s\n", desc )
    for j in range(n):
        if wi[j] == <double>0.0 :
            printf( " %6.2f", wr[j] )
        else:
            printf( " (%6.2f,%6.2f)", wr[j], wi[j] )
    printf( "\n" )

# /* Auxiliary routine: printing eigenvectors */
cdef void print_eigenvectors( char* desc, SIZE_t n, DTYPE_t* wi, DTYPE_t* v, SIZE_t ldv ) nogil:
    cdef SIZE_t i, j;
    printf( "\n %s\n", desc )
    for i in range(n):
        j = 0
        while j < n:
            if  wi[j] == <double>0.0:
                printf( " %6.2f", v[i+j*ldv] )
                j+=1
            else:
                printf( " (%6.2f,%6.2f)", v[i+j*ldv], v[i+(j+1)*ldv] )
                printf( " (%6.2f,%6.2f)", v[i+j*ldv], -v[i+(j+1)*ldv] )
                j += 2
        printf( "\n" )

# /* Auxiliary routine: printing a matrix */
cdef void print_matrix( SIZE_t ordering, char* desc, SIZE_t m, SIZE_t n, DTYPE_t* a, SIZE_t lda ) nogil:
    cdef SIZE_t i, j, idx;
    printf( "\n %s\n", desc )
    for i in range(m):
        for j in range(n):
            if ordering == 0:
                # row-major
                idx = i*lda +j
            else:
                # column-major
                idx = i+j*lda
            printf( " %6.3f", a[idx] )
        printf( "\n" )

# /* Auxiliary routine: printing a vector of integers */
cdef void print_int_vector( char* desc, SIZE_t n, SIZE_t* a ) nogil:
    cdef SIZE_t j;
    printf( "\n %s\n", desc )
    for j in range(n):
        printf( " %6i", a[j] )
    printf( "\n" )

cdef void print_double_vector( char* desc, SIZE_t n, DTYPE_t* a ) nogil:
    cdef int j;
    printf( "\n %s\n", desc )
    for j in range(n):
        printf( " %6.3f", a[j] )
    printf( "\n" )

# =============================================================================
# Compute MU and Condition Number using LAPACK
# =============================================================================
# Note that LAPACK interface uses FORTRAN ordering for matrices (column-major)

cdef SIZE_t get_mu_and_ci_c(DTYPE_t* sq_rowmaj_mtx, SIZE_t N,
                          DTYPE_t* condition_number, DTYPE_t* mu) nogil:

    # Note that lapack works column-major while c-style matrices are
    # row-major. We need to transpose the matrix

    # /* Locals */
    cdef int n = N, lda = N, ldvl = N, ldvr = N, info, lwork
    cdef int nrhs = N, ldb = N
    cpdef int i,j, x
    cdef double max_eigval, tmp, val

    cdef double wkopt;
    cdef double* work;
    # /* Local arrays */
    cdef double *wr
    cdef double *wi
    cdef double *vl
    cdef double *vr
    cdef double *P
    cdef double *P_star
    cdef double *A # inverse of the fundamental matrix
    cdef double *B
    cdef int *ipiv
    # a = &mtx[0,0]

    P = <double*>malloc(n * n * sizeof(double))
    A = <double*>malloc(n * n * sizeof(double))
    B = <double*>malloc(n * n * sizeof(double))
    P_star = <double*>malloc(n * n * sizeof(double))

    for i in range(N):
        for j in range(N):
            P[i + j * n] = sq_rowmaj_mtx[i * n + j]
    print_matrix(1, "P (column-major)", n, n, P, n)

    wr = <double*>malloc(n * sizeof(double))
    wi = <double*>malloc(n * sizeof(double))
    vl = <double*>malloc(n * ldvl * sizeof(double))
    vr = <double*>malloc(n * ldvr * sizeof(double))
    mu = <double*>malloc(n *sizeof(double))
    ipiv = <int*>malloc(n * n * sizeof(int))
    # /* Query and allocate the optimal workspace */
    lwork = -1
    dgeev( "Vectors", "Vectors", &n, P, &lda, wr, wi, vl, &ldvl, vr, &ldvr,
     &wkopt, &lwork, &info )
    lwork = <int>wkopt
    work = <double*>malloc( lwork*sizeof(double) )
    # /* Solve eigenproblem */
    dgeev( "Vectors", "Vectors", &n, P, &lda, wr, wi, vl, &ldvl, vr, &ldvr,
     work, &lwork, &info )
    # /* Check for convergence */
    if info > 0:
        printf( "The algorithm failed to compute eigenvalues.\n" )
        return 1
    # /* Print eigenvalues */
    print_eigenvalues( "Eigenvalues", n, wr, wi )
    # /* Print left eigenvectors */
    print_eigenvectors( "Left eigenvectors", n, wi, vl, ldvl )
    # /* Print right eigenvectors */
    print_eigenvectors( "Right eigenvectors", n, wi, vr, ldvr )

    # get max real eigenvalue
    j = 0
    x = 0
    max_eigval = wr[0]
    while j < n:
        if max_eigval < wr[j]:
            max_eigval = wr[j]
            x = j
        if wi[j] == <double>0.0:
            j += 1
        else:
            j+=2
    printf("\n max eig: %d (%f)", x, max_eigval)

    # mu -> real part of the max eigen value
    for j in range(n): #col
        mu[j] = vr[j + x * ldvr]
        for i in range(n):
            P_star[i + j * n] = mu[j]
    for i in range(n):
        printf( " %6.3f", mu[i])
    printf("\n")
    print_matrix(1, "P_star", n, n, P_star, n )

    for i in range(n):
        for j in range(n):
            idx = i + j*n
            A[idx] = - sq_rowmaj_mtx[i * n + j] + P_star[idx]
            B[idx] = - P_star[idx]
            if i == j:
                A[idx] += 1.
                B[idx] += 1.

    print_matrix(1, "A", n, n, A, n )
    print_matrix(1, "B", n, n, B, n )

    dgesv( &n, &nrhs, A, &lda, ipiv, B, &ldb, &info )
    # /* Check for the exact singularity */
    if info > 0:
        printf( "The diagonal element of the triangular factor of A,\n" )
        printf( "U(%i,%i) is zero, so that A is singular;\n", info, info )
        printf( "the solution could not be computed.\n" )
        return 1
    # /* Print solution */
    print_matrix(1, "Solution", n, nrhs, B, ldb )
    # /* Print details of LU factorization */
    print_matrix(1, "Details of LU factorization", n, n, A, lda )
    # /* Print pivot indices */
    print_int_vector( "Pivot indices", n, ipiv )


    condition_number[0] = -1.0  # condition number of deviation matrix
    for i in range(0, n):  # Seneta's condition number
        for j in range(i + 1, n):
            # compute L1-norm
            tmp = 0.0
            for x in range(n):
                tmp += fabs(B[i+x*n] - B[j+x*n])
            # finally the conditioning number
            condition_number[0] = max(condition_number[0], 0.5 * tmp)
    printf("condition number: %f\n", condition_number[0])

    # /* Free workspace */
    free(work)
    free(wr)
    free(wi)
    free(vl)
    free(vr)
    free(P)
    free(P_star)
    free(A)
    free(B)
    free(ipiv)
    return 0



def get_mu_and_ci(P):
    cdef DTYPE_t ci
    cdef DTYPE_t* mu_ptr
    cdef SIZE_t n = P.shape[0]
    mu = np.empty((n,), dtype=np.float64)
    mu_ptr = <DTYPE_t*> np.PyArray_DATA(mu)
    get_mu_and_ci_c(mu_ptr, n, &ci, mu_ptr)
    return mu, ci