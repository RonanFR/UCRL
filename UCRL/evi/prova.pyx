from scipy.linalg.cython_lapack cimport dgeev, dgesv
from libc.stdio cimport printf
from libc.stdlib cimport malloc, free
from libc.math cimport fabs

#  /* Auxiliary routine: printing eigenvalues */
cdef void print_eigenvalues( char* desc, int n, double* wr, double* wi ):
    cdef int j;
    printf( "\n %s\n", desc );
    for j in range(n):
        if wi[j] == <double>0.0 :
            printf( " %6.2f", wr[j] );
        else:
            printf( " (%6.2f,%6.2f)", wr[j], wi[j] );
    printf( "\n" );

# /* Auxiliary routine: printing eigenvectors */
cdef void print_eigenvectors( char* desc, int n, double* wi, double* v, int ldv ):
    cdef int i, j;
    printf( "\n %s\n", desc );
    for i in range(n):
        j = 0;
        while j < n:
            if  wi[j] == <double>0.0:
                printf( " %6.2f", v[i+j*ldv] );
                j+=1;
            else:
                printf( " (%6.2f,%6.2f)", v[i+j*ldv], v[i+(j+1)*ldv] );
                printf( " (%6.2f,%6.2f)", v[i+j*ldv], -v[i+(j+1)*ldv] );
                j += 2;
        printf( "\n" );

# /* Auxiliary routine: printing a matrix */
cdef void print_matrix( char* desc, int m, int n, double* a, int lda ):
    cdef int i, j;
    printf( "\n %s\n", desc );
    for i in range(m):
        for j in range(n):
            printf( " %6.3f", a[i+j*lda] );
        printf( "\n" );

# /* Auxiliary routine: printing a vector of integers */
cdef void print_int_vector( char* desc, int n, int* a ):
    cdef int j;
    printf( "\n %s\n", desc );
    for j in range(n):
        printf( " %6i", a[j] );
    printf( "\n" );

cpdef cython_cn(double[:,:] mtx, int N):

    # Note that lapack works column-major while c-style matrices are
    # row-major. We need to transpose the matrix

    # /* Locals */
    cdef int n = N, lda = N, ldvl = N, ldvr = N, info, lwork
    cdef int nrhs = N, ldb = N
    cpdef int i,j, x
    cdef double max_eigval, tmp, val, condition_nb

    cdef double wkopt;
    cdef double* work;
    # /* Local arrays */
    cdef double *wr
    cdef double *wi
    cdef double *vl
    cdef double *vr
    cdef double *a
    cdef double *mu
    cdef double *P_star
    cdef double *A # inverse of the fundamental matrix
    cdef double *B
    cdef int *ipiv
    # a = &mtx[0,0]

    a = <double*>malloc(n * n * sizeof(double))
    A = <double*>malloc(n * n * sizeof(double))
    B = <double*>malloc(n * n * sizeof(double))
    P_star = <double*>malloc(n * n * sizeof(double))
    for i in range(N):
        for j in range(N):
            a[i + j*n] = mtx[i,j]
    print_matrix("A", n, n, a, n)

    wr = <double*>malloc(n * sizeof(double))
    wi = <double*>malloc(n * sizeof(double))
    vl = <double*>malloc(n * ldvl * sizeof(double))
    vr = <double*>malloc(n * ldvr * sizeof(double))
    mu = <double*>malloc(n *sizeof(double))
    ipiv = <int*>malloc(n * n * sizeof(int))
    # /* Query and allocate the optimal workspace */
    lwork = -1;
    dgeev( "Vectors", "Vectors", &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr,
     &wkopt, &lwork, &info );
    lwork = <int>wkopt;
    work = <double*>malloc( lwork*sizeof(double) );
    # /* Solve eigenproblem */
    dgeev( "Vectors", "Vectors", &n, a, &lda, wr, wi, vl, &ldvl, vr, &ldvr,
     work, &lwork, &info );
    # /* Check for convergence */
    if info > 0:
        printf( "The algorithm failed to compute eigenvalues.\n" );
        return 1;
    # /* Print eigenvalues */
    print_eigenvalues( "Eigenvalues", n, wr, wi );
    # /* Print left eigenvectors */
    print_eigenvectors( "Left eigenvectors", n, wi, vl, ldvl );
    # /* Print right eigenvectors */
    print_eigenvectors( "Right eigenvectors", n, wi, vr, ldvr );

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
    print_matrix( "P_star", n, n, P_star, n );

    for i in range(n):
        for j in range(n):
            idx = i + j*n
            A[idx] = - mtx[i,j] + P_star[idx]
            B[idx] = - P_star[idx]
            if i == j:
                A[idx] += 1.
                B[idx] += 1.

    print_matrix( "A", n, n, A, n );
    print_matrix( "B", n, n, B, n );

    dgesv( &n, &nrhs, A, &lda, ipiv, B, &ldb, &info )
    # /* Check for the exact singularity */
    if info > 0:
        printf( "The diagonal element of the triangular factor of A,\n" );
        printf( "U(%i,%i) is zero, so that A is singular;\n", info, info );
        printf( "the solution could not be computed.\n" );
        return 1
    # /* Print solution */
    print_matrix( "Solution", n, nrhs, B, ldb );
    # /* Print details of LU factorization */
    print_matrix( "Details of LU factorization", n, n, A, lda );
    # /* Print pivot indices */
    print_int_vector( "Pivot indices", n, ipiv );


    condition_nb = 0  # condition number of deviation matrix
    for i in range(0, n):  # Seneta's condition number
        for j in range(i + 1, n):
            # compute L1-norm
            tmp = 0.0
            for x in range(n):
                tmp += fabs(B[i+x*n] - B[j+x*n])
            # finally the conditioning number
            condition_nb = max(condition_nb, 0.5 * tmp)
    printf("condition number: %f\n", condition_nb)

    # /* Free workspace */
    free(work)
    free(mu)
    free(wi)
    free(wr)
    free(vl)
    free(vr)
    free(a)
    free(P_star)
    free(A)
    free(B)
    free(ipiv)
    return 0