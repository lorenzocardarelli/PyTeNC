import numpy as np
cimport numpy as np
cimport cython


# =============================================================================
# Extract element-wise from the sparse matrix the elements of the dense matrix.
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def cython_extract(double[:, :] matrix_source not None, # the arrays are referenced to as memoryviews
                   double[:, :] matrix_target not None, # the arrays are referenced to as memoryviews
                   unsigned long[:] row_ind not None,    # the arrays are referenced to as memoryviews
                   unsigned long[:] col_ind not None,    # the arrays are referenced to as memoryviews
                   ):

    cdef np.intp_t i, j
    cdef np.intp_t rows = matrix_target.shape[0]
    cdef np.intp_t cols = matrix_target.shape[1]
    cdef unsigned long ri, cj

    for i in range(rows):
        ri = row_ind[i]
        for j in range(cols):
            cj = col_ind[j]
            matrix_target[i, j] = matrix_source[ri, cj]


# =============================================================================
# Place element-wise in the sparse matrix the elements of the dense matrix.
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def cython_extract_from_vector_to_matrix(double[:] vector_source not None,    # the arrays are referenced to as memoryviews
                                         double[:, :] matrix_target not None, # the arrays are referenced to as memoryviews
                                         unsigned long[:] indices not None,    # the arrays are referenced to as memoryviews
                                         ):

    cdef np.intp_t i, j, index
    cdef np.intp_t dummie = 0
    cdef np.intp_t rows = matrix_target.shape[0]
    cdef np.intp_t cols = matrix_target.shape[1]

    for i in range(rows):
        for j in range(cols):
            index = indices[dummie]
            matrix_target[i, j] = vector_source[index]
            dummie += 1


# =============================================================================
# Place element-wise in the sparse matrix the elements of the dense matrix.
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def cython_place_back(double[:, :] matrix_target not None,  # the arrays are referenced to as memoryviews
                      double[:, :] matrix_source not None,  # the arrays are referenced to as memoryviews
                      unsigned long[:] row_indices not None, # the arrays are referenced to as memoryviews
                      unsigned long[:] col_indices not None, # the arrays are referenced to as memoryviews
                      ):

    cdef np.intp_t i, j
    cdef np.intp_t rows = matrix_source.shape[0]
    cdef np.intp_t cols = matrix_source.shape[1]
    cdef unsigned long ri, cj

    for i in range(rows):
        ri = row_indices[i]
        for j in range(cols):
            cj = col_indices[j]
            matrix_target[ri, cj] = matrix_source[i, j]


# =============================================================================
# Place element-wise in the sparse vector the elements of the dense matrix.
# =============================================================================
@cython.boundscheck(False)
@cython.wraparound(False)
def cython_place_back_from_matrix_to_vector(double[:] vector_target not None,    # the arrays are referenced to as memoryviews
                                            double[:, :] matrix_source not None, # the arrays are referenced to as memoryviews
                                            unsigned long[:] indices not None,    # the arrays are referenced to as memoryviews
                                            ):

    cdef np.intp_t rows = matrix_source.shape[0]
    cdef np.intp_t cols = matrix_source.shape[1]
    cdef np.intp_t dummie = 0

    cdef np.intp_t i, j, index
    for i in range(rows):
        for j in range(cols):
            index = indices[dummie]
            vector_target[index] = matrix_source[i, j]
            dummie += 1


