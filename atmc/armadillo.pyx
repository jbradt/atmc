cimport armadillo
import numpy as np
cimport numpy as np

cdef mat * np2mat(np.ndarray[np.double_t, ndim=2] arr):
    cdef int i
    cdef int j
    cdef int dim0 = arr.shape[0]
    cdef int dim1 = arr.shape[1]

    cdef mat* armaArr = new mat(dim0, dim1)
    cdef double* armaMem = armaArr.memptr()

    for i in range(dim0):
        for j in range(dim1):
            armaMem[i + j*dim0] = arr[i, j]

    return armaArr

cdef Mat[unsigned short] * np2uint16mat(np.ndarray[np.uint16_t, ndim=2] arr):
    cdef int i
    cdef int j
    cdef int dim0 = arr.shape[0]
    cdef int dim1 = arr.shape[1]

    cdef Mat[unsigned short]* armaArr = new Mat[unsigned short](dim0, dim1)
    cdef unsigned short* armaMem = armaArr.memptr()

    for i in range(dim0):
        for j in range(dim1):
            armaMem[i + j*dim0] = arr[i, j]

    return armaArr

cdef vec * np2vec(np.ndarray[np.double_t, ndim=1] arr):
    cdef int i
    cdef int dim = arr.shape[0]

    cdef vec* armaArr = new vec(dim)
    cdef double* armaMem = armaArr.memptr()

    for i in range(dim):
        armaMem[i] = arr[i]

    return armaArr

cdef np.ndarray[np.double_t, ndim=2] mat2np(const mat & armaArr):
    cdef np.ndarray[np.double_t, ndim=2] arr = np.empty((armaArr.n_rows, armaArr.n_cols), dtype=np.double, order='F')

    for i in range(armaArr.n_rows):
        for j in range(armaArr.n_cols):
            arr[i, j] = armaArr(i, j)

    return arr

cdef np.ndarray[np.double_t, ndim=1] vec2np(const vec & armaArr):
    cdef np.ndarray[np.double_t, ndim=1] arr = np.empty(armaArr.n_elem, dtype=np.double)

    for i in range(armaArr.n_elem):
        arr[i] = armaArr(i)

    return arr
