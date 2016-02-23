cimport armadillo
import numpy as np
cimport numpy as np

cdef mat * np2mat(np.ndarray[np.double_t, ndim=2] arr):
    if not (arr.flags.f_contiguous or arr.flags.owndata):
        arr = arr.copy(order='F')
    cdef double[:, :] arrView = arr
    cdef mat* armaArr = new mat(&arrView[0, 0], arr.shape[0], arr.shape[1], False, True)
    return armaArr

cdef Mat[unsigned short] * np2uint16mat(np.ndarray[np.uint16_t, ndim=2] arr):
    if not (arr.flags.f_contiguous or arr.flags.owndata):
        arr = arr.copy(order='F')
    cdef unsigned short[:, :] arrView = arr
    cdef Mat[unsigned short]* armaArr = new Mat[unsigned short](&arrView[0, 0], arr.shape[0], arr.shape[1], False, True)
    return armaArr


cdef vec * np2vec(np.ndarray[np.double_t, ndim=1] arr):
    if not (arr.flags.f_contiguous or arr.flags.owndata):
        arr = arr.copy()
    cdef double[:] arrView = arr
    cdef vec* armaVec = new vec(&arrView[0], arr.shape[0], False, True)
    return armaVec

cdef np.ndarray[np.double_t, ndim=2] mat2np(const mat & armaArr):
    arr = np.empty((armaArr.n_rows, armaArr.n_cols), dtype=np.double, order='F')

    cdef double[:, :] dstView = arr
    for i in range(armaArr.n_rows):
        for j in range(armaArr.n_cols):
            dstView[i, j] = armaArr(i, j)

    return arr

cdef np.ndarray[np.double_t, ndim=2] vec2np(const vec & armaArr):
    arr = np.empty(armaArr.n_elem, dtype=np.double)

    cdef const double* srcPtr = armaArr.memptr()
    cdef double[:] dstView = arr
    for i in range(armaArr.n_elem):
        dstView[i] = srcPtr[i]

    return arr
