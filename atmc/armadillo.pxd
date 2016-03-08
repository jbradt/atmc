cimport numpy as np

cdef extern from "armadillo" namespace "arma":

    cdef cppclass mat:
        mat(double * aux_mem, int n_rows, int n_cols, bint copy_aux_mem, bint strict) except +
        mat(double * aux_mem, int n_rows, int n_cols) except +
        mat(int n_rows, int n_cols) except +
        mat() except +
        int n_rows
        int n_cols
        double* memptr()
        double operator()(int, int)

    cdef cppclass Mat[T]:
        Mat(T* aux_mem, int n_rows, int n_cols, bint copy_aux_mem, bint strict) except +
        Mat(T* aux_mem, int n_rows, int n_cols) except +
        Mat(int n_rows, int n_cols) except +
        Mat() except +
        int n_rows
        int n_cols
        double* memptr()
        double operator()(int, int)

    cdef cppclass vec:
        vec(double * aux_mem, int n_elem, bint copy_aux_mem, bint strict) except +
        vec(double * aux_mem, int n_elem) except +
        vec(int n_elem) except +
        vec() except +
        int n_elem
        double* memptr()
        double operator()(int)

cdef mat * np2mat(np.ndarray[np.double_t, ndim=2] arr)
cdef Mat[unsigned short] * np2uint16mat(np.ndarray[np.uint16_t, ndim=2] arr)
cdef vec * np2vec(np.ndarray[np.double_t, ndim=1] arr)
cdef np.ndarray[np.double_t, ndim=2] mat2np(const mat & armaArr)
cdef np.ndarray[np.double_t, ndim=2] vec2np(const vec & armaArr)
