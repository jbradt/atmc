cimport mcopt_wrapper
cimport libcpp.vector as cppvec
cimport armadillo as arma
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref
from libc.stdio cimport printf

cdef cppvec[double] np2cppvec(np.ndarray[np.double_t, ndim=1] v):
    cdef cppvec[double] res
    cdef double[:] vView = v
    for i in range(v.shape[0]):
        res.push_back(v[i])
    return res

cdef class PyTracker:
    cdef Tracker *thisptr
    def __cinit__(self, int massNum, int chargeNum, np.ndarray[np.double_t, ndim=1] eloss,
                  np.ndarray[np.double_t, ndim=1] efield, np.ndarray[np.double_t, ndim=1] bfield):
        cdef cppvec[double] elossVec = np2cppvec(eloss)
        cdef arma.vec *efieldVec
        cdef arma.vec *bfieldVec
        try:
            efieldVec = arma.np2vec(efield)
            bfieldVec = arma.np2vec(bfield)
            self.thisptr = new Tracker(massNum, chargeNum, elossVec, deref(efieldVec), deref(bfieldVec))
        finally:
            # del efieldVec, bfieldVec
            pass

    def __dealloc__(self):
        del self.thisptr

    def track_particle(self, double x0, double y0, double z0, double enu0, double azi0, double pol0):
        cdef Track tr = self.thisptr.trackParticle(x0, y0, z0, enu0, azi0, pol0)
        cdef arma.mat trmat = tr.getMatrix()
        return arma.mat2np(trmat)
