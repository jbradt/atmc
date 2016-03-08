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
            del efieldVec, bfieldVec

    def __dealloc__(self):
        del self.thisptr

    def track_particle(self, double x0, double y0, double z0, double enu0, double azi0, double pol0):
        cdef Track tr = self.thisptr.trackParticle(x0, y0, z0, enu0, azi0, pol0)
        cdef arma.mat trmat = tr.getMatrix()
        return arma.mat2np(trmat)

cdef class PyPadPlane:
    cdef PadPlane *thisptr
    def __cinit__(self, np.ndarray[np.uint16_t, ndim=2] lut, double xLB, double xDelta,
                  double yLB, double yDelta, double rotAngle=0):
        cdef arma.Mat[unsigned short] *lutMat
        try:
            lutMat = arma.np2uint16mat(lut)
            self.thisptr = new PadPlane(deref(lutMat), xLB, xDelta, yLB, yDelta, rotAngle)
        finally:
            del lutMat

    def __dealloc__(self):
        del self.thisptr

    def get_pad_number_from_coordinates(self, double x, double y):
        return self.thisptr.getPadNumberFromCoordinates(x, y)

cdef class PyEventGenerator:
    cdef EventGenerator *thisptr

    def __cinit__(self, PyPadPlane pads, np.ndarray[np.double_t, ndim=1] vd, double clock, double shape,
                  unsigned massNum, double ioniz, double gain, double tilt,
                  np.ndarray[np.double_t, ndim=1] beamCtr = np.zeros(3, dtype=np.double)):
        cdef arma.vec *vdVec
        cdef arma.vec *beamCtrVec

        try:
            vdVec = arma.np2vec(vd)
            beamCtrVec = arma.np2vec(beamCtr)
            self.thisptr = new EventGenerator(deref(pads.thisptr), deref(vdVec), clock, shape, massNum, ioniz, gain,
                                              tilt, deref(beamCtrVec))
        finally:
            del vdVec, beamCtrVec

    def __dealloc__(self):
        del self.thisptr

    def make_peaks(self, np.ndarray[np.double_t, ndim=2] pos, np.ndarray[np.double_t, ndim=1] en):
        cdef arma.mat *posMat
        cdef arma.vec *enVec
        cdef arma.mat peaks
        cdef np.ndarray[np.double_t, ndim=2] res
        try:
            posMat = arma.np2mat(pos)
            enVec = arma.np2vec(en)
            peaks = self.thisptr.makePeaksTableFromSimulation(deref(posMat), deref(enVec))
            res = arma.mat2np(peaks)
        finally:
            del posMat, enVec

        return res

cdef class PyMCminimizer:
    cdef MCminimizer *thisptr

    def __cinit__(self, PyTracker tr, PyEventGenerator evtgen):
        self.thisptr = new MCminimizer(deref(tr.thisptr), deref(evtgen.thisptr))

    def __dealloc__(self):
        del self.thisptr

    def minimize(self, np.ndarray[np.double_t, ndim=1] ctr0, np.ndarray[np.double_t, ndim=1] sigma0,
                 np.ndarray[np.double_t, ndim=2] expPos, np.ndarray[np.double_t, ndim=1] expMesh,
                 unsigned numIters=10, unsigned numPts=200, double redFactor=0.8):
        cdef arma.vec *ctr0Arr
        cdef arma.vec *sigma0Arr
        cdef arma.vec *expMeshArr
        cdef arma.mat *expPosArr
        cdef MCminimizeResult minres

        try:
            ctr0Arr = arma.np2vec(ctr0)
            sigma0Arr = arma.np2vec(sigma0)
            expPosArr = arma.np2mat(expPos)
            expMeshArr = arma.np2vec(expMesh)
            minres = self.thisptr.minimize(deref(ctr0Arr), deref(sigma0Arr), deref(expPosArr), deref(expMeshArr),
                                           numIters, numPts, redFactor)
        finally:
            del ctr0Arr, sigma0Arr, expPosArr, expMeshArr

        cdef np.ndarray[np.double_t, ndim=1] ctr = arma.vec2np(minres.ctr)
        cdef np.ndarray[np.double_t, ndim=2] allParams = arma.mat2np(minres.allParams)
        cdef np.ndarray[np.double_t, ndim=1] minPosChis = arma.vec2np(minres.minPosChis)
        cdef np.ndarray[np.double_t, ndim=1] minEnChis = arma.vec2np(minres.minEnChis)
        cdef np.ndarray[np.double_t, ndim=1] goodParamIdx = arma.vec2np(minres.goodParamIdx)

        return ctr, allParams, minPosChis, minEnChis, goodParamIdx
