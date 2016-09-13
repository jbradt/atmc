cimport mcopt
from libcpp.vector cimport vector as cppvec
from libcpp.map cimport map as cppmap
from libcpp.pair cimport pair as cpppair
cimport armadillo as arma
import numpy as np
cimport numpy as np
from cython.operator cimport dereference as deref, preincrement as preinc
from libc.stdio cimport printf


cdef cppvec[double] np2cppvec(np.ndarray[np.double_t, ndim=1] v)


cdef class Gas:
    cdef mcopt.Gas *thisptr


cdef class Tracker:
    cdef mcopt.Tracker *thisptr
    cdef Gas pyGas


cdef class PadPlane:
    cdef mcopt.PadPlane *thisptr


cdef class EventGenerator:
    cdef mcopt.EventGenerator *thisptr
    cdef PadPlane pyPadPlane


cdef class Minimizer:
    cdef mcopt.MCminimizer *thisptr
    cdef Tracker pyTracker
    cdef EventGenerator pyEvtGen


cdef class Annealer:
    cdef mcopt.Annealer *thisptr
    cdef Tracker pyTracker
    cdef EventGenerator pyEvtGen
