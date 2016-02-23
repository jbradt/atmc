from libcpp.vector cimport vector as cppvec
cimport armadillo as arma

cdef extern from "mcopt/mcopt.h" namespace "mcopt":

    cdef cppclass Track:
        Track()
        arma.mat getMatrix()
        arma.mat getPositionMatrix()
        arma.vec getEnergyVector()
        size_t numPts()

    cdef cppclass Tracker:
        Tracker(unsigned massNum, unsigned chargeNum, cppvec[double]& eloss,
                arma.vec& efield, arma.vec& bfield)
        Track trackParticle(const double x0, const double y0, const double z0,
                            const double enu0,  const double azi0, const double pol0)

    # cdef cppclass PadPlane:
    #     PadPlane()
