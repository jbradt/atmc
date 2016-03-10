from libcpp.vector cimport vector as cppvec
from libcpp.map cimport map as cppmap
cimport armadillo as arma

cdef extern from "mcopt/mcopt.h" namespace "mcopt":

    ctypedef unsigned short pad_t

    cdef cppclass Track:
        Track() except+
        arma.mat getMatrix() except+
        arma.mat getPositionMatrix() except+
        arma.vec getEnergyVector() except+
        size_t numPts() except+

    cdef cppclass Tracker:
        Tracker(unsigned massNum, unsigned chargeNum, cppvec[double]& eloss,
                arma.vec& efield, arma.vec& bfield) except+
        Track trackParticle(const double x0, const double y0, const double z0,
                            const double enu0,  const double azi0, const double pol0) except+

    cdef cppclass PadPlane:
        PadPlane(const arma.Mat[pad_t]& lt, const double xLB, const double xDelta,
                 const double yLB, const double yDelta, const double rotAngle) except+
        pad_t getPadNumberFromCoordinates(const double x, const double y) except+

    cdef cppclass EventGenerator:
        EventGenerator(const PadPlane& pads, const arma.vec& vd, const double clock, const double shape,
                       const unsigned massNum, const double ioniz, const double gain, const double tilt,
                       const arma.vec& beamCtr) except+

        cppmap[pad_t, arma.vec] makeEvent(const arma.mat& pos, const arma.vec& en) except+
        arma.mat makePeaksTableFromSimulation(const arma.mat& pos, const arma.vec& en) except+
        arma.vec makeMeshSignal(const arma.mat& pos, const arma.vec& en) except+

    cdef cppclass MCminimizeResult:
        MCminimizeResult() except +
        arma.vec ctr
        arma.mat allParams
        arma.vec minPosChis
        arma.vec minEnChis
        arma.vec goodParamIdx

    cdef cppclass MCminimizer:
        MCminimizer(const Tracker& tracker, const EventGenerator& evtgen) except+

        arma.mat makeParams(const arma.vec& ctr, const arma.vec& sigma, const unsigned numSets,
                            const arma.vec& mins, const arma.vec& maxes) except+
        arma.mat findPositionDeviations(const arma.mat& simPos, const arma.mat& expPos) except+
        arma.vec findEnergyDeviation(const arma.mat& simPos, const arma.vec& simEn, const arma.vec& expMesh) except+
        MCminimizeResult minimize(const arma.vec& ctr0, const arma.vec& sigma0, const arma.mat& expPos,
                                  const arma.vec& expMesh, const unsigned numIters, const unsigned numPts,
                                  const double redFactor) except+
