from libcpp.vector cimport vector as cppvec
from libcpp.map cimport map as cppmap
cimport armadillo as arma

cdef extern from "mcopt/mcopt.h" namespace "mcopt":

    ctypedef unsigned short pad_t

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

    cdef cppclass PadPlane:
        PadPlane(const arma.Mat[unsigned short]& lt, const double xLB, const double xDelta,
                 const double yLB, const double yDelta, const double rotAngle)
        pad_t getPadNumberFromCoordinates(const double x, const double y)

    cdef cppclass EventGenerator:
        EventGenerator(const PadPlane& pads, const arma.vec& vd, const double clock, const double shape,
                       const unsigned massNum, const double ioniz, const double gain, const double tilt,
                       const arma.vec& beamCtr)

        cppmap[pad_t, arma.vec] makeEvent(const arma.mat& pos, const arma.vec& en)
        arma.mat makePeaksTableFromSimulation(const arma.mat& pos, const arma.vec& en)
        arma.vec makeMeshSignal(const arma.mat& pos, const arma.vec& en)

    cdef cppclass MCminimizeResult:
        MCminimizeResult() except +
        arma.vec ctr
        arma.mat allParams
        arma.vec minPosChis
        arma.vec minEnChis
        arma.vec goodParamIdx

    cdef cppclass MCminimizer:
        MCminimizer(const Tracker& tracker, const EventGenerator& evtgen)

        arma.mat makeParams(const arma.vec& ctr, const arma.vec& sigma, const unsigned numSets,
                                   const arma.vec& mins, const arma.vec& maxes)
        arma.mat findPositionDeviations(const arma.mat& simPos, const arma.mat& expPos)
        arma.vec findEnergyDeviation(const arma.mat& simPos, const arma.vec& simEn, const arma.vec& expMesh)
        MCminimizeResult minimize(const arma.vec& ctr0, const arma.vec& sigma0, const arma.mat& expPos,
                                  const arma.vec& expMesh, const unsigned numIters, const unsigned numPts,
                                  const double redFactor)
