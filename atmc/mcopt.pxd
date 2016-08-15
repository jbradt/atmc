from libcpp.vector cimport vector as cppvec
from libcpp.map cimport map as cppmap
cimport armadillo as arma

cdef extern from "mcopt/mcopt.h" namespace "mcopt":
    ctypedef unsigned short pad_t


    cdef cppclass Gas:
        Gas(const cppvec[double]& eloss, const cppvec[double]& enVsZ) except+
        double energyLoss(const double energy) except+
        double energyFromVertexZPosition(const double z) except+


    cdef cppclass Track:
        Track() except+
        arma.mat getMatrix() except+
        arma.mat getPositionMatrix() except+
        arma.vec getEnergyVector() except+
        size_t numPts() except+


    cdef cppclass Tracker:
        Tracker(unsigned massNum, unsigned chargeNum, const Gas* gas,
                arma.vec& efield, arma.vec& bfield) except+
        Track trackParticle(const double x0, const double y0, const double z0,
                            const double enu0,  const double azi0, const double pol0) except+
        unsigned int getMassNum()
        unsigned int getChargeNum()
        arma.vec getEfield()
        arma.vec getBfield()


    cdef cppclass PadPlane:
        PadPlane(const arma.Mat[pad_t]& lt, const double xLB, const double xDelta,
                 const double yLB, const double yDelta, const double rotAngle) except+
        pad_t getPadNumberFromCoordinates(const double x, const double y) except+
        @staticmethod
        cppvec[cppvec[cppvec[double]]] generatePadCoordinates(const double rotation_angle)


    cdef cppclass EventGenerator:
        EventGenerator(const PadPlane* pads, const arma.vec& vd, const double clock, const double shape,
                       const unsigned massNum, const double ioniz, const double gain, const double tilt,
                       const double diffSigma, const arma.vec& beamCtr) except+

        cppmap[pad_t, arma.vec] makeEvent(const arma.mat& pos, const arma.vec& en) except+
        arma.mat makePeaksTableFromSimulation(const arma.mat& pos, const arma.vec& en) except+
        arma.vec makeMeshSignal(const arma.mat& pos, const arma.vec& en) except+
        arma.vec makeHitPattern(const arma.mat& pos, const arma.vec& en) except+

        double getClock()
        void setClock(const double c)
        double getShape()
        void setShape(const double s)

        arma.vec vd
        unsigned massNum
        double ioniz
        double gain
        double tilt
        double diffSigma
        arma.vec beamCtr


    cdef cppclass MCminimizeResult:
        MCminimizeResult() except +
        arma.vec ctr
        arma.mat allParams
        arma.mat minChis
        arma.vec goodParamIdx


    cdef cppclass Chi2Set:
        Chi2Set() except+
        double posChi2
        double enChi2
        double vertChi2


    cdef cppclass MCminimizer:
        MCminimizer(const Tracker* tracker, const EventGenerator* evtgen) except+

        arma.mat makeParams(const arma.vec& ctr, const arma.vec& sigma, const unsigned numSets,
                            const arma.vec& mins, const arma.vec& maxes) except+
        arma.mat findPositionDeviations(const arma.mat& simPos, const arma.mat& expPos) except+
        arma.vec findHitPatternDeviation(const arma.mat& simPos, const arma.vec& simEn, const arma.vec& expHits) except+
        MCminimizeResult minimize(const arma.vec& ctr0, const arma.vec& sigma0, const arma.mat& expPos,
                                  const arma.vec& expMesh, const unsigned numIters, const unsigned numPts,
                                  const double redFactor) except+
        Chi2Set runTrack(const arma.vec& params, const arma.mat& expPos, const arma.vec& expHits) except+
        arma.mat runTracks(const arma.mat& params, const arma.mat& expPos, const arma.vec& expHits) except+

        bint posChi2Enabled
        bint enChi2Enabled
        bint vertChi2Enabled

        double posChi2Norm
        double enChi2NormFraction
        double vertChi2Norm


    cdef cppclass AnnealStopReason:
        bint operator==(const AnnealStopReason&) except+


    cdef:
        AnnealStopReason ANNEAL_CONVERGED "mcopt::AnnealStopReason::converged"
        AnnealStopReason ANNEAL_MAX_ITERS "mcopt::AnnealStopReason::maxIters"
        AnnealStopReason ANNEAL_TOO_MANY_CALLS "mcopt::AnnealStopReason::tooManyCalls"


    cdef cppclass AnnealResult:
        AnnealResult() except+
        arma.mat ctrs
        arma.mat chis
        AnnealStopReason stopReason
        int numCalls


    cdef cppclass Annealer:
        Annealer(const Tracker* tracker, const EventGenerator* evtgen, const double T0, const double coolRate,
                 const int numIters, const int maxCallsPerIter) except+
        AnnealResult minimize(const arma.vec& ctr0, const arma.vec& sigma0, const arma.mat& expPos,
                              const arma.vec& expHits) except+
        arma.vec randomStep(const arma.vec& ctr, const arma.vec& sigma) except+
        double T0
        double coolRate
        int numIters
        int maxCallsPerIter
        size_t multiMinimizeNumTrials

        bint posChi2Enabled
        bint enChi2Enabled
        bint vertChi2Enabled

        double posChi2Norm
        double enChi2NormFraction
        double vertChi2Norm
