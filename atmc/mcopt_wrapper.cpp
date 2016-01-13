#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION

extern "C" {
    #include <Python.h>
    #include <numpy/arrayobject.h>
}
#include "mcopt.h"
#include <exception>
#include <cassert>
#include <stdio.h>
#include <string>

class WrongDimensions : public std::exception
{
public:
    WrongDimensions() {}
    const char* what() const noexcept { return msg.c_str(); }

private:
    std::string msg = "The dimensions were incorrect";
};

class BadArrayLayout : public std::exception
{
public:
    BadArrayLayout() {}
    const char* what() const noexcept { return msg.c_str(); }

private:
    std::string msg = "The matrix was not contiguous";
};

static std::vector<double> convertPyArrayToVector(PyArrayObject* pyarr)
{
    int ndim = PyArray_NDIM(pyarr);
    if (ndim != 1) throw WrongDimensions();
    npy_intp* dims = PyArray_SHAPE(pyarr);

    double* dataPtr = static_cast<double*>(PyArray_DATA(pyarr));
    return std::vector<double>(dataPtr, dataPtr+dims[0]);
}

static const std::vector<npy_intp> getPyArrayDimensions(PyArrayObject* pyarr)
{
    npy_intp ndims = PyArray_NDIM(pyarr);
    npy_intp* dims = PyArray_SHAPE(pyarr);
    std::vector<npy_intp> result;
    for (int i = 0; i < ndims; i++) {
        result.push_back(dims[i]);
    }
    return result;
}

/* Checks the dimensions of the given array. Pass -1 for either dimension to say you don't
 * care what the size is in that dimension. Pass dimensions (X, 1) for a vector.
 */
static bool checkPyArrayDimensions(PyArrayObject* pyarr, const npy_intp dim0, const npy_intp dim1)
{
    const auto dims = getPyArrayDimensions(pyarr);
    assert(dims.size() <= 2 and dims.size() > 0);
    if (dims.size() == 1) {
        return (dims[0] == dim0) and (dim1 == 1);
    }
    else {
        return (dims[0] == dim0 or dim0 == -1) and (dims[1] == dim1 or dim1 == -1);
    }
}

static arma::mat convertPyArrayToArma(PyArrayObject* pyarr, int nrows, int ncols)
{
    if (!checkPyArrayDimensions(pyarr, nrows, ncols)) throw WrongDimensions();
    const auto dims = getPyArrayDimensions(pyarr);
    if (dims.size() == 1) {
        double* dataPtr = static_cast<double*>(PyArray_DATA(pyarr));
        return arma::vec(dataPtr, dims[0], true);
    }
    else {
        // Convert the array to a Fortran-contiguous (col-major) array of doubles, as required by Armadillo
        PyArray_Descr* reqDescr = PyArray_DescrFromType(NPY_DOUBLE);
        if (reqDescr == NULL) throw std::bad_alloc();
        PyArrayObject* cleanArr = (PyArrayObject*)PyArray_FromArray(pyarr, reqDescr, NPY_ARRAY_FARRAY);
        if (cleanArr == NULL) throw std::bad_alloc();
        reqDescr = NULL;  // The new reference from DescrFromType was stolen by FromArray

        double* dataPtr = static_cast<double*>(PyArray_DATA(cleanArr));
        arma::mat result (dataPtr, dims[0], dims[1], true);  // this copies the data from cleanArr
        Py_DECREF(cleanArr);
        return result;
    }
}

static PyObject* convertArmaToPyArray(const arma::mat& matrix)
{
    npy_intp ndim = matrix.is_colvec() ? 1 : 2;
    npy_intp nRows = static_cast<npy_intp>(matrix.n_rows);  // NOTE: This narrows the integer
    npy_intp nCols = static_cast<npy_intp>(matrix.n_cols);
    npy_intp dims[2] = {nRows, nCols};

    PyObject* result = PyArray_SimpleNew(ndim, dims, NPY_DOUBLE);
    if (result == NULL) throw std::bad_alloc();

    double* resultDataPtr = static_cast<double*>(PyArray_DATA((PyArrayObject*)result));
    for (int i = 0; i < nRows; i++) {
        for (int j = 0; j < nCols; j++) {
            resultDataPtr[i * nCols + j] = matrix(i, j);
        }
    }

    return result;
}

extern "C" {
    const char* mcopt_wrapper_track_particle_docstring =
        "Simulate the trajectory of a particle.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "x0, y0, z0, enu0, azi0, pol0 : float\n"
        "    The initial position (m), energy per nucleon (MeV/u), and azimuthal and polar angles (rad).\n"
        "massNum, chargeNum : int\n"
        "    The mass and charge numbers of the projectile.\n"
        "eloss : ndarray\n"
        "    The energy loss for the particle, in MeV/m, as a function of projectile energy.\n"
        "    This should be indexed in 1-keV steps.\n"
        "efield, bfield : array-like\n"
        "    The electric and magnetic fields, in SI units.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "ndarray\n"
        "    The simulated track. The columns are x, y, z, time, energy/nucleon, azimuthal angle, polar angle.\n"
        "    The positions are in meters, the time is in seconds, and the energy is in MeV/u.\n"
        "\n"
        "Raises\n"
        "------\n"
        "ValueError\n"
        "    If the dimensions of an input array were invalid.\n"
        "RuntimeError\n"
        "    If tracking fails for some reason.\n";
    static PyObject* mcopt_wrapper_track_particle(PyObject* self, PyObject* args)
    {
        double x0, y0, z0, enu0, azi0, pol0;
        unsigned int massNum, chargeNum;
        double efield[3];
        double bfield[3];
        PyArrayObject* eloss_array = NULL;
        if (!PyArg_ParseTuple(args, "ddddddiiO!(ddd)(ddd)", &x0, &y0, &z0, &enu0, &azi0, &pol0,
                              &massNum, &chargeNum, &PyArray_Type, &eloss_array, &efield[0], &efield[1], &efield[2],
                              &bfield[0], &bfield[1], &bfield[2])) {
            return NULL;
        }

        std::vector<double> eloss;
        try {
            eloss = convertPyArrayToVector(eloss_array);
        }
        catch (std::exception& err) {
            PyErr_SetString(PyExc_ValueError, err.what());
            return NULL;
        }

        mcopt::Tracker tracker {massNum, chargeNum, eloss, arma::vec(efield, 3), arma::vec(bfield, 3)};

        mcopt::Track tr;
        try {
            tr = tracker.trackParticle(x0, y0, z0, enu0, azi0, pol0);
        }
        catch (std::exception& err) {
            PyErr_SetString(PyExc_RuntimeError, err.what());
            return NULL;
        }

        PyObject* result = NULL;
        try {
            result = convertArmaToPyArray(tr.getMatrix());
        }
        catch (const std::bad_alloc&){
            PyErr_NoMemory();
            return NULL;
        }

        return result;
    }

    const char* mcopt_wrapper_MCminimize_docstring =
        "Perform chi^2 minimization for the track.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "ctr0 : ndarray\n"
        "    The initial guess for the track's parameters. These are (x0, y0, z0, enu0, azi0, pol0, bmag0).\n"
        "sig0 : ndarray\n"
        "    The initial width of the parameter space in each dimension. The distribution will be centered on `ctr0` with a\n"
        "    width of `sig0 / 2` in each direction.\n"
        "trueValues : ndarray\n"
        "    The experimental data points, as (x, y, z) triples.\n"
        "massNum, chargeNum : int\n"
        "    The mass and charge number of the projectile.\n"
        "eloss : ndarray\n"
        "    The energy loss for the particle, in MeV/m, as a function of projectile energy.\n"
        "    This should be indexed in 1-keV steps.\n"
        "ioniz : float\n"
        "    The mean ionization potential of the gas, in eV.\n"
        "efield : ndarray\n"
        "    The electric field vector.\n"
        "numIters : int\n"
        "    The number of iterations to perform before stopping. Each iteration draws `numPts` samples and picks the best one.\n"
        "numPts : int\n"
        "    The number of samples to draw in each iteration. The tracking function will be evaluated `numPts * numIters` times.\n"
        "redFactor : float\n"
        "    The factor to multiply the width of the parameter space by on each iteration. Should be <= 1.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "ctr : ndarray\n"
        "    The fitted track parameters.\n"
        "minChis : ndarray\n"
        "    The minimum chi^2 value at the end of each iteration.\n"
        "allParams : ndarray\n"
        "    The parameters from all generated tracks. There will be `numIters * numPts` rows.\n"
        "goodParamIdx : ndarray\n"
        "    The row numbers in `allParams` corresponding to the best points from each iteration, i.e. the ones whose\n"
        "    chi^2 values are in `minChis`.\n"
        "\n"
        "Raises\n"
        "------\n"
        "ValueError\n"
        "    If a provided array has invalid dimensions.\n"
        "RuntimeError\n"
        "    If tracking fails for some reason.\n";
    static PyObject* mcopt_wrapper_MCminimize(PyObject* self, PyObject* args)
    {
        PyArrayObject* ctr0Arr = NULL;
        PyArrayObject* sig0Arr = NULL;
        PyArrayObject* trueValuesArr = NULL;
        unsigned massNum;
        unsigned chargeNum;
        PyArrayObject* elossArr = NULL;
        double ioniz;
        PyArrayObject* efieldArr = NULL;
        unsigned numIters;
        unsigned numPts;
        double redFactor;

        if (!PyArg_ParseTuple(args, "O!O!O!iiO!dO!iid",
                              &PyArray_Type, &ctr0Arr, &PyArray_Type, &sig0Arr, &PyArray_Type, &trueValuesArr,
                              &massNum, &chargeNum, &PyArray_Type, &elossArr, &ioniz, &PyArray_Type, &efieldArr,
                              &numIters, &numPts, &redFactor)) {
            return NULL;
        }
        arma::vec ctr0, sig0, efield;
        arma::mat trueValues;
        std::vector<double> eloss;

        try {
            ctr0 = convertPyArrayToArma(ctr0Arr, 7, 1);
            sig0 = convertPyArrayToArma(sig0Arr, 7, 1);
            efield = convertPyArrayToArma(efieldArr, 3, 1);
            trueValues = convertPyArrayToArma(trueValuesArr, -1, 3);
            eloss = convertPyArrayToVector(elossArr);
        }
        catch (std::exception& err) {
            PyErr_SetString(PyExc_ValueError, err.what());
            return NULL;
        }

        mcopt::MCminimizer minimizer {massNum, chargeNum, eloss, efield, arma::zeros<arma::vec>(3), ioniz};

        arma::vec ctr;
        arma::mat allParams;
        arma::vec minChis;
        arma::vec goodParamIdx;
        try {
            std::tie(ctr, allParams, minChis, goodParamIdx) =
                minimizer.minimize(ctr0, sig0, trueValues, numIters, numPts, redFactor);
        }
        catch (std::exception& err) {
            PyErr_SetString(PyExc_RuntimeError, err.what());
            return NULL;
        }

        assert(ctr.n_rows == ctr0.n_rows);

        // Allocate numpy arrays to return
        PyObject* ctrArr = NULL;
        PyObject* allParamsArr = NULL;
        PyObject* minChisArr = NULL;
        PyObject* goodParamIdxArr = NULL;

        try {
            ctrArr = convertArmaToPyArray(ctr);
            allParamsArr = convertArmaToPyArray(allParams);
            minChisArr = convertArmaToPyArray(minChis);
            goodParamIdxArr = convertArmaToPyArray(goodParamIdx);
        }
        catch (std::bad_alloc) {
            Py_XDECREF(ctrArr);
            Py_XDECREF(allParamsArr);
            Py_XDECREF(minChisArr);
            Py_XDECREF(goodParamIdxArr);

            PyErr_NoMemory();
            return NULL;
        }

        PyObject* result = Py_BuildValue("OOOO", ctrArr, minChisArr, allParamsArr, goodParamIdxArr);
        Py_DECREF(ctrArr);
        Py_DECREF(allParamsArr);
        Py_DECREF(minChisArr);
        Py_DECREF(goodParamIdxArr);
        return result;
    }

    const char* mcopt_wrapper_find_deviations_docstring =
        "Find the deviations. These can be summed to find chi^2.\n"
        "\n"
        "Parameters\n"
        "----------\n"
        "simArr : ndarray\n"
        "    The simulated track.\n"
        "expArr : ndarray\n"
        "    The experimental data.\n"
        "\n"
        "Returns\n"
        "-------\n"
        "devArr : ndarray\n"
        "    The array of differences (or deviations).\n";
    static PyObject* mcopt_wrapper_find_deviations(PyObject* self, PyObject* args)
    {
        PyArrayObject* simArr = NULL;
        PyArrayObject* expArr = NULL;

        if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &simArr, &PyArray_Type, &expArr)) {
            return NULL;
        }

        PyObject* devArr = NULL;
        try {
            arma::mat simMat = convertPyArrayToArma(simArr, -1, -1);
            arma::mat expMat = convertPyArrayToArma(expArr, -1, -1);

            // printf("SimMat has shape (%lld, %lld)", simMat.n_rows, simMat.n_cols);
            // printf("ExpMat has shape (%lld, %lld)", expMat.n_rows, expMat.n_cols);

            arma::mat devs = mcopt::MCminimizer::findDeviations(simMat, expMat);

            devArr = convertArmaToPyArray(devs);
        }
        catch (std::bad_alloc) {
            PyErr_NoMemory();
            Py_XDECREF(devArr);
            return NULL;
        }
        catch (std::exception& err) {
            PyErr_SetString(PyExc_RuntimeError, err.what());
            Py_XDECREF(devArr);
            return NULL;
        }

        return devArr;
    }

    static PyMethodDef mcopt_wrapper_methods[] =
    {
        {"track_particle", mcopt_wrapper_track_particle, METH_VARARGS, mcopt_wrapper_track_particle_docstring},
        {"MCminimize", mcopt_wrapper_MCminimize, METH_VARARGS, mcopt_wrapper_MCminimize_docstring},
        {"find_deviations", mcopt_wrapper_find_deviations, METH_VARARGS, mcopt_wrapper_find_deviations_docstring},
        {NULL, NULL, 0, NULL}
    };

    static struct PyModuleDef mcopt_wrapper_module = {
       PyModuleDef_HEAD_INIT,
       "mcopt_wrapper",   /* name of module */
       NULL, /* module documentation, may be NULL */
       -1,       /* size of per-interpreter state of the module,
                    or -1 if the module keeps state in global variables. */
       mcopt_wrapper_methods
    };

    PyMODINIT_FUNC
    PyInit_mcopt_wrapper(void)
    {
        import_array();
        return PyModule_Create(&mcopt_wrapper_module);
    }
}
