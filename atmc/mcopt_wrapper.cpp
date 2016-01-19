#define NPY_NO_DEPRECATED_API NPY_1_9_API_VERSION

extern "C" {
    #include <Python.h>
    #include <numpy/arrayobject.h>
    #include "docstrings.h"
}
#include "mcopt.h"
#include <exception>
#include <cassert>
#include <stdio.h>
#include <string>
#include <type_traits>
#include <map>

class WrongDimensions : public std::exception
{
public:
    WrongDimensions() {}
    const char* what() const noexcept { return msg.c_str(); }

private:
    std::string msg = "The dimensions were incorrect";
};

class NotImplemented : public std::exception
{
public:
    NotImplemented() {}
    const char* what() const noexcept { return msg.c_str(); }

private:
    std::string msg = "Not implemented";
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

template<typename outT>
static arma::Mat<outT> convertPyArrayToArma(PyArrayObject* pyarr, int nrows, int ncols)
{
    if (!checkPyArrayDimensions(pyarr, nrows, ncols)) throw WrongDimensions();

    int arrTypeCode;
    if (std::is_same<outT, uint16_t>::value) {
        arrTypeCode = NPY_UINT16;
    }
    else if (std::is_same<outT, double>::value) {
        arrTypeCode = NPY_DOUBLE;
    }
    else {
        throw NotImplemented();
    }

    const auto dims = getPyArrayDimensions(pyarr);
    if (dims.size() == 1) {
        outT* dataPtr = static_cast<outT*>(PyArray_DATA(pyarr));  // HACK: Potentially dangerous casting...
        return arma::Col<outT>(dataPtr, dims[0], true);
    }
    else {
        PyArray_Descr* reqDescr = PyArray_DescrFromType(arrTypeCode);
        if (reqDescr == NULL) throw std::bad_alloc();
        PyArrayObject* cleanArr = (PyArrayObject*)PyArray_FromArray(pyarr, reqDescr, NPY_ARRAY_FARRAY);
        if (cleanArr == NULL) throw std::bad_alloc();
        reqDescr = NULL;  // The new reference from DescrFromType was stolen by FromArray

        outT* dataPtr = static_cast<outT*>(PyArray_DATA(cleanArr));
        arma::Mat<outT> result (dataPtr, dims[0], dims[1], true);  // this copies the data from cleanArr
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

static mcopt::Track convertArmaToTrack(const arma::mat& matrix)
{
    if (matrix.n_cols != 7) throw BadArrayLayout();
    mcopt::Track tr;
    for (arma::uword i = 0; i < matrix.n_rows; i++) {
        arma::rowvec r = matrix.row(i);
        tr.append(r(0), r(1), r(2), r(3), r(4), r(5), r(6));
    }
    return tr;
}

// -------------------------------------------------------------------------------------------------------------------

extern "C" {
    typedef struct MCTracker {
        PyObject_HEAD
        mcopt::Tracker* tracker = NULL;
    } MCTracker;

    static int MCTracker_init(MCTracker* self, PyObject* args, PyObject* kwargs)
    {
        unsigned massNum;
        unsigned chargeNum;
        PyArrayObject* elossArray = NULL;
        double efield[3];
        double bfield[3];

        char* kwlist[] = {"mass_num", "charge_num", "eloss", "efield", "bfield", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IIO!(ddd)(ddd)", kwlist,
                                         &massNum, &chargeNum, &PyArray_Type, &elossArray,
                                         &efield[0], &efield[1], &efield[2],
                                         &bfield[0], &bfield[1], &bfield[2])) {
            return -1;
        }

        std::vector<double> eloss;
        try {
            eloss = convertPyArrayToVector(elossArray);
        }
        catch (std::exception& err) {
            PyErr_SetString(PyExc_ValueError, err.what());
            return -1;
        }

        if (self->tracker != NULL) {
            delete self->tracker;
            self->tracker = NULL;
        }

        self->tracker = new mcopt::Tracker(massNum, chargeNum, eloss, arma::vec(efield, 3), arma::vec(bfield, 3));
        return 0;
    }

    static void MCTracker_dealloc(MCTracker* self)
    {
        if (self->tracker != NULL) {
            delete self->tracker;
        }
    }

    static PyObject* MCTracker_trackParticle(MCTracker* self, PyObject* args)
    {
        double x0, y0, z0, enu0, azi0, pol0;

        if (self->tracker == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "The internal mcopt::Tracker object was NULL.");
            return NULL;
        }
        if (!PyArg_ParseTuple(args, "dddddd", &x0, &y0, &z0, &enu0, &azi0, &pol0)) {
            return NULL;
        }

        mcopt::Track tr;
        try {
            tr = self->tracker->trackParticle(x0, y0, z0, enu0, azi0, pol0);
        }
        catch (const std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
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

    static PyMethodDef MCTracker_methods[] = {
        {"track_particle", (PyCFunction)MCTracker_trackParticle, METH_VARARGS,
         "Track a particle"
        },
        {NULL}  /* Sentinel */
    };

    static PyTypeObject MCTrackerType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "mcopt_wrapper.Tracker",   /* tp_name */
        sizeof(MCTracker),         /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)MCTracker_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,        /* tp_flags */
        "mcopt Tracker",           /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        MCTracker_methods,         /* tp_methods */
        0,                         /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)MCTracker_init,  /* tp_init */
        0,                         /* tp_alloc */
        0,                         /* tp_new */
    };

    // ---------------------------------------------------------------------------------------------------------------

    typedef struct MCMCminimizer {
        PyObject_HEAD
        mcopt::MCminimizer* minimizer = NULL;
    } MCMCminimizer;

    static int MCMCminimizer_init(MCMCminimizer* self, PyObject* args, PyObject* kwargs)
    {
        unsigned massNum, chargeNum;
        PyArrayObject* elossArray = NULL;
        double efield[3], bfield[3];
        double ioniz;

        char* kwlist[] = {"mass_num", "charge_num", "eloss", "efield", "bfield", "ioniz", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "IIO!(ddd)(ddd)d", kwlist,
                                         &massNum, &chargeNum, &PyArray_Type, &elossArray,
                                         &efield[0], &efield[1], &efield[2],
                                         &bfield[0], &bfield[1], &bfield[2], &ioniz)) {
            return -1;
        }

        std::vector<double> eloss;
        try {
            eloss = convertPyArrayToVector(elossArray);
        }
        catch (std::exception& err) {
            PyErr_SetString(PyExc_ValueError, err.what());
            return -1;
        }

        if (self->minimizer != NULL) {
            delete self->minimizer;
            self->minimizer = NULL;
        }

        self->minimizer = new mcopt::MCminimizer(mcopt::Tracker(massNum, chargeNum, eloss,
                                                                arma::vec(efield, 3), arma::vec(bfield, 3)));
        return 0;
    }

    static void MCMCminimizer_dealloc(MCMCminimizer* self)
    {
        if (self->minimizer != NULL) {
            delete self->minimizer;
            self->minimizer = NULL;
        }
    }

    static PyObject* MCMCminimizer_minimize(MCMCminimizer* self, PyObject* args, PyObject* kwargs)
    {
        PyArrayObject* ctr0Arr = NULL;
        PyArrayObject* sig0Arr = NULL;
        PyArrayObject* trueValuesArr = NULL;
        unsigned numIters = 10;
        unsigned numPts = 200;
        double redFactor = 0.8;
        bool details = false;

        char* kwlist[] = {"ctr0", "sig0", "true_values", "num_iters", "num_pts", "red_factor", "details", NULL};

        if (self->minimizer == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "The internal mcopt::MCminimizer object was NULL.");
            return NULL;
        }
        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!O!|IIdp", kwlist,
                                         &PyArray_Type, &ctr0Arr, &PyArray_Type, &sig0Arr,
                                         &PyArray_Type, &trueValuesArr, &numIters, &numPts,
                                         &redFactor, &details)) {
            return NULL;
        }

        arma::vec ctr0, sig0;
        arma::mat trueValues;
        try {
            ctr0 = convertPyArrayToArma<double>(ctr0Arr, 7, 1);
            sig0 = convertPyArrayToArma<double>(sig0Arr, 7, 1);
            trueValues = convertPyArrayToArma<double>(trueValuesArr, -1, 4);
        }
        catch (std::exception& err) {
            PyErr_SetString(PyExc_ValueError, err.what());
            return NULL;
        }

        arma::vec ctr;
        arma::mat allParams;
        arma::vec minChis;
        arma::vec goodParamIdx;
        try {
            std::tie(ctr, allParams, minChis, goodParamIdx) =
                self->minimizer->minimize(ctr0, sig0, trueValues, numIters, numPts, redFactor);
        }
        catch (std::exception& err) {
            PyErr_SetString(PyExc_RuntimeError, err.what());
            return NULL;
        }

        PyObject* ctrArr = NULL;
        try {
            ctrArr = convertArmaToPyArray(ctr);
        }
        catch (const std::bad_alloc&) {
            PyErr_NoMemory();
            return NULL;
        }

        if (details) {
            PyObject* allParamsArr = NULL;
            PyObject* minChisArr = NULL;
            PyObject* goodParamIdxArr = NULL;

            try {
                allParamsArr = convertArmaToPyArray(allParams);
                minChisArr = convertArmaToPyArray(minChis);
                goodParamIdxArr = convertArmaToPyArray(goodParamIdx);
            }
            catch (const std::bad_alloc&) {
                Py_DECREF(ctrArr);
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
        else {
            double lastChi = minChis(minChis.n_rows-1);
            PyObject* result = Py_BuildValue("Od", ctrArr, lastChi);
            Py_DECREF(ctrArr);
            return result;
        }
    }

    static PyMethodDef MCMCminimizer_methods[] = {
        {"minimize", (PyCFunction)MCMCminimizer_minimize, METH_VARARGS | METH_KEYWORDS,
         "Perform MC minimization"
        },
        {NULL}  /* Sentinel */
    };

    static PyTypeObject MCMCminimizerType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "mcopt_wrapper.Minimizer", /* tp_name */
        sizeof(MCMCminimizer),     /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)MCMCminimizer_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,    /* tp_flags */
        "mcopt Minimizer",         /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        MCMCminimizer_methods,     /* tp_methods */
        0,                         /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)MCMCminimizer_init,  /* tp_init */
        0,                         /* tp_alloc */
        0,                         /* tp_new */
    };

    //  --------------------------------------------------------------------------------------------------------------

    typedef struct MCPadPlane {
        PyObject_HEAD
        mcopt::PadPlane* padplane = NULL;
    } MCPadPlane;

    static int MCPadPlane_init(MCPadPlane* self, PyObject* args, PyObject* kwargs)
    {
        PyArrayObject* ltArray = NULL;
        double xLB, xDelta, yLB, yDelta, rotAngle = 0;

        char* kwlist[] = {"lookup_table", "x_lower_bound", "x_delta", "y_lower_bound", "y_delta", "rot_angle", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!dddd|d", kwlist,
                                         &PyArray_Type, &ltArray, &xLB, &xDelta, &yLB, &yDelta, &rotAngle)) {
            return -1;
        }

        arma::Mat<mcopt::pad_t> lt;
        try {
            lt = convertPyArrayToArma<mcopt::pad_t>(ltArray, -1, -1);
        }
        catch (const std::exception& err) {
            PyErr_SetString(PyExc_RuntimeError, err.what());
            return -1;
        }

        if (self->padplane != NULL) {
            delete self->padplane;
            self->padplane = NULL;
        }

        self->padplane = new mcopt::PadPlane(lt, xLB, xDelta, yLB, yDelta, rotAngle);
        return 0;
    }

    static void MCPadPlane_dealloc(MCPadPlane* self)
    {
        if (self->padplane != NULL) {
            delete self->padplane;
            self->padplane = NULL;
        }
    }

    static PyObject* MCPadPlane_getPadNumberFromCoordinates(MCPadPlane* self, PyObject* args)
    {
        double x, y;

        if (!PyArg_ParseTuple(args, "dd", &x, &y)) return NULL;

        if (self->padplane == NULL) {
            PyErr_SetString(PyExc_RuntimeError, "Internal mcopt::PadPlane object was NULL.");
            return NULL;
        }

        mcopt::pad_t pad;
        try {
            pad = self->padplane->getPadNumberFromCoordinates(x, y);
        }
        catch (const std::exception& err) {
            PyErr_SetString(PyExc_RuntimeError, err.what());
            return NULL;
        }

        return Py_BuildValue("H", pad);
    }

    static mcopt::PadPlane* MCPadPlane_getObjPointer(MCPadPlane* self)
    {
        return self->padplane;
    }

    static PyMethodDef MCPadPlane_methods[] = {
        {"get_pad_number_from_coordinates", (PyCFunction)MCPadPlane_getPadNumberFromCoordinates, METH_VARARGS,
         "Get pad number from x and y coordinates."
        },
        {NULL}  /* Sentinel */
    };

    static PyTypeObject MCPadPlaneType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "mcopt_wrapper.PadPlane",  /* tp_name */
        sizeof(MCPadPlane),     /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)MCPadPlane_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,    /* tp_flags */
        "mcopt PadPlane",          /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        MCPadPlane_methods,     /* tp_methods */
        0,                         /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)MCPadPlane_init,  /* tp_init */
        0,                         /* tp_alloc */
        0,                         /* tp_new */
    };

    // ---------------------------------------------------------------------------------------------------------------

    typedef struct MCEventGenerator {
        PyObject_HEAD
        PyObject* padplaneObj = NULL;
        mcopt::EventGenerator* evtgen = NULL;
    } MCEventGenerator;

    static int MCEventGenerator_init(MCEventGenerator* self, PyObject* args, PyObject* kwargs)
    {
        PyObject* pads = NULL;
        double vd[3], clock, shape;
        unsigned massNum;
        double ioniz;
        unsigned gain;

        char* kwlist[] = {"pad_plane", "vd", "clock", "shape", "mass_num", "ioniz", "gain", NULL};

        if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!(ddd)ddIdI", kwlist,
                                         &MCPadPlaneType, &pads, &vd[0], &vd[1], &vd[2], &clock, &shape,
                                         &massNum, &ioniz, &gain)) {
            return -1;
        }

        PyObject* tmp = self->padplaneObj;
        Py_INCREF(pads);
        self->padplaneObj = pads;
        Py_XDECREF(tmp);

        if (self->evtgen != NULL) {
            delete self->evtgen;
            self->evtgen = NULL;
        }

        mcopt::PadPlane* padObjPtr = MCPadPlane_getObjPointer((MCPadPlane*)(self->padplaneObj));  // Can this be NULL?

        self->evtgen = new mcopt::EventGenerator(*padObjPtr, arma::vec(vd, 3), clock, shape, massNum, ioniz, gain);
        return 0;
    }

    static void MCEventGenerator_dealloc(MCEventGenerator* self)
    {
        if (self->evtgen != NULL) {
            delete self->evtgen;
            self->evtgen = NULL;
        }
        Py_XDECREF(self->padplaneObj);
    }

    static PyObject* MCEventGenerator_makeEvent(MCEventGenerator* self, PyObject* args)
    {
        PyArrayObject* trArr = NULL;
        if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &trArr)) return NULL;

        mcopt::Track tr;
        try {
            arma::mat trMat = convertPyArrayToArma<double>(trArr, -1, 7);
            tr = convertArmaToTrack(trMat);
        }
        catch (const std::exception& err) {
            PyErr_SetString(PyExc_RuntimeError, err.what());
            return NULL;
        }

        std::map<mcopt::pad_t, arma::vec> evt;
        try {
            evt = self->evtgen->makeEvent(tr);
        }
        catch (const std::exception& err) {
            PyErr_SetString(PyExc_RuntimeError, err.what());
            return NULL;
        }

        PyObject* resDict = PyDict_New();
        if (resDict == NULL) return NULL;

        for (const auto& pair : evt) {
            const mcopt::pad_t pad = pair.first;
            const arma::vec& sig = pair.second;

            PyObject* key = NULL;
            key = PyLong_FromUnsignedLong(pad);
            if (key == NULL) {
                Py_XDECREF(resDict);
                return NULL;
            }

            PyObject* sigArr = NULL;
            try {
                sigArr = convertArmaToPyArray(sig);
            }
            catch (const std::exception& err) {
                PyErr_SetString(PyExc_RuntimeError, err.what());
                Py_XDECREF(key);
                Py_XDECREF(resDict);
                return NULL;
            }

            if (PyDict_SetItem(resDict, key, sigArr) < 0) {
                PyErr_SetString(PyExc_RuntimeError, "Failed to insert result into dict.");
                Py_XDECREF(key);
                Py_XDECREF(sigArr);
                Py_XDECREF(resDict);
                return NULL;
            }
            Py_DECREF(key);
            Py_DECREF(sigArr);
        }

        return resDict;
    }

    static PyMethodDef MCEventGenerator_methods[] = {
        {"make_event", (PyCFunction)MCEventGenerator_makeEvent, METH_VARARGS,
         "Generate an event from the tracked particle matrix."
        },
        {NULL}  /* Sentinel */
    };

    static PyTypeObject MCEventGeneratorType = {
        PyVarObject_HEAD_INIT(NULL, 0)
        "mcopt_wrapper.EventGenerator",  /* tp_name */
        sizeof(MCEventGenerator),  /* tp_basicsize */
        0,                         /* tp_itemsize */
        (destructor)MCEventGenerator_dealloc, /* tp_dealloc */
        0,                         /* tp_print */
        0,                         /* tp_getattr */
        0,                         /* tp_setattr */
        0,                         /* tp_reserved */
        0,                         /* tp_repr */
        0,                         /* tp_as_number */
        0,                         /* tp_as_sequence */
        0,                         /* tp_as_mapping */
        0,                         /* tp_hash  */
        0,                         /* tp_call */
        0,                         /* tp_str */
        0,                         /* tp_getattro */
        0,                         /* tp_setattro */
        0,                         /* tp_as_buffer */
        Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,    /* tp_flags */
        "mcopt PadPlane",          /* tp_doc */
        0,                         /* tp_traverse */
        0,                         /* tp_clear */
        0,                         /* tp_richcompare */
        0,                         /* tp_weaklistoffset */
        0,                         /* tp_iter */
        0,                         /* tp_iternext */
        MCEventGenerator_methods,  /* tp_methods */
        0,                         /* tp_members */
        0,                         /* tp_getset */
        0,                         /* tp_base */
        0,                         /* tp_dict */
        0,                         /* tp_descr_get */
        0,                         /* tp_descr_set */
        0,                         /* tp_dictoffset */
        (initproc)MCEventGenerator_init,  /* tp_init */
        0,                         /* tp_alloc */
        0,                         /* tp_new */
    };

    // ---------------------------------------------------------------------------------------------------------------

    static PyObject* mcopt_wrapper_find_deviations(PyObject* self, PyObject* args)
    {
        PyArrayObject* simArr = NULL;
        PyArrayObject* expArr = NULL;

        if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &simArr, &PyArray_Type, &expArr)) {
            return NULL;
        }

        PyObject* devArr = NULL;
        try {
            arma::mat simMat = convertPyArrayToArma<double>(simArr, -1, -1);
            arma::mat expMat = convertPyArrayToArma<double>(expArr, -1, -1);

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

        MCTrackerType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&MCTrackerType) < 0) return NULL;

        MCMCminimizerType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&MCMCminimizerType) < 0) return NULL;

        MCPadPlaneType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&MCPadPlaneType) < 0) return NULL;

        MCEventGeneratorType.tp_new = PyType_GenericNew;
        if (PyType_Ready(&MCEventGeneratorType) < 0) return NULL;

        PyObject* m = PyModule_Create(&mcopt_wrapper_module);
        if (m == NULL) return NULL;

        Py_INCREF(&MCTrackerType);
        PyModule_AddObject(m, "Tracker", (PyObject*)&MCTrackerType);

        Py_INCREF(&MCMCminimizerType);
        PyModule_AddObject(m, "Minimizer", (PyObject*)&MCMCminimizerType);

        Py_INCREF(&MCPadPlaneType);
        PyModule_AddObject(m, "PadPlane", (PyObject*)&MCPadPlaneType);

        Py_INCREF(&MCEventGeneratorType);
        PyModule_AddObject(m, "EventGenerator", (PyObject*)&MCEventGeneratorType);

        return m;
    }
}
