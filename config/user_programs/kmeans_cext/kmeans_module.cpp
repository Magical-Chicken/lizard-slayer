#include <Python.h>
#include "structmember.h"
#include "program.h"

extern "C" {

#define DEVICE_MEMORY "device_memory"
    
static PyObject *cuda_test(PyObject *self, PyObject *arg) {
    printf("hello world!\n");
    printf("hello world!\n");
    //printf("In c: %s %s %i\n", PyUnicode_AsUTF8(arg->first), PyUnicode_AsUTF8(arg->last), arg->number);

    //Py_buffer view;
    //int r = PyObject_GetBuffer(arg, &view, 0);
    //printf("result: %i\n", r);
    //for (int i = 0; i< 3; i++) {
        //printf("%lf\n", ((double*)view.buf)[i]);
        //((double *)view.buf)[i] = 122.2;
    //}
    //TYPE res = aggregate(view.buf, view.len, view.itemsize, 1, 1, 0);

    //PyBuffer_Release(&view);
    //printf("result: %lf\n",res);
    return Py_None;
}

static PyObject *cuda_pin_gpu_memory(PyObject *self, PyObject *data) {
    Py_buffer view;
    void *pinned_mem;

    if(PyObject_GetBuffer(data, &view, 0) != 0)
        return NULL;

    printf("VIEW LENGTH %i\n", view.len);

    // FIXME ERROR CHECK
    if (!deviceMalloc(&pinned_mem, view.len))
        return NULL;
    if (!cudaMemcpyToDevice(pinned_mem, view.buf, view.len)) {
        return NULL;
    }
    // FIXME destructor for capsule should not be needed
    return PyCapsule_New(pinned_mem, DEVICE_MEMORY, NULL);
}

static PyObject *cuda_unpin_gpu_memory(PyObject *self, PyObject *capsule) {
    void *pinned_mem;

    if (!PyCapsule_CheckExact(capsule)) {
        PyErr_SetString(PyExc_TypeError, "must be a python capsule");
        return NULL;
    }

    if ((pinned_mem = PyCapsule_GetPointer(capsule, DEVICE_MEMORY)) == NULL) {
        return NULL;
    }
    
    deviceFree(pinned_mem);
    return Py_None;
}

static PyObject *cuda_aggregate(PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *data = NULL;
    Py_buffer view;
    int Dg, Db, Ns;
    printf("cuda_aggregate\n");

    static char *kwlist[] = {"data", "Dg", "Db", "Ns", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "O|iii", kwlist,
                &data, &Dg, &Db, &Ns))
        return NULL;

    if(PyObject_GetBuffer(data, &view, 0) != 0)
        return NULL;


    //printf("%li\n", view.len);
    //printf("%i, %i, %i\n", Dg, Db, Ns);
    TYPE res = aggregate(view.buf, view.len, view.itemsize, Dg, Db, Ns);
    PyBuffer_Release(&view);
    //printf("result: %lf\n",res);
    return Py_BuildValue("d", res);
}

static PyObject *cuda_kmeans_iteration(
        PyObject *self, PyObject *args, PyObject *kwds) {
    PyObject *centers = NULL;
    PyObject *points = NULL;
    // partial aggregation
    PyObject *partial_results = NULL;
    PyObject *count_results = NULL;
    Py_buffer centers_view, points_view;
    Py_buffer partial_results_view, count_results_view;
    int k, dim, Dg, Db, Ns;
    int count;

    TYPE *dev_points = NULL, *dev_partial_results = NULL;
    int *dev_count_results = NULL;
    bool using_capsule = false;
    printf("cuda_kmeans_iteration\n");

    static char *kwlist[] = {
        "centers", 
        "points", 
        "partial_results", 
        "count_results", 
        "count", 
        "k", 
        "dim", 
        "Dg", 
        "Db", 
        "Ns", 
        NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "OOOO|iiiiii", kwlist,
                &centers, 
                &points, 
                &partial_results, 
                &count_results, 
                &count,
                &k, 
                &dim, 
                &Dg, 
                &Db, 
                &Ns))
        return NULL;

    if(PyObject_GetBuffer(centers, &centers_view, 0) != 0)
        return NULL;

    if(PyCapsule_IsValid(points, DEVICE_MEMORY)) 
    {
        printf("ISvalud\n");
        using_capsule = true;
        void *p = PyCapsule_GetPointer(points, DEVICE_MEMORY);
        printf("ISvalud\n");
        //deviceMalloc((void**)&dev_points, count * dim * sizeof(double));
        //cudaMemcpyToDevice(dev_points, p, count * dim * sizeof(double));
        dev_points = (double* )p;
        printf("ISvalud\n");

    } else if(PyObject_GetBuffer(points, &points_view, 0) != 0) {
        return NULL;
    } else {
        printf("not ISvalud\n");
        deviceMalloc((void**)&dev_points, points_view.len);
        cudaMemcpyToDevice(dev_points, points_view.buf, points_view.len);
    }

    if(PyObject_GetBuffer(partial_results, &partial_results_view, 0) != 0)
        return NULL;

    if(PyObject_GetBuffer(count_results, &count_results_view, 0) != 0)
        return NULL;


    deviceMalloc((void**)&dev_partial_results, partial_results_view.len);
    cudaMemcpyToDevice(
            dev_partial_results,
            partial_results_view.buf,
            partial_results_view.len);

    deviceMalloc((void**)&dev_count_results, count_results_view.len);
    cudaMemcpyToDevice(
            dev_count_results, count_results_view.buf, count_results_view.len);

        printf("befpreo terate: %i\n",count);
    kmeans_iteration(
            (TYPE*)centers_view.buf,
            dev_points,
            dev_partial_results,
            dev_count_results,
            count,
            //points_view.len,
            //points_view.itemsize,
            k, dim, Dg, Db, Ns);

    cudaMemcpyToHost(
            partial_results_view.buf,
            dev_partial_results,
            partial_results_view.len);
    cudaMemcpyToHost(
            count_results_view.buf,
            dev_count_results,
            count_results_view.len);

    // release the buffer view
    PyBuffer_Release(&centers_view);
    if (!using_capsule)
        PyBuffer_Release(&points_view);
    PyBuffer_Release(&partial_results_view);
    PyBuffer_Release(&count_results_view);
        printf("befpreo free\n");

    // free cuda memory
    deviceFree(dev_partial_results);
    deviceFree(dev_count_results);
    return Py_None;
}

static PyMethodDef CudaMethods[] = {
    {"test",  (PyCFunction)cuda_test, METH_O, "Execute a shell command."},
    {"aggregate",  (PyCFunction)cuda_aggregate, METH_VARARGS|METH_KEYWORDS, 
        "Perform aggregate on GPU."},
    {"kmeans_iteration",  (PyCFunction)cuda_kmeans_iteration, METH_VARARGS|METH_KEYWORDS, 
        "Perform kmeans on GPU."},
    {"pin_gpu_memory",  (PyCFunction)cuda_pin_gpu_memory, METH_O, 
        "Pin memory in gpu."},
    {"unpin_gpu_memory",  (PyCFunction)cuda_unpin_gpu_memory, METH_O, 
        "Unpin memory in gpu."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cudamodule = {
    PyModuleDef_HEAD_INIT,
    "user_program",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CudaMethods
};

PyMODINIT_FUNC
PyInit_user_program(void)
{
    PyObject *m;

    m = PyModule_Create(&cudamodule);
    if(m == NULL)
        return NULL;

    return m;

}
}
