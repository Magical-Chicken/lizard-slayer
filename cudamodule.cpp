#include <Python.h>
#include "structmember.h"
#include "cuda.h"

extern "C" {
    
// FIXME FIXME FIXME 
// This is only an exmaple of TYPE extension that calls into a trivial cuda
// kernel
typedef struct {
    PyObject_HEAD
    PyObject *first;
    PyObject *last;
    int number;
} CustomObject;

static void Custom_dealloc(CustomObject *self) {
    Py_XDECREF(self->first);
    Py_XDECREF(self->last);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

static PyObject *
Custom_new(PyTypeObject *type, PyObject *args, PyObject *kwds)
{
    CustomObject *self;

    self = (CustomObject *)type->tp_alloc(type, 0);
    if (self != NULL) {
        self->first = PyUnicode_FromString("");
        if (self->first == NULL) {
            Py_DECREF(self);
            return NULL;
        }

        self->last = PyUnicode_FromString("");
        if (self->last == NULL) {
            Py_DECREF(self);
            return NULL;
        }

        self->number = 0;
    }

    return (PyObject *)self;
}

static int
Custom_init(CustomObject *self, PyObject *args, PyObject *kwds)
{
    PyObject *first=NULL, *last=NULL, *tmp;

    static char *kwlist[] = {"first", "last", "number", NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "|OOi", kwlist,
                                      &first, &last,
                                      &self->number))
        return -1;

    if (first) {
        tmp = self->first;
        Py_INCREF(first);
        self->first = first;
        Py_XDECREF(tmp);
    }

    if (last) {
        tmp = self->last;
        Py_INCREF(last);
        self->last = last;
        Py_XDECREF(tmp);
    }

    return 0;
}

static PyMemberDef Custom_members[] = {
    {"first", T_OBJECT_EX, offsetof(CustomObject, first), 0,
        "first name"},
    {"last", T_OBJECT_EX, offsetof(CustomObject, last), 0,
        "last name"},
    {"number", T_INT, offsetof(CustomObject, number), 0,
        "custom number"},
    {NULL}  /* Sentinel */
};

static PyTypeObject CustomType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "custom.Custom",             /* tp_name */
    sizeof(CustomObject), /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor)Custom_dealloc,                         /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_as_async */
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
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "Custom objects",           /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    0,             /* tp_methods */
    Custom_members,             /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)Custom_init,      /* tp_init */
    0,                         /* tp_alloc */
    Custom_new,                 /* tp_new */
};

static PyObject *cuda_test(PyObject *self, PyObject *arg) {
    printf("hello world!\n");
    //printf("In c: %s %s %i\n", PyUnicode_AsUTF8(arg->first), PyUnicode_AsUTF8(arg->last), arg->number);

    Py_buffer view;
    int r = PyObject_GetBuffer(arg, &view, 0);
    printf("result: %i\n", r);
    for (int i = 0; i< 3; i++) {
        printf("%lf\n", ((double*)view.buf)[i]);
        ((double *)view.buf)[i] = 122.2;
    }
    TYPE res = aggregate(view.buf, view.len, view.itemsize, 1, 1, 0);

    PyBuffer_Release(&view);
    printf("result: %lf\n",res);
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

    TYPE *dev_points = NULL, *dev_partial_results = NULL;
    int *dev_count_results = NULL;
    printf("cuda_kmeans_iteration\n");

    static char *kwlist[] = {
        "centers", 
        "points", 
        "partial_results", 
        "count_results", 
        "k", 
        "dim", 
        "Dg", 
        "Db", 
        "Ns", 
        NULL};

    if (! PyArg_ParseTupleAndKeywords(args, kwds, "OOOO|iiiii", kwlist,
                &centers, 
                &points, 
                &partial_results, 
                &count_results, 
                &k, 
                &dim, 
                &Dg, 
                &Db, 
                &Ns))
        return NULL;

    if(PyObject_GetBuffer(centers, &centers_view, 0) != 0)
        return NULL;

    if(PyObject_GetBuffer(points, &points_view, 0) != 0)
        return NULL;

    if(PyObject_GetBuffer(partial_results, &partial_results_view, 0) != 0)
        return NULL;

    if(PyObject_GetBuffer(count_results, &count_results_view, 0) != 0)
        return NULL;

    deviceMalloc((void**)&dev_points, points_view.len);
    cudaMemcpyToDevice(dev_points, points_view.buf, points_view.len);

    deviceMalloc((void**)&dev_partial_results, partial_results_view.len);
    cudaMemcpyToDevice(
            dev_partial_results,
            partial_results_view.buf,
            partial_results_view.len);

    deviceMalloc((void**)&dev_count_results, count_results_view.len);
    cudaMemcpyToDevice(
            dev_count_results, count_results_view.buf, count_results_view.len);

    kmeans_iteration(
            (TYPE*)centers_view.buf,
            dev_points,
            dev_partial_results,
            dev_count_results,
            points_view.len,
            points_view.itemsize,
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
    PyBuffer_Release(&points_view);
    PyBuffer_Release(&partial_results_view);
    PyBuffer_Release(&count_results_view);

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
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef cudamodule = {
    PyModuleDef_HEAD_INIT,
    "cuda",   /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
    CudaMethods
};

PyMODINIT_FUNC
PyInit_cuda(void)
{
    PyObject *m;

	/*CustomType.tp_new = PyType_GenericNew;*/
    if(PyType_Ready(&CustomType) < 0)
        return NULL;

    m = PyModule_Create(&cudamodule);
    if(m == NULL)
        return NULL;

    Py_INCREF(&CustomType);
    PyModule_AddObject(m, "Custom", (PyObject *)&CustomType);
    return m;

}
}
