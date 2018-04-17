#include <stdio.h>
#include <float.h>

#include "cuda.h"

/*
 * Wrapper around cudaMemcpy to copy memory to device.
 * returns true on success
 */
bool cudaMemcpyToDevice(void *dst, void *src, long size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess;
}

/*
 * Wrapper around cudaMemcpy to copy memory to device.
 * returns true on success
 */
bool cudaMemcpyToHost(void *dst, void *src, long size) {
    return cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice) == cudaSuccess;
}

/*
 * Wrapper around cudaMalloc. 
 * returns true on success
 */
bool deviceMalloc(void **dev_ptr, long size) {
    return cudaMalloc(dev_ptr, size) == cudaSuccess;
}

/*static void launch_kernel(int Dg, int Db, int Ns, void (*kernel)()) {*/
    /*kernel<<<Dg, Db, Ns>>>();*/
/*}*/

static __global__ void kmeans_iteration_kernel(TYPE *centers, TYPE *points, 
        long count, int dim, int k, TYPE *result) {
    long index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < count) {

        int cluster; 
        TYPE shortest = DBL_MAX;
        for (int i = 0; i < k; i++) {
            TYPE mag = 0;

            for (int d = 0; d < dim; d++) {
                TYPE c = points[index * dim + d] - centers[i * dim + d];
                mag = c * c;
            }

            if (mag < shortest)
                cluster = i;
        }
         
        for (int d = 0; d < dim; d++)
            atomicAdd(&result[cluster * dim + d], points[index * dim + d]);
    }
}

static __global__ void aggregate_kernel(void *buf, long count, void *result) {
    /*printf("hello from aggregate kernel\n");*/
    long index = threadIdx.x + blockIdx.x * blockDim.x;

    // FIXME 
    // initial value
    /*TYPE result = 0;*/
    TYPE  *array = (TYPE *)buf;

    if (index < count) {
        atomicAdd((TYPE*)result, array[index]);
    }
}

/*void kmeans_iteration(void *centers, void *points, long size, long itemsize,*/
        /*int k, int dim, int Dg, int Db, int Ns) {*/
    /*void *device_centers = NULL;*/
    /*void *device_points = NULL;*/
    /*void *device_result = NULL;*/

    /*cudaMalloc(&device_result, itemsize * k * dim);*/
    /*cudaMemcpy(device_result, &result, itemsize, cudaMemcpyHostToDevice);*/

    /*cudaMalloc(&device_data, size);*/
    /*cudaMemcpy(device_data, buf, size, cudaMemcpyHostToDevice);*/

    /*kmeans_iteration_kernel<<<Dg, Db, Ns>>>();*/
/*}*/

TYPE aggregate(void *buf, long size, long itemsize, int Dg, int Db, int Ns) {
    printf("hello from cuda\n");
    /*void *device_data = malloc(sizeof(void *));*/
    void *device_data = NULL;
    void *device_result = NULL;

    TYPE result = 0;

    cudaMalloc(&device_result, itemsize);
    cudaMemcpy(device_result, &result, itemsize, cudaMemcpyHostToDevice);

    cudaMalloc(&device_data, size);
    cudaMemcpy(device_data, buf, size, cudaMemcpyHostToDevice);

    /*launch_kernel(Dg, Db, Ns, aggregate_kernel);*/
    aggregate_kernel<<<Dg, Db, Ns>>>(device_data, size / itemsize,
            device_result);

    /*cudaDeviceSynchronize();*/
    cudaMemcpy(&result, device_result, itemsize, cudaMemcpyDeviceToHost);
    cudaMemcpy(buf, device_data, size, cudaMemcpyDeviceToHost);
    cudaFree(device_data);
    cudaFree(device_result);
    return result;
}
