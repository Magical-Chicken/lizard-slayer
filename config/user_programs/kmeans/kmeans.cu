#include <stdio.h>
#include <float.h>

#include "program.h"

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
    return cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost) == cudaSuccess;
}

/*
 * Wrapper around cudaMalloc. 
 * returns true on success
 */
bool deviceMalloc(void **dev_ptr, long size) {
    return cudaMalloc(dev_ptr, size) == cudaSuccess;
}

bool deviceFree(void *dev_ptr) {
    return cudaFree(dev_ptr) == cudaSuccess;
}

static __global__ void kmeans_iteration_kernel(double *centers, double *points,
        double *partial_results, int *count_results, long count, int dim, int k) {
    long index = threadIdx.x + blockIdx.x * blockDim.x;

    if (index < count) {
        int cluster = -1; 
        double shortest = DBL_MAX;
        for (int i = 0; i < k; i++) {
            TYPE mag = 0;

            for (int d = 0; d < dim; d++) {
                double c = points[index * dim + d] - centers[i * dim + d];
                mag += c * c;
            }
            /*printf("mag %lf\n", mag);*/

            if (mag < shortest) {
                shortest = mag;
                cluster = i;
            }
        }
        /*printf("kernel: point %lf\n", points[index]);*/
        /*printf("kernel: cluster %i\n", cluster);*/
         
        for (int d = 0; d < dim; d++) {
            atomicAdd(&partial_results[cluster * dim + d], points[index * dim + d]);
            atomicAdd(&count_results[cluster], 1);
            /*printf("results: %lf\n", results[cluster *dim+d]);*/
        }
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

void kmeans_iteration(double *centers, double *dev_points, double *dev_partial_results, 
        int *dev_count_results, long size, long itemsize, int k, int dim, int Dg, int Db, int Ns) {
    double *dev_centers = NULL;

    cudaMalloc(&dev_centers, itemsize * k * dim);
    cudaMemcpy(dev_centers, centers, itemsize * k * dim, cudaMemcpyHostToDevice);

    /*printf("count: %i\n", size / itemsize/ dim);*/
    /*for (int i = 0; i < 4; i++) */
        /*printf("%lf\n", centers[i]);*/

    kmeans_iteration_kernel<<<Dg, Db, Ns>>>(dev_centers, dev_points,
            dev_partial_results, dev_count_results, size / itemsize / dim, dim, k);

    cudaFree(dev_centers);
}

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
