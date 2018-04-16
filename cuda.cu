#include <stdio.h>
#include "cuda.h"

void *cudaMemcpyToDevice(void *host_data, long size, long item_size) {
    void *device_data = NULL;

    cudaMalloc(&device_data, size);
    cudaMemcpy(device_data, host_data, size, cudaMemcpyHostToDevice);
    return device_data;
}

static void launch_kernel(int Dg, int Db, int Ns, void (*kernel)()) {
    kernel<<<Dg, Db, Ns>>>();
}


static __global__ void aggregate_kernel(void *buf, long count, void *result) {
    /*printf("hello from aggregate kernel\n");*/
    long index = threadIdx.x + blockIdx.x * blockDim.x;

    // FIXME 
    // initial value
    /*type result = 0;*/
    type  *array = (type *)buf;

    if (index < count) {
        atomicAdd((type*)result, array[index]);
    }
}

type aggregate(void *buf, long size, long itemsize, int Dg, int Db, int Ns) {
    printf("hello from cuda\n");
    /*void *device_data = malloc(sizeof(void *));*/
    void *device_data = NULL;
    void *device_result = NULL;

    type result = 0;

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
