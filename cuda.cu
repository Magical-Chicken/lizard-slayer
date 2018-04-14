#include <stdio.h>
#include "cuda.h"

static launch_kernel(int Dg, int Db, int Ns, void *kernel) {
    kernel<<<Dg, Db, Ns>>>();
}

static __global__ void aggregate_kernel() {
    printf("hello from aggregate kernel\n");
}

int aggregate(int Dg, int Db, int Ns,) {
    printf("hello from cuda\n");
    launch_kernel(1, 1, 0, aggregate_kernel);
    cudaDeviceSynchronize();
    return 0;
}
