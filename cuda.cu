#include <stdio.h>
#include "cuda.h"

static __global__ void testkernel() {
    printf("hello from kernel\n");
}

int test() {
    printf("hello from cuda\n");
    testkernel<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}
