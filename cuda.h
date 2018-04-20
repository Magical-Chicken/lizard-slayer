#ifndef SHARED_H__
#define SHARED_H__

#define TYPE double

bool cudaMemcpyToDevice(void *dst, void *src, long size);
bool cudaMemcpyToHost(void *dst, void *src, long size);
bool deviceMalloc(void **dev_ptr, long size);
bool deviceFree(void *dev_ptr);

TYPE aggregate(void *buf, long size, long itemsize, int Dg, int Db, int Ns);

void kmeans_iteration(double *centers, double *points, 
        double *dev_partial_results, int *dev_count_results, 
        long size, long itemsize, int k, int dim, int Dg, int Db, int Ns);
#endif


