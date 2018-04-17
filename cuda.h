#ifndef SHARED_H__
#define SHARED_H__

#define TYPE double

bool cudaMemcpyToDevice(void *dst, void *src, long size);
bool cudaMemcpyToHost(void *dst, void *src, long size);
bool deviceMalloc(void **dev_ptr, long size);

TYPE aggregate(void *buf, long size, long itemsize, int Dg, int Db, int Ns);
#endif


