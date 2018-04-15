#ifndef SHARED_H__
#define SHARED_H__
typedef double type;
void *cudaMemcpyToDevice(void *dst, void *src, long size, long item_size);

type aggregate(void *buf, long size, long itemsize, int Dg, int Db, int Ns);
#endif


