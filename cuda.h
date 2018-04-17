#ifndef SHARED_H__
#define SHARED_H__

#define TYPE double

void *cudaMemcpyToDevice(void *dst, void *src, long size, long item_size);

TYPE aggregate(void *buf, long size, long itemsize, int Dg, int Db, int Ns);
#endif


