#include "scan.h"
#include <string.h>

int get_num_gpus() {
    int devs;
    cudaGetDeviceCount(&devs);
    return devs;
}

void get_gpu_data(int gpu_index, struct gpu_props *props) {
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, gpu_index);
    props->comp_level_major = p.major;
    props->comp_level_minor = p.minor;
    props->gpu_index = gpu_index;
    props->max_sm_threads = p.maxThreadsPerMultiProcessor;
    props->max_block_size = p.maxThreadsPerBlock;
    props->sm_count = p.multiProcessorCount;
    strncpy(props->name, p.name, 256);
    // NOTE: this logic is brittle and may break with future CUDA versions
    if (p.major > 5) props->max_sm_blocks = 32;
    else if (p.major > 3) props->max_sm_blocks = 16;
    else props->max_sm_blocks = 8;
    props->max_total_threads = props->max_sm_threads * props->sm_count;
    props->max_total_blocks = props->max_sm_blocks * props->sm_count;
}
