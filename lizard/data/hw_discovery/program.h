#ifndef _SCAN_H
#define _SCAN_H

struct gpu_props {
    int gpu_index, comp_level_major, comp_level_minor,
        sm_count, max_sm_threads, max_sm_blocks, max_block_size,
        max_total_threads, max_total_blocks;
    char name[256];
};

extern "C" int get_num_gpus();

extern "C" void get_gpu_data(int gpu_index, struct gpu_props *props);

#endif
