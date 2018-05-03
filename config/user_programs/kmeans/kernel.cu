#include "program.h"
#include <math.h>
#include <stdio.h>

void check_cuda_err(cudaError_t err) {
    if (err != cudaSuccess)
        printf("Error: %s\n", cudaGetErrorString(err));
}

__device__ float distance(
        global_params_t *params, float *point1, float *point2) {
    int i;
    float val, accum = 0;
    for (i = 0; i < params->dims; i++) {
        // Faster than using pow()
        val = point2[i] - point1[i];
        accum += val * val;
    }
    return sqrt(accum);
}

__device__ int nearest_centroid(
        global_params_t *params, global_state_t *state, float *point) {
    int i, min_idx;
    float min_val, cur_val;
    // iterate through all centroids and determine which is the nearest point
    for (i = 0, min_val = -1; i < params->num_centroids; i++) {
        cur_val = distance(params, point, state->centroids[i]);
        if (cur_val < min_val || min_val == -1) {
            min_val = cur_val;
            min_idx = i;
        }
    }
    return min_idx;
}

__global__ void kernel(
        global_params_t params, dataset_t data, global_state_t state,
        agg_res_t result) {
    extern __shared__ char shared_start[];
    int i, idx, centroid_idx, *update_counts;
    float *updates;
    /*result.centroid_updates[0][0] = params.num_centroids;*/
    /*result.update_counts[0] = params.dims;*/
    // calculate address for updates and update counts
    update_counts = (int *)shared_start;
    updates = (float *)(shared_start + params.num_centroids * sizeof(int));
    // zero out update counters
    if ((idx = threadIdx.x) < params.num_centroids) {
        update_counts[idx] = 0;
        for (i = 0; i < params.dims; i++)
            updates[idx * params.dims + i] = 0;
    }
    // synchronize threads
    __syncthreads();
    // get index of datapoint to operate on
    idx = blockIdx.x * blockDim.x + threadIdx.x;
    // if datapoint is past end of data return
    if (idx >= data.num_points) return;
    // find nearest centroid
    centroid_idx = nearest_centroid(&params, &state, data.points[idx]);
    // add to update counts and update values atomically in shared memory
    atomicAdd(&update_counts[centroid_idx], 1);
    for (i = 0; i < params.dims; i++)
        atomicAdd(
            &updates[centroid_idx * params.dims + i], data.points[idx][i]);
    // synchronize threads
    __syncthreads();
    // add to global updates
    if ((idx = threadIdx.x) < params.num_centroids) {
        atomicAdd(&result.update_counts[idx], update_counts[idx]);
        for (i = 0; i < params.dims; i++)
            atomicAdd(
                &result.centroid_updates[idx][i],
                updates[idx * params.dims + i]);
    }
}

void run_iteration(
        int block_size, global_params_t *params,
        dataset_t *data, global_state_t *state, agg_res_t *result) {
    int blocks;
    size_t shared_size;
    // calculate number of blocks to run for the dataset
    blocks = data->num_points / block_size;
    if (data->num_points % block_size) blocks++;
    // calculate shared allocation size for update_counts and updates
    shared_size = params->num_centroids * sizeof(int);
    shared_size += params->num_centroids * params->dims * sizeof(float);
    // run kernel
    kernel<<<blocks, block_size, shared_size>>>(
        *params, *data, *state, *result);
    check_cuda_err(cudaPeekAtLastError());
    cudaDeviceSynchronize();
}

void setup_dataset(dataset_t *data, global_params_t *params) {
    int i;
    check_cuda_err(
        cudaMallocManaged(&data->points, data->num_points * sizeof(float*)));
    for (i = 0; i < data->num_points; i++)
        check_cuda_err(
            cudaMallocManaged(&data->points[i], params->dims * sizeof(float)));
}

void setup_global_state(global_state_t *state, global_params_t *params) {
    int i;
    check_cuda_err(cudaMallocManaged(
        &state->centroids, params->num_centroids * sizeof(float*)));
    for (i = 0; i < params->num_centroids; i++)
        check_cuda_err(cudaMallocManaged(
            &state->centroids[i], params->dims * sizeof(float)));
}

void setup_aggregation_result(agg_res_t *result, global_params_t *params) {
    int i;
    check_cuda_err(cudaMallocManaged(
        &result->centroid_updates, params->num_centroids * sizeof(float*)));
    for (i = 0; i < params->num_centroids; i++)
        check_cuda_err(cudaMallocManaged(
            &result->centroid_updates[i], params->dims * sizeof(float)));
    check_cuda_err(cudaMallocManaged(
        &result->update_counts, params->num_centroids * sizeof(int)));
}

void free_dataset(dataset_t *data, global_params_t *params) {
    int i;
    for (i = 0; i < data->num_points; i++)
        cudaFree(data->points[i]);
    cudaFree(data->points);
}

void free_global_state(global_state_t *state, global_params_t *params) {
    int i;
    for (i = 0; i < params->num_centroids; i++)
        cudaFree(state->centroids[i]);
    cudaFree(state->centroids);
}

void free_aggregation_result(agg_res_t *result, global_params_t *params) {
    int i;
    for (i = 0; i < params->num_centroids; i++)
        cudaFree(result->centroid_updates[i]);
    cudaFree(result->centroid_updates);
    cudaFree(result->update_counts);
}
