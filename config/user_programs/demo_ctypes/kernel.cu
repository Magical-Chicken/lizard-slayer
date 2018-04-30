#include "program.h"

/**
 * NOTE: in a real program this would call into the cuda kernel
 *       however, this is not a real program and does nothing interesting
 */
void run_iteration(int blocks, int block_size, global_params_t *params,
        dataset_t *data, global_state_t *state, agg_res_t *result) {
    int i, j;
    for (i = 0; i < params->dims; i++) result->values[i] = 0;
    for (i = 0; i < data->num_points; i++)
        for (j = 0; j < params->dims; j++)
            result->values[j] += data->points[i][j];
}

void setup_dataset(dataset_t *data, global_params_t *params) {
    int i;
    cudaMallocManaged(&data->points, data->num_points * sizeof(float*));
    for (i = 0; i < data->num_points; i++)
        cudaMallocManaged(&data->points[i], params->dims * sizeof(float));
}

void setup_global_state(global_state_t *state, global_params_t *params) {
    cudaMallocManaged(&state->values, params->dims * sizeof(float));
    state->iteration = 0;
}

void setup_aggregation_result(agg_res_t *result, global_params_t *params) {
    cudaMallocManaged(&result->values, params->dims * sizeof(float));
}

void free_dataset(dataset_t *data, global_params_t *params) {
    int i;
    for (i = 0; i < data->num_points; i++)
        cudaFree(data->points[i]);
    cudaFree(data->points);
}

void free_global_state(global_state_t *state, global_params_t *params) {
    cudaFree(state->values);
}

void free_aggregation_result(agg_res_t *result, global_params_t *params) {
    cudaFree(result->values);
}
