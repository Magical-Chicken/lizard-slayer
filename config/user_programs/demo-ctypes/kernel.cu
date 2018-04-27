#include "program.h"

void setup_dataset(dataset_t *data, global_params_t *params) {
    int i;
    cudaMallocManaged(&data->points, data->num_points * sizeof(int*));
    for (i = 0; i < data->num_points; i++)
        cudaMallocManaged(&data->points[i], params->dims * sizeof(int));
}

void setup_global_state(global_state_t *state, global_params_t *params) {
    int i;
    cudaMallocManaged(&state->values, params->dims * sizeof(int));
    for (i = 0; i < params->dims; i++) state->values[i] = 0;
    state->iteration = 0;
}

void setup_aggregation_result(agg_res_t *result, global_params_t *params) {
    cudaMallocManaged(&result->values, params->dims * sizeof(int));
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
