#include "program.h"

void run_iteration(int block_size, global_params_t *params,
        dataset_t *data, global_state_t *state, agg_res_t *result) {
}

void setup_dataset(dataset_t *data, global_params_t *params) {
    int i;
    cudaMallocManaged(&data->points, data->num_points * sizeof(float*));
    for (i = 0; i < data->num_points; i++)
        cudaMallocManaged(&data->points[i], params->dims * sizeof(float));
}

void setup_global_state(global_state_t *state, global_params_t *params) {
    int i;
    cudaMallocManaged(
        &state->centroids, params->num_centroids * sizeof(float*));
    for (i = 0; i < params->num_centroids; i++)
        cudaMallocManaged(&state->centroids[i], params->dims * sizeof(float));
}

void setup_aggregation_result(agg_res_t *result, global_params_t *params) {
    int i;
    cudaMallocManaged(
        &result->centroid_updates, params->num_centroids * sizeof(float*));
    for (i = 0; i < params->num_centroids; i++)
        cudaMallocManaged(
            &result->centroid_updates[i], params->dims * sizeof(float));
    cudaMallocManaged(
        &result->update_counts, params->num_centroids * sizeof(int));
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
