#ifndef _EMPTY_PROG
#define _EMPTY_PROG

typedef struct gobal_state {
    int iteration, *values;
} global_state_t;

typedef struct aggregation_result {
    int *values;
} agg_res_t;

typedef struct dataset {
    int **points;
} dataset_t;

typedef struct dataset_params {
    int dims, num_points;
} dataset_params_t;

extern "C" {
    void run_iteration(int blocks, int block_size,
        dataset_t *data, global_state_t *state, agg_res_t *result);
    void setup_dataset(dataset_t *data, dataset_params_t *params);
    void setup_global_state(global_state_t *state, dataset_params_t *params);
    void setup_aggregation_result(agg_res_t *result, dataset_params_t *params);
    void free_dataset(dataset_t *data, dataset_params_t *params);
    void free_global_state(global_state_t *state, dataset_params_t *params);
    void free_aggregation_result(agg_res_t *result, dataset_params_t *params);
}

#endif
