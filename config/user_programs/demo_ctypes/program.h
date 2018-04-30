#ifndef _EMPTY_PROG
#define _EMPTY_PROG

typedef struct gobal_state {
    int done, iteration, *values;
} global_state_t;

typedef struct aggregation_result {
    int *values;
} agg_res_t;

typedef struct dataset {
    int num_points;
    int **points;
} dataset_t;

typedef struct global_params {
    int dims, max_iterations;
} global_params_t;

extern "C" {
    void run_iteration(int blocks, int block_size, global_params_t *params,
        dataset_t *data, global_state_t *state, agg_res_t *result);
    void setup_dataset(dataset_t *data, global_params_t *params);
    void setup_global_state(global_state_t *state, global_params_t *params);
    void setup_aggregation_result(agg_res_t *result, global_params_t *params);
    void free_dataset(dataset_t *data, global_params_t *params);
    void free_global_state(global_state_t *state, global_params_t *params);
    void free_aggregation_result(agg_res_t *result, global_params_t *params);
}

#endif
