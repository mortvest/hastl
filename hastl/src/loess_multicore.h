// Generated by Futhark 0.22.0
// git: e8b7244 (Wed Jan 12 23:26:03 2022 +0100)
#pragma once


// Headers\n")
#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdio.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

// Initialisation
struct futhark_context_config;
struct futhark_context_config *futhark_context_config_new(void);
void futhark_context_config_free(struct futhark_context_config *cfg);
void futhark_context_config_set_debugging(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_profiling(struct futhark_context_config *cfg,
                                          int flag);
void futhark_context_config_set_logging(struct futhark_context_config *cfg,
                                        int flag);
void futhark_context_config_set_num_threads(struct futhark_context_config *cfg,
                                            int n);
struct futhark_context;
struct futhark_context *futhark_context_new(struct futhark_context_config *cfg);
void futhark_context_free(struct futhark_context *ctx);
int futhark_context_sync(struct futhark_context *ctx);
int futhark_context_config_set_tuning_param(struct futhark_context_config *cfg,
                                            const char *param_name,
                                            size_t param_value);
int futhark_get_tuning_param_count(void);
const char *futhark_get_tuning_param_name(int);
const char *futhark_get_tuning_param_class(int);

// Arrays
struct futhark_f64_2d;
struct futhark_f64_2d *futhark_new_f64_2d(struct futhark_context *ctx, const
                                          double *data, int64_t dim0,
                                          int64_t dim1);
struct futhark_f64_2d *futhark_new_raw_f64_2d(struct futhark_context *ctx, const
                                              unsigned char *data,
                                              int64_t offset, int64_t dim0,
                                              int64_t dim1);
int futhark_free_f64_2d(struct futhark_context *ctx,
                        struct futhark_f64_2d *arr);
int futhark_values_f64_2d(struct futhark_context *ctx,
                          struct futhark_f64_2d *arr, double *data);
unsigned char *futhark_values_raw_f64_2d(struct futhark_context *ctx,
                                         struct futhark_f64_2d *arr);
const int64_t *futhark_shape_f64_2d(struct futhark_context *ctx,
                                    struct futhark_f64_2d *arr);

// Opaque values


// Entry points
int futhark_entry_main(struct futhark_context *ctx,
                       struct futhark_f64_2d **out0, const
                       struct futhark_f64_2d *in0, const int64_t in1, const
                       int64_t in2, const int64_t in3, const int64_t in4, const
                       int64_t in5);

// Miscellaneous
char *futhark_context_report(struct futhark_context *ctx);
char *futhark_context_get_error(struct futhark_context *ctx);
void futhark_context_set_logging_file(struct futhark_context *ctx, FILE *f);
void futhark_context_pause_profiling(struct futhark_context *ctx);
void futhark_context_unpause_profiling(struct futhark_context *ctx);
int futhark_context_clear_caches(struct futhark_context *ctx);
#define FUTHARK_BACKEND_multicore

#ifdef __cplusplus
}
#endif
