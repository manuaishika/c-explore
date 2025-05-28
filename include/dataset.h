#ifndef DATASET_H
#define DATASET_H

#include "ml_common.h"
#include "matrix.h"

typedef struct {
    Matrix *features;      
    Matrix *targets;       
    char **feature_names;  
    char **target_names;   
    int n_samples;         
    int n_features;       
    int n_targets;         
} Dataset;

typedef struct {
    double *means;         
    double *stds;          
    double *mins;        
    double *maxs;          
    int n_features;        
    normalize_t type;      
} NormalizationParams;

// Dataset creation and destruction
Dataset* dataset_create(int n_samples, int n_features, int n_targets);
Dataset* dataset_create_from_matrices(Matrix *features, Matrix *targets);
void dataset_free(Dataset *dataset);
Dataset* dataset_copy(const Dataset *src);

// Data loading and saving
Dataset* dataset_load_csv(const char *filename, bool has_header, int target_col);
ml_error_t dataset_save_csv(const Dataset *dataset, const char *filename, bool save_header);
Dataset* dataset_load_libsvm(const char *filename);
ml_error_t dataset_save_libsvm(const Dataset *dataset, const char *filename);

// Data preprocessing
ml_error_t dataset_normalize(Dataset *dataset, normalize_t type, NormalizationParams **params);
ml_error_t dataset_apply_normalization(Dataset *dataset, const NormalizationParams *params);
ml_error_t dataset_standardize(Dataset *dataset, NormalizationParams **params);
ml_error_t dataset_minmax_scale(Dataset *dataset, double min_val, double max_val, NormalizationParams **params);

// Data splitting
ml_error_t dataset_train_test_split(const Dataset *dataset, double test_ratio, 
                                   Dataset **train, Dataset **test, unsigned int seed);
ml_error_t dataset_k_fold_split(const Dataset *dataset, int k, Dataset ***folds);
ml_error_t dataset_stratified_split(const Dataset *dataset, double test_ratio,
                                   Dataset **train, Dataset **test, unsigned int seed);

// Data manipulation
ml_error_t dataset_shuffle(Dataset *dataset, unsigned int seed);
ml_error_t dataset_add_feature(Dataset *dataset, const Matrix *feature, const char *name);
ml_error_t dataset_remove_feature(Dataset *dataset, int feature_idx);
ml_error_t dataset_select_features(const Dataset *dataset, const int *feature_indices, 
                                  int n_selected, Dataset **result);

// Data statistics
void dataset_print_info(const Dataset *dataset);
void dataset_print_statistics(const Dataset *dataset);
ml_error_t dataset_correlation_matrix(const Dataset *dataset, Matrix **correlation);
ml_error_t dataset_feature_importance(const Dataset *dataset, Matrix **importance);

// Data validation
bool dataset_is_valid(const Dataset *dataset);
ml_error_t dataset_check_missing_values(const Dataset *dataset, bool **missing_mask);
ml_error_t dataset_handle_missing_values(Dataset *dataset, const char *strategy);

// Feature engineering
ml_error_t dataset