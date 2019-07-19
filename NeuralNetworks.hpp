//
//  NeuralNetworks.hpp
//  Word2Vector
//
//  Created by cly on 10/6/2019.
//  Copyright © 2019 cly. All rights reserved.
//

#ifndef NeuralNetworks_hpp
#define NeuralNetworks_hpp

#include <stdio.h>
#include <chrono>
#include <random>
#include <math.h>
#include "Matrix.hpp"
#include "DataPretreat.hpp"

#define BATCH_SIZE 32
#define MEAN 0
#define STANDARD_DEVIATIONE 1
#define NEGATIVE_SAMPLING_QUANTITY 5
#define UPPER_BOUND 10000
#define LOWER_BOUND 0.0001

class NeuralNetworks:public Matrix
{
public:
    NeuralNetworks(std::string model,
                   unsigned long vocabulary_size,
                   unsigned long distribution_representation_size,
                   double learn_rate);
    double forward_propagation(context_target_type input_data[]);
    void back_propagation(void);
    void train(context_target_type input_data[]);
    matrix_type WI;
    double loss;
    
private:
    matrix_type softmax(matrix_type U);
    matrix_type negtive_sampling(matrix_type U);
    double cross_entropy(matrix_type Y_output, matrix_type Y_truth);
    void calculate_negtive_sampling_vector_list(int positive_word_index,int row);
    void calculate_partial_derivative_matrix_of_LOSS_by_U_prime(void);
    void calculate_partial_derivative_matrix_of_U_prime_by_U(void);
    void calculate_partial_derivative_matrix_of_U_by_WO(void);
    void calculate_partial_derivative_matrix_of_U_by_H(void);
    void calculate_partial_derivative_matrix_of_H_by_WI(void);
    void calculate_partial_derivative_matrix_of_LOSS_by_WO(void);
    void calculate_partial_derivative_matrix_of_LOSS_by_WI(void);
    double constraint(double n);
    void init_WI_matrix(void);
    void init_WO_matrix(void);
    double random(double mean, double stdard_deviation);
    matrix_type WO;
    matrix_type X;
    matrix_type Y_truth;
    matrix_type Y_output;
    matrix_type H;
    matrix_type U;
    matrix_type U_prime;
    vector_type v_loss;
//    bool first_or_not;
    double learn_rate;
    std::string model_type;
    unsigned long one_hot_dimention;
    unsigned long word_meaning_dimention;
    std::vector<int> negtive_sampling_vector_list[BATCH_SIZE];
    int positive_word_index_list[BATCH_SIZE];
    double Y_output_sum[BATCH_SIZE];
    std::vector<int> negtive_sampling_and_positive_word_vector_list[BATCH_SIZE];
    matrix_type partial_derivative_matrix_of_LOSS_by_U_prime;
    matrix_type partial_derivative_matrix_of_U_prime_by_U;
    matrix_type partial_derivative_matrix_of_U_by_WO[BATCH_SIZE];
    std::vector<matrix_type> partial_derivative_matrix_of_U_by_H;
    matrix_type partial_derivative_matrix_of_H_by_WI[BATCH_SIZE];
    matrix_type partial_derivative_matrix_of_LOSS_by_WO;
    matrix_type partial_derivative_matrix_of_LOSS_by_WI;
};

#endif /* NeuralNetworks_hpp */
