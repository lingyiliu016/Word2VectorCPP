//
//  main.cpp
//  Word2Vector
//
//  Created by cly on 8/6/2019.
//  Copyright © 2019 cly. All rights reserved.
//

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <cmath>

#include "DataPretreat.hpp"
#include "Matrix.hpp"
#include "NeuralNetworks.hpp"

using namespace std;

int main(int argc, const char * argv[]) {
    string model_type;
    double learn_rate;
    double vocabulary_size;
    int distribution_representation_size;
    int windows_size;
    int epoch;
    unsigned long train_data_size;
    context_target_type input_data[BATCH_SIZE];

    cout << "请选择 Word2Vec类型，CBOW 还是Skip-Gram?" << endl;
    cin >> model_type;
    cout << "请输入训练模型的学习率。比如：0.0001" << endl;
    cin >> learn_rate;
    cout << "请输入词向量的维度" << endl;
    cin >> distribution_representation_size;
    cout << "请输入上下文大小" << endl;
    cin >> windows_size;
    cout << "请输入epoch" << endl;
    cin >> epoch;

    DataPretreat PrepareData(model_type, windows_size);
    vocabulary_size = PrepareData.vocabulary_list.size();
    train_data_size = PrepareData.input_output_list.size();
    cout << "词汇量=" << vocabulary_size << "\t 训练集大小=" << train_data_size << endl;

    NeuralNetworks word2vec(model_type,vocabulary_size,distribution_representation_size,learn_rate);

    for (int i=0; i < epoch; i++)
    {

        PrepareData.random_shuffle();
        for (int j=0; j < train_data_size/BATCH_SIZE; j++)
        {
            for (int k=0; k < BATCH_SIZE; k++)
            {
                input_data[k] = PrepareData.input_output_list[j*BATCH_SIZE+k];
            }
            // 训练一次网络
            word2vec.train(input_data);
            cout << "epoch=" << i << "\t loss=" << word2vec.loss << endl;
        }
    }
    // 保存词向量信息至文件中
    PrepareData.save_data(word2vec.WI);
    word2vec.WI = PrepareData.load_data(distribution_representation_size);
    
//    vector<double> U = {};
//    vector<int> Sample = {185, 252, 505, 865, 1108, 1419};
//    vector<double> U_prime;
//    int j = 0;
//    for (int i=0; i < U.size(); i++)
//    {
//        if (j < Sample.size() && i == Sample[j])
//        {
//            U_prime.push_back(U[i]);
//            j++;
//        }
//
//    }
//    std::cout << "U_prime:" << U_prime << endl;
//    vector<double> Y_output(Sample.size(),0.0);
//    double sum = 0;
//    for (int i=0; i < 6; i++)
//    {
//        sum = sum + exp(U_prime[i]);
//        cout << "U_prime[" << i << "]=" << U_prime[i] << "\t exp(u)=" << exp(U_prime[i]) << endl;
//    }
//    cout << "sum=" << sum << endl;
//    for (int i=0; i < 6; i++)
//    {
//        Y_output[i] = exp(U_prime[i]) / sum;
//    }
//    cout << "Y_output=" << Y_output << endl;
//    int postive_index = 0;
//    double loss;
//    loss = -1 * log(Y_output[postive_index]);
//    cout << "loss=" << loss << endl;
//    loss = -1 * U_prime[postive_index] + log(sum);
//    cout << "loss=" << loss << endl;
//
//
//    Matrix matrix;
//    vector_type h = matrix.plus(WI1, WI2);
//
//
//    for (int i=0; i < h.size(); i++)
//    {
//        cout << h[i]/2 << "\t";
//    }
    

    return 0;
}
