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

bool comp(const double &a, const double &b)
{
    return a < b;
}
int test(int argc, const char * argv[]) {
    
    
    //    Matrix matrix;
    //    // 测试一
    //    vector_type v1 = {1,2,3,4,5};
    //    vector_type v2 = {6,7,8,9,0};
    //    vector_type v = matrix.plus(v1, v2);
    //    cout << "v1+v2=" << v << endl;
    //    // 测试二
    //    double m1[15] = {1,2,3,4,5,6,7,8,9,0,1,2,3,4,5};
    //    int row1 = 5;
    //    int column1 = 3;
    //    double m2[15] = {2,3,4,5,6,7,8,9,0,1,2,3,4,5,6};
    //    int row2 = 5;
    //    int column2 = 3;
    //    matrix_type M1 = matrix.assign(m1, row1, column1);
    //    matrix_type M2 = matrix.assign(m2, row2, column2);
    //    matrix_type M = matrix.plus(M1, M2);
    //    cout << "M1+M2=" << endl;
    //    for (int i=0; i < row1; i++)
    //    {
    //        cout << M[i] << endl;
    //    }
    //    // 测试三
    //    v = matrix.minus(v1, v2);
    //    cout << "v1-v2=" << v << endl;
    //    // 测试四
    //    M = matrix.minus(M1, M2);
    //    cout << "M1-M2=" << endl;
    //    for (int i=0; i < row1; i++)
    //    {
    //        cout << M[i] << endl;
    //    }
    //    // 测试五
    //    double p1 = matrix.dot_product(v1, v2);
    //    cout << "v1*v2=" << p1 << endl;
    //    // 测试六
    //    vector_type p2 = matrix.dot_product(v1, M1);
    //    cout << "v1*M1=" << p2 << endl;
    //    // 测试七
    //    int row=3;
    //    int column=4;
    //    double m[12] ={1,2,3,4,5,6,7,8,9,0,1,2};
    //    M = matrix.assign(m, row, column);
    //    cout << "将m赋值给M:" << endl;
    //    for (int i=0; i < row; i++)
    //    {
    //        cout << M[i] << endl;
    //    }
    //    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    //    std::default_random_engine gen(seed);
    //    std::normal_distribution<double> dis(0,1);
    //    std::cout << "正态分布随机数:" << endl;
    //
    //    for (int i=0; i < 10; i++)
    //    {
    //        std::cout << dis(gen) << endl;
    //    }
    //    NeuralNetworks word2vector("CBOW",32,8,10);
    //    word2vector.forward_propagation(<#context_target_type *input_data#>)
    
    std::default_random_engine gen(0);
    std::uniform_int_distribution<unsigned> unsign(0,10);
    
    for(int i=0; i < 100; i++)
    {
        cout << unsign(gen) << endl;
    }
    
    
    return 0;
}

