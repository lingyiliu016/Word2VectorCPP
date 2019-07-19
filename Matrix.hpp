//
//  Matrix.hpp
//  Word2Vector
//
//  Created by cly on 10/6/2019.
//  Copyright © 2019 cly. All rights reserved.
//

#ifndef Matrix_hpp
#define Matrix_hpp

#include <stdio.h>
#include <vector>
#include <iostream>

typedef std::vector<std::vector<double> > matrix_type;
typedef std::vector<double> vector_type;

class Matrix
{
public:
    vector_type plus(vector_type v1, vector_type v2);
    matrix_type plus(matrix_type m1, matrix_type m2);
    vector_type minus(vector_type v1, vector_type v2);
    matrix_type minus(matrix_type m1, matrix_type m2);
    double dot_product(vector_type v1, vector_type v2);
    vector_type dot_product(vector_type v, matrix_type m);
    matrix_type dot_product(matrix_type m1, matrix_type m2);
    matrix_type product(double a, matrix_type m);
    matrix_type assign(double m[], unsigned long row, unsigned long column);
    unsigned long get_matrix_row(matrix_type m);
    unsigned long get_matrix_column(matrix_type m);
private:
    
};

#endif /* Matrix_hpp */
