//
//  Matrix.cpp
//  Word2Vector
//
//  Created by cly on 10/6/2019.
//  Copyright © 2019 cly. All rights reserved.
//

#include "Matrix.hpp"

vector_type Matrix::plus(vector_type v1, vector_type v2)
{
    if(v1.size() != v2.size())
    {
        std::cout << "v1的维度与v2的维度不相等，无法进行加法计算" << std::endl;
        exit(1);
    }
    
    vector_type sum(v1.size(),0.0);
    for (int i=0; i < v1.size(); i++)
    {
        sum[i] = v1[i] + v2[i];
    }
    return sum;
}

matrix_type Matrix::plus(matrix_type m1, matrix_type m2)
{
    unsigned long row1 = get_matrix_row(m1);
    unsigned long column1 = get_matrix_column(m1);
    unsigned long row2 = get_matrix_row(m2);
    unsigned long column2 = get_matrix_column(m2);
    
    if (row1 != row2 || column1 != column2)
    {
        std::cout << "矩阵m1与矩阵m2的形状不相同。m1行数：" << row1 << "\t列数：" << column1 << "\t m2行数：" << row2 << "\t 列数：" << column2 << std::endl;
    }
    
    matrix_type sum;
    for (int i=0; i < row1; i++)
    {
        vector_type v(column1, 0.0);
        for(int j=0; j < column1; j++)
        {
            v[j] = m1[i][j] + m2[i][j];
        }
        sum.push_back(v);
    }
    
    return sum;
}

vector_type Matrix::minus(vector_type v1, vector_type v2)
{
    if(v1.size() != v2.size())
    {
        std::cout << "v1的维度与v2的维度不相等，无法进行加法计算" << std::endl;
        exit(1);
    }
    
    vector_type difference(v1.size(),0.0);
    for (int i=0; i < v1.size(); i++)
    {
        difference[i] = v1[i] - v2[i];
    }
    return difference;
}

matrix_type Matrix::minus(matrix_type m1, matrix_type m2)
{
    unsigned long row1 = get_matrix_row(m1);
    unsigned long column1 = get_matrix_column(m1);
    unsigned long row2 = get_matrix_row(m2);
    unsigned long column2 = get_matrix_column(m2);
    
    if (row1 != row2 || column1 != column2)
    {
        std::cout << "矩阵m1与矩阵m2的形状不相同。m1行数：" << row1 << "\t列数：" << column1 << "\t m2行数：" << row2 << "\t 列数：" << column2 << std::endl;
    }
    
    matrix_type difference;
    for (int i=0; i < row1; i++)
    {
        vector_type v(column1, 0.0);
        for(int j=0; j < column1; j++)
        {
            v[j] = m1[i][j] - m2[i][j];
        }
        difference.push_back(v);
    }
    
    return difference;
}

double Matrix::dot_product(vector_type v1, vector_type v2)
{
    if(v1.size() != v2.size())
    {
        std::cout << "v1的维度与v2的维度不相等，无法进行乘法计算" << std::endl;
        exit(1);
    }
    
    double product = 0.0;
    
    for (int i=0; i < v1.size(); i++)
    {
        product = product + v1[i] * v2[i];
    }
    
    return product;
}

vector_type Matrix::dot_product(vector_type v, matrix_type m)
{
    unsigned long column = get_matrix_column(m);
    unsigned long row = get_matrix_row(m);
    if(v.size() != row)
    {
        std::cout << "v的维度与矩阵的行数不相等，无法进行乘法计算" << std::endl;
        exit(1);
    }
    vector_type product(column, 0.0);
    for(int j=0; j < column; j++)
    {
        for(int i=0; i < row; i++)
        {
            product[j] = product[j] + v[i]*m[i][j];
        }
    }
    
    return product;
}

matrix_type Matrix::dot_product(matrix_type m1, matrix_type m2)
{
    unsigned long m1_row = get_matrix_row(m1);
    unsigned long m1_column = get_matrix_column(m1);
    unsigned long m2_row = get_matrix_row(m2);
    unsigned long m2_column = get_matrix_column(m2);
    
    if (m1_column != m2_row)
    {
        std::cout << "矩阵m1的列数与矩阵m2的行数不相等，无法进行乘法运算" << std::endl;
        exit(1);
    }
    matrix_type product(m1_row);
    vector_type v(m2_column,0.0);
    for (int i=0; i < m1_row; i++)
    {
        v = dot_product(m1[i], m2);
        product[i] = v;
    }
    
    return product;
}

matrix_type Matrix::product(double a, matrix_type m)
{
    matrix_type prod;
    unsigned long row = get_matrix_row(m);
    unsigned long column = get_matrix_column(m);
    
    vector_type v(column,0.0);
    for (int i=0; i < row; i++)
    {
        for (int j=0; j < column; j++)
        {
            v[j] = a * m[i][j];
        }
        prod.push_back(v);
    }
    
    return prod;
}

matrix_type Matrix::assign(double m[], unsigned long row, unsigned long column)
{
    matrix_type matrix;
    for (int i=0; i < row; i++)
    {
        vector_type v(column, 0.0);
        for (int j=0; j < column; j++)
        {
            v[j] = m[i * column + j];
        }
        matrix.push_back(v);
    }
    return matrix;
}

unsigned long Matrix::get_matrix_row(matrix_type m)
{
    unsigned long row = 0;
    row = m.size();
    return row;
}
unsigned long Matrix::get_matrix_column(matrix_type m)
{
    unsigned long column = 0;
    column = m[0].size();
    return column;
}
