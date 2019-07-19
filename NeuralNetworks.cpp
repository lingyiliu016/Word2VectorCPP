//
//  NeuralNetworks.cpp
//  Word2Vector
//
//  Created by cly on 10/6/2019.
//  Copyright © 2019 cly. All rights reserved.
//

#include "NeuralNetworks.hpp"

NeuralNetworks::NeuralNetworks(std::string model,
                               unsigned long vocabulary_size,
                               unsigned long distribution_representation_size,
                               double learn_rate) : Matrix()
{
    
    model_type = model;
    one_hot_dimention = vocabulary_size;
    word_meaning_dimention = distribution_representation_size;
    learn_rate = learn_rate;
    
    // 初始化矩阵WI
    std::cout << "初始化矩阵WI" << std::endl;
    init_WI_matrix();
    
    // 初始化矩阵WO
    std::cout << "初始化矩阵WO" << std::endl;
    init_WO_matrix();
}
double NeuralNetworks::forward_propagation(context_target_type input_data[])
{
    // 把预处理的数据j转换成神经网络的输入(X)和正确的输出(Y_truth)
    if (model_type == "CBOW")
    {
        for (int i=0; i < BATCH_SIZE; i++)
        {
            vector_type x(one_hot_dimention,0.0);
            vector_type y_truth(one_hot_dimention,0.0);
            x = input_data[i].context;
            y_truth = input_data[i].target;
            
            X.push_back(x);
            Y_truth.push_back(y_truth);
        }
    }
    else if (model_type == "Skip-Gram")
    {
        for (int i=0; i < BATCH_SIZE; i++)
        {
            vector_type x(one_hot_dimention,0.0);
            vector_type y_truth(one_hot_dimention,0.0);
            x = input_data[i].target;
            y_truth = input_data[i].context;
            
            X.push_back(x);
            Y_truth.push_back(y_truth);
        }
    }
    else
    {
        std::cout << "无效的模型" << std::endl;
        exit(1);
    }
    
    // 隐藏层的输出
    H = dot_product(X, WI);
    
    // 输出层的输出
    U = dot_product(H, WO);
    
    // negtive_sampling
    U_prime = negtive_sampling(U);
    
    // softmax
    Y_output = softmax(U_prime);

    // cross entropy
    loss = cross_entropy(Y_output, Y_truth);
    
    return loss;
}

void NeuralNetworks::back_propagation(void)
{
    // 计算LOSS对U'的偏导数矩阵
    calculate_partial_derivative_matrix_of_LOSS_by_U_prime();
    // 计算U'对U的偏导数矩阵
    calculate_partial_derivative_matrix_of_U_prime_by_U();
    // 计算U对WO的偏导数矩阵
    calculate_partial_derivative_matrix_of_U_by_WO();
    // 计算U对H的偏导数矩阵
    calculate_partial_derivative_matrix_of_U_by_H();
    // 计算H对WI的偏导数矩阵
    calculate_partial_derivative_matrix_of_H_by_WI();
    // 计算LOSS对WO的偏导矩阵
    calculate_partial_derivative_matrix_of_LOSS_by_WO();
    // 计算LOSS对WI的偏导矩阵
    calculate_partial_derivative_matrix_of_LOSS_by_WI();
    
}
void NeuralNetworks::train(context_target_type input_data[])
{
    forward_propagation(input_data);
    back_propagation();
    
    // WO和WI的权重更新
    // WO = WO - learn_rate * partial_derivative_matrix_of_LOSS_by_WO
    WO = minus(WO , product(learn_rate,partial_derivative_matrix_of_LOSS_by_WO));
    // WI = WI - learn_rate * partial_derivative_matrix_of_LOSS_by_WI
    WI = minus(WI, product(learn_rate, partial_derivative_matrix_of_LOSS_by_WI));
    
//    for (int j=0; j < one_hot_dimention; j ++)
//    {
//        std::cout << "partial_derivative_matrix_of_LOSS_by_WO" ;
//        for (int i=0; i < word_meaning_dimention; i++)
//        {
//            std::cout << "[" << j << "]=" << partial_derivative_matrix_of_LOSS_by_WO[i][j] << "\t";
//        }
//        std::cout << std::endl;
//    }
//
//
//    for (int i=0; i < one_hot_dimention; i++)
//    {
//        std::cout << "partial_derivative_matrix_of_LOSS_by_WI[" << i << "]=\t" << partial_derivative_matrix_of_LOSS_by_WI[i] << std::endl;
//    }
    
    // 释放partial_derivative_matrix_of_LOSS_by_WO内存
    partial_derivative_matrix_of_LOSS_by_WO.clear();
    partial_derivative_matrix_of_LOSS_by_WO.shrink_to_fit();
    // 释放partial_derivative_matrix_of_LOSS_by_WI内存
    partial_derivative_matrix_of_LOSS_by_WI.clear();
    partial_derivative_matrix_of_LOSS_by_WI.shrink_to_fit();
}

matrix_type NeuralNetworks::softmax(matrix_type U_prime)
{
    unsigned long column = get_matrix_column(U_prime);
    unsigned long row = get_matrix_row(U_prime);
    
    matrix_type Y_output;
    for (int i=0; i < row; i++)
    {
        vector_type y_output(column, 0.0);
        double sum = 0.0;
        for (int j=0; j < column; j++)
        {
            sum =  sum + constraint(exp(U_prime[i][j]));
        }
        Y_output_sum[i] = constraint(sum);
        for (int j=0; j < column; j++)
        {
            y_output[j] = constraint(exp(U_prime[i][j])) / Y_output_sum[i];
        }
        
        Y_output.push_back(y_output);
    }
    
    return Y_output;
    
}
matrix_type NeuralNetworks::negtive_sampling(matrix_type U)
{
    int word_index = 0;
    // 获取正例的下标列表
    for (int k=0; k < BATCH_SIZE; k++)
    {
        for (int j=0; j < one_hot_dimention; j++)
        {
            if (Y_truth[k][j] == 1)
            {
                word_index = j;
                positive_word_index_list[k] = word_index;
            }
        }
    }
    // 生成负采样矩阵
    std::vector<int> v;
    for(int k=0; k < BATCH_SIZE; k++)
    {
        word_index = positive_word_index_list[k];
        calculate_negtive_sampling_vector_list(word_index, k);
    }
    
    // 进行负采样，得到u_prime
    matrix_type U_prime(BATCH_SIZE);
    for (int k=0; k < BATCH_SIZE; k++)
    {
        int w = 0;
        vector_type v(NEGATIVE_SAMPLING_QUANTITY+1,0.0);
        for (int j=0; j < one_hot_dimention; j++)
        {
            if(negtive_sampling_and_positive_word_vector_list[k][w] == j)
            {
                v[w] = U[k][j];
                w++;
                if(w >= NEGATIVE_SAMPLING_QUANTITY+1)
                {
                    break;
                }
            }
        }
        U_prime[k] = v;
    }
    
    return U_prime;
}

double NeuralNetworks::cross_entropy(matrix_type Y_output, matrix_type Y_truth)
{
    double LOSS = 0.0;
    vector_type V_loss(BATCH_SIZE,0.0);
    int positive_index_in_Y_truth;
    int positive_index_in_Y_output;
    
    for (int k=0; k < BATCH_SIZE; k++)
    {
        positive_index_in_Y_truth = positive_word_index_list[k];
        for (int j=0; j < NEGATIVE_SAMPLING_QUANTITY+1; j++)
        {
            // 获取正例在negative_sampling 后Y_output中d哪一列？即获取uj中j具体是多少。
            if (negtive_sampling_and_positive_word_vector_list[k][j] == positive_index_in_Y_truth)
            {
                positive_index_in_Y_output = j;
            }
        }
        // loss_k = -1 * ln(U[k][j]);
        V_loss[k] = -1 * log(Y_output[k][positive_index_in_Y_output]);
        LOSS = LOSS + V_loss[k];
    }
    v_loss = V_loss;
    return LOSS;
}

void NeuralNetworks::calculate_negtive_sampling_vector_list(int positive_word_index,int row)
{
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::uniform_int_distribution<unsigned> unsign(0,int(one_hot_dimention-1));
    
    // 生成负采样--随机数list，其中NEGATIVE_SAMPLING_QUANTITY个负采样，其中1个正采样;其中k+1个值不相等
    int random_number = 0;
    // 将正采样首先插入negtive_sampling_numerical_list中
    std::vector<int> negtive_sampling_and_positive_word_vector(NEGATIVE_SAMPLING_QUANTITY+1, 0.0);
    std::vector<int> negtive_sampling_vector(NEGATIVE_SAMPLING_QUANTITY, 0.0);
    negtive_sampling_and_positive_word_vector[0] = positive_word_index;
    
    for (int i=1; i < NEGATIVE_SAMPLING_QUANTITY+1; i++)
    {
        // 预先生成一个random_number
        CREATE_RANDOM:
            random_number = unsign(gen);
        // 判断生成的random_number在之前的negtive_sampling_numerical_list中是否存在；如果存在则再生成一次随机数，知道不存在为止
        for (int j=0; j < i; j++)
        {
            if (negtive_sampling_and_positive_word_vector[j] == random_number)
            {
                goto CREATE_RANDOM;
            }
        }
        // 如果该随机数与之前的随机数都不一样，则插入其中
        negtive_sampling_and_positive_word_vector[i] = random_number;
        // 因为negtive_sampling_vector_list中没有正例，索引比i小1
        negtive_sampling_vector[i-1] = random_number;
    }
    negtive_sampling_and_positive_word_vector_list[row] = negtive_sampling_and_positive_word_vector;
    negtive_sampling_vector_list[row] = negtive_sampling_vector;

    // 对负采样按照从小到大进行排序
    sort(negtive_sampling_and_positive_word_vector_list[row].begin(), negtive_sampling_and_positive_word_vector_list[row].end());
    
    // 记录正例在Y_truth中的索引
    positive_word_index_list[row] = positive_word_index;
//    std::cout << "负采样和正例索引列表：" << negtive_sampling_and_positive_word_vector_list[row] << std::endl;
//    std::cout << "正例索引:" << positive_word_index_list[row] << std::endl;

}

void NeuralNetworks::calculate_partial_derivative_matrix_of_LOSS_by_U_prime(void)
{
    matrix_type m(BATCH_SIZE);
    
    int u;
    for (int i=0; i < BATCH_SIZE; i++)
    {
        vector_type v(one_hot_dimention,0.0);
        u = 0;
        for (int j=0; j < one_hot_dimention; j++)
        {
            // 正例的导数
            if(j == positive_word_index_list[i])
            {
                v[j] = (-1 + constraint(exp(U_prime[i][j]))/Y_output_sum[i]) / BATCH_SIZE;
//                std::cout << "v[j] = (-1 + exp(U_prime[i][j])/Y_output_sum[i]) / BATCH_SIZE ="<< v[j] << std::endl;
//                std::cout << "Y_output_sum[" << i << "]=" << Y_output_sum[i] << std::endl;
//                std::cout << "U_prime[" << i << "][" << j << "]=" << U_prime[i][j] << std::endl;
            }
            // 负采样的导数
            else if (u < NEGATIVE_SAMPLING_QUANTITY && j == negtive_sampling_vector_list[i][u])
            {
                v[j] = (constraint(exp(U_prime[i][j]))/Y_output_sum[i]) / BATCH_SIZE;
                u++;
//                std::cout << "(exp(U_prime[i][j])/Y_output_sum[i]) / BATCH_SIZE=" << v[j] << std::endl;
//                std::cout << "Y_output_sum[" << i << "]=" << Y_output_sum[i] << std::endl;
//                std::cout << "U_prime[" << i << "][" << j << "]=" << U_prime[i][j] << std::endl;
            }
            // 其他的导数
            else
            {
                v[j] = 0;
            }
        }
        m[i] = v;
    }
    
    partial_derivative_matrix_of_LOSS_by_U_prime = m;
}
void NeuralNetworks::calculate_partial_derivative_matrix_of_U_prime_by_U(void)
{
    matrix_type m(BATCH_SIZE);
    
    int u;
    // 注意这里的m[i][j]=P_i[j,j]
    for (int i=0; i < BATCH_SIZE; i++)
    {
        u = 0;
        vector_type v(one_hot_dimention,0.0);
        for (int j=0; j < one_hot_dimention; j++)
        {
            if (u < NEGATIVE_SAMPLING_QUANTITY+1 && j == negtive_sampling_and_positive_word_vector_list[i][u])
            {
                v[j] = 1;
                u++;
            }
            else
            {
                v[j] = 0;
            }
        }
        m[i] = v;
    }
    partial_derivative_matrix_of_U_prime_by_U = m;

}
void NeuralNetworks::calculate_partial_derivative_matrix_of_U_by_WO(void)
{
    matrix_type m(word_meaning_dimention);
    
    for (int k=0; k < BATCH_SIZE; k++)
    {
        for (int i=0; i < word_meaning_dimention; i++)
        {
            vector_type v(one_hot_dimention,0.0);
            for (int j=0; j < one_hot_dimention; j++)
            {
                v[j] = H[k][i];
            }
            m[i] = v;
        }
        partial_derivative_matrix_of_U_by_WO[k] = m;
    }
}

void NeuralNetworks::calculate_partial_derivative_matrix_of_U_by_H(void)
{
    matrix_type m(one_hot_dimention);
    vector_type v(word_meaning_dimention,0.0);
    for (int k=0; k < one_hot_dimention; k++)
    {
        for (int i=0; i < BATCH_SIZE; i++)
        {
            for (int j=0; j < word_meaning_dimention; j++)
            {
                v[j] = WO[j][k];
            }
            m[i] = v;
        }
        partial_derivative_matrix_of_U_by_H.push_back(m);
    }
}

void NeuralNetworks::calculate_partial_derivative_matrix_of_H_by_WI(void)
{
    matrix_type m(one_hot_dimention);
    vector_type v(word_meaning_dimention,0.0);
    for (int k=0; k< BATCH_SIZE; k++)
    {
        for (int i=0; i < one_hot_dimention; i++)
        {
            for (int j=0; j< word_meaning_dimention; j++)
            {
                v[j] = X[k][i];
            }
            m[i] = v;
        }
        partial_derivative_matrix_of_H_by_WI[k] = m;
    }
}
void NeuralNetworks::calculate_partial_derivative_matrix_of_LOSS_by_WO(void)
{
    double partial_derivative_LOSS_by_WO;
    vector_type v_partial_derivative_LOSS_by_WO(one_hot_dimention,0.0);

    for (int i=0; i < word_meaning_dimention; i++)
    {
        for (int j=0; j < one_hot_dimention; j++)
        {
            partial_derivative_LOSS_by_WO = 0.0;
            for (int m=0; m < BATCH_SIZE; m++)
            {
                partial_derivative_LOSS_by_WO += partial_derivative_matrix_of_LOSS_by_U_prime[m][j] * partial_derivative_matrix_of_U_prime_by_U[m][j] * partial_derivative_matrix_of_U_by_WO[m][i][j];
            }
            v_partial_derivative_LOSS_by_WO[j] = partial_derivative_LOSS_by_WO;
        }
        partial_derivative_matrix_of_LOSS_by_WO.push_back(v_partial_derivative_LOSS_by_WO);
    }
    
}
void NeuralNetworks::calculate_partial_derivative_matrix_of_LOSS_by_WI(void)
{
    double partial_derivative_LOSS_by_WI;
    vector_type v_partial_derivative_LOSS_by_WI(word_meaning_dimention,0.0);
    
    for (int i=0; i < one_hot_dimention; i++)
    {
        for (int j=0; j < word_meaning_dimention; j++)
        {
            partial_derivative_LOSS_by_WI = 0.0;
            for (int m=0; m < BATCH_SIZE; m++)
            {
                for (int l=0; l < NEGATIVE_SAMPLING_QUANTITY+1; l++)
                {
                    if (partial_derivative_matrix_of_LOSS_by_U_prime[m][l] != 0 && partial_derivative_matrix_of_U_prime_by_U[m][l] != 0)
                    {
                        //                        std::cout << "LOSS对U'的偏导数(" << m << "," << l << ")=" << partial_derivative_matrix_of_LOSS_by_U_prime[m][l] << std::endl;
                        //                        std::cout << "U'对U的偏导数(" << m << "," << l << ")="<< partial_derivative_matrix_of_U_prime_by_U[m][l] << std::endl;
                        //                        std::cout << "U对H的偏导数(" << l << "," << m << "," << j << ")=" << partial_derivative_matrix_of_U_by_H[l][m][j] << std::endl;
                        //                        std::cout << "H对Wd偏导数(" << m << "," << i << "," << j << ")="  << partial_derivative_matrix_of_H_by_WI[m][i][j] << std::endl;
                        
                        partial_derivative_LOSS_by_WI += partial_derivative_matrix_of_LOSS_by_U_prime[m][l] *
                        partial_derivative_matrix_of_U_prime_by_U[m][l] * partial_derivative_matrix_of_U_by_H[l][m][j] * partial_derivative_matrix_of_H_by_WI[m][i][j];
                    }
                }
            }
            v_partial_derivative_LOSS_by_WI[j] = partial_derivative_LOSS_by_WI;
        }
        partial_derivative_matrix_of_LOSS_by_WI.push_back(v_partial_derivative_LOSS_by_WI);
    }
    
}
void NeuralNetworks::init_WI_matrix(void)
{
    unsigned long row = one_hot_dimention;
    unsigned long column = word_meaning_dimention;
    double wi[row * column];
    for (int i=0; i < row; i++)
    {
        for (int j=0; j < column; j++)
        {
            wi[i * column + j] = random(MEAN, STANDARD_DEVIATIONE);
        }
    }
    
    WI = assign(wi, row, column);
}

double NeuralNetworks::constraint(double n)
{
    // n ∈[lower_bound, upper_bound] 或者 [-upper_bound, -lower_bound]
    if ((n >= LOWER_BOUND && n <= UPPER_BOUND) ||  (n >= -1*UPPER_BOUND  && n <= -1*LOWER_BOUND))
    {
        return n;
    }
    // n ∈(upper_bound, ♾)
    else if (n > UPPER_BOUND)
    {
        return UPPER_BOUND;
    }
    // n ∈(-♾,-upper_bound)
    else if (n < -1*UPPER_BOUND)
    {
        return -1*UPPER_BOUND;
    }
    // n ∈(-lower_bound,0)
    else if (n > -1*LOWER_BOUND && n < 0)
    {
        return LOWER_BOUND;
    }
    // n ∈(0,lower_bound)
    else if (n > 0 && n < LOWER_BOUND)
    {
        return LOWER_BOUND;
    }
    // n=0
    else
    {
        return n;
    }
}
void NeuralNetworks::init_WO_matrix(void)
{
    unsigned long row = word_meaning_dimention;
    unsigned long column = one_hot_dimention;
    double wo[row * column];
    for (int i=0; i < row; i++)
    {
        for (int j=0; j < column; j++)
        {
            wo[i * column + j] = random(MEAN, STANDARD_DEVIATIONE);
        }
    }
    
    WO = assign(wo, row, column);
}

double NeuralNetworks::random(double mean, double stdandard_deviation)
{
    double random_number;
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(mean,stdandard_deviation);
    
    random_number = dis(gen);
    
    return random_number;
}

//bool NeuralNetworks::comp(const double &a, const double &b)
//{
//    return a < b;
//}
