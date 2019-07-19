//
//  DataPretreat.hpp
//  Word2Vector
//
//  Created by cly on 8/6/2019.
//  Copyright © 2019 cly. All rights reserved.
//

#ifndef DataPretreat_hpp
#define DataPretreat_hpp

#include <stdio.h>
#include <fstream>
#include <vector>
#include <sys/types.h>
#include <dirent.h>
#include <iostream>
#include <algorithm>
#include <map>
#include <cstdlib>

#include "/Users/cly/github/cppjieba/include/cppjieba/Jieba.hpp"
#include "/Users/cly/github/limonp/include/limonp/Logging.hpp"
#include "Matrix.hpp"

const char* const DICT_PATH = "/Users/cly/github/cppjieba/dict/jieba.dict.utf8";
const char* const HMM_PATH = "/Users/cly/github/cppjieba/dict/hmm_model.utf8";
const char* const USER_DICT_PATH = "/Users/cly/github/cppjieba/dict/user.dict.utf8";
const char* const IDF_PATH = "/Users/cly/github/cppjieba/dict/idf.utf8";
const char* const STOP_WORD_PATH = "/Users/cly/github/cppjieba/dict/stop_words.utf8";



struct context_target_type
{
    std::string context_word;
    std::string target_word;
    std::vector<double> context;
    std::vector<double> target;
};



class DataPretreat:public cppjieba::Jieba
{
public:
    DataPretreat(std::string model, int windows_size);
    void random_shuffle(void);
    void save_data(matrix_type WI);
    matrix_type load_data(int distribution_representation_size);
    
    std::vector<std::string> split_to_sentence(std::string filename);
    // filenames_list的每个元素是一个文件名
    std::vector<std::string> filenames_list;
    // sentences_list的每个元素是一句话的内容
    std::vector<std::string> sentences_list;
    // words_segment_list的第一层元素是一句话，而第二层元素是该句话的分词
    std::vector<std::vector<std::string>> words_segment_list;
    // vocabulary_list的每个元素是一个单词
    std::vector<std::string> vocabulary_list;
    // one_hot_encode_list的键：单词，值：one_hot编码
    std::map<std::string,std::vector<double>> one_hot_encode_list;
    // input_output_list的元素(input,output),其中input和output均为one_hot编码格式
    std::vector<context_target_type> input_output_list;
    
private:
    void get_filename(void);
    std::string& trim(std::string &s);
    std::vector<std::string> split(const std::string &s,std::vector<std::string> seperator);
    void get_vocabulary(void);
    void one_hot_encoding(void);
    void cbow_pretreat(int windows_size);
    void skip_gram_pretreat(int windows_size);
    void swap(context_target_type &a, context_target_type &b);
    vector_type string_to_vector(std::string str, int vector_size);
    std::vector<double> plus(const std::vector<double> v1, const std::vector<double> v2);
    std::vector<double> divide(const std::vector<double> dividend, const int divisor);
    const char* data_path = "/Users/cly/Desktop/Study/code/project/Word2vec/Word2Vector/Word2Vector/data/";
    unsigned long vocabulary_size;
    
};

#endif /* DataPretreat_hpp */
