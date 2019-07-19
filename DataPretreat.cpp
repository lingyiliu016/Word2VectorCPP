//
//  DataPretreat.cpp
//  Word2Vector
//
//  Created by cly on 8/6/2019.
//  Copyright © 2019 cly. All rights reserved.
//

#include "DataPretreat.hpp"

DataPretreat::DataPretreat(std::string model, int windows_size) : cppjieba::Jieba(DICT_PATH,
                                                                                  HMM_PATH,
                                                                                  USER_DICT_PATH,
                                                                                  IDF_PATH,
                                                                                  STOP_WORD_PATH)
{
    std::vector<std::string> temp_sentences;
    std::vector<std::string> temp_words_segment_list;
    int i,j;
    
    std::cout << "DataPretreat开始" << std::endl;
    // 获取data下面的所有filenames
    get_filename();
    std::cout << "获取文件名成功！" << std::endl;
    
    // 获取所有的sentences
    for(i = 0; i < filenames_list.size(); i++)
    {
        temp_sentences = split_to_sentence(filenames_list[i]);
        for(j = 0; j < temp_sentences.size(); j++)
        {
            sentences_list.push_back(temp_sentences[j]);
        }
    }
    std::cout << "获取所有的sentence成功！" << std::endl;
    
    // 获取所有的words_segment
    for (i = 0; i < sentences_list.size(); i++)
    {
        Cut(sentences_list[i],temp_words_segment_list,true);
        words_segment_list.push_back(temp_words_segment_list);
    }
    std::cout << "对所有的句子分词成功！" << std::endl;
    
    // 获取所有的单词
    get_vocabulary();
    std::cout << "获取词汇成功！" << std::endl;
    
    // one_hot 编码
    one_hot_encoding();
    std::cout << "对每一个单词进行one-hot编码成功！" << std::endl;
    
    // 模式选择,CBOW or Skip-Gram
    if (model == "CBOW")
    {
        cbow_pretreat(windows_size);
        std::cout << "以CBOW的方式处理数据结束！" << std::endl;
    }
    else if(model == "Skip-Gram")
    {
        skip_gram_pretreat(windows_size);
        std::cout << "以Skip-Gram的方式处理数据结束！" << std::endl;
    }
    else
    {
        std::cout << "无效的模型" << std::endl;
        exit(1);
    }
}
void DataPretreat::random_shuffle(void)
{
    srand(time(NULL));
    
    for (int i=0; i < input_output_list.size(); i++)
    {
        swap(input_output_list[i], input_output_list[rand()%(i+1)]);
    }
}
void DataPretreat::save_data(matrix_type WI)
{
    std::string data_path = "/Users/cly/Desktop/Study/code/project/Word2vec/Word2Vector/Word2Vector/word2vector.data";
    
    std::ofstream outfile(data_path);
    std::string word;
    
    if(outfile.is_open() == false)
    {
        std::cout << "文件打开失败！" << std::endl;
    }
    
    for (int i=0; i < WI.size(); i++)
    {
        word = vocabulary_list[i];
        outfile << word << "\t" << WI[i] << std::endl;
    }
    
    outfile.close();
    
}
matrix_type DataPretreat::load_data(int distribution_representation_size)
{
    std::string data_path = "/Users/cly/Desktop/Study/code/project/Word2vec/Word2Vector/Word2Vector/word2vector.data";
    std::ifstream infile(data_path);
    std::string temp;
    
    if (infile.is_open() == false)
    {
        std::cout << "文件打开失败！" << std::endl;
    }
    if (infile.eof())
    {
        std::cout << "文件为空！" << std::endl;
    }
    std::vector<std::string> word_and_vector;
    std::vector<std::string> seperator = {"\t"};
    
    vector_type v(distribution_representation_size, 0.0);
    vector_type zero(distribution_representation_size, 0.0);
    matrix_type WI;
    
    while (!infile.eof())
    {
        getline(infile, temp);
        word_and_vector = split(temp,seperator);
        v = string_to_vector(word_and_vector[1], distribution_representation_size);
        // 因为加载文件”最后一行“单词为空，向量为[0, 0, ..., 0],要舍去
        if (v != zero)
        {
            WI.push_back(v);
            vocabulary_list.push_back(word_and_vector[0]);
        }
        
    }
    
    infile.close();
    return WI;
}
void DataPretreat::get_filename(void)
{
    DIR *path = opendir(data_path);
    if (path == NULL)
    {
        perror("opendir error");
        exit(1);
    }
    struct dirent* entry;
    while ((entry=readdir(path)) != NULL)
    {
        if (std::string(entry->d_name) != std::string(".") && std::string(entry->d_name) != std::string(".."))
        {
            //            puts(entry->d_name);
            filenames_list.push_back(std::string(data_path) + std::string(entry->d_name));
        }
        
    }
    closedir(path);
}

std::vector<std::string> DataPretreat::split_to_sentence(std::string filename)
{
    std::string line, temp_file;
    std::ifstream infile;
    std::vector<std::string> sentences_list;
    
    infile.open(filename);
    if (infile.is_open() != true)
    {
        std::cout << "文件:" << filename << "\t打开失败"<< std::endl;
    }
    // 去掉line的首尾的空格、制表符、换行符等，然后再拼接
    while(getline(infile,line))
    {
        line = trim(line);
        temp_file = temp_file + line;
    }
    // 按照合适的分隔符对temp_file进行分割，分割后，每个单元为一个整句子。
    std::vector<std::string> seperator = {"？”",
        "！”",
        "。”",
        "?”",
        "!”",
        ".“",
        "。",
        "！",
        "？",
        ".",
        "!",
        "?"
        "……"};
    sentences_list = split(temp_file,seperator);
    
    return sentences_list;
}

std::string& DataPretreat::trim(std::string &s)
{
    std::string whitespaces ("　 \t\f\v\n\r");
    
    if(s.empty())
    {
        return s;
    }
    
    s.erase(0,s.find_first_not_of(whitespaces));
    s.erase(s.find_last_not_of(whitespaces)+1);
    
    return s;
}

std::vector<std::string> DataPretreat::split(const std::string &s,std::vector<std::string> seperator)
{
    std::vector<std::string> result;
    typedef std::string::size_type string_size;
    std::string str = s;
    string_size head = 0,tail = 0;
    
    
    // 将最后一个分隔符左边的部分压入result中
    while(tail != str.size())
    {
        for(string_size i = 0; i < seperator.size(); ++i)
        {
            // 如果当前的字符与分隔符中的任意一个相同，则考虑将该分隔符之前的一小段压入result中
            if(str.substr(tail,seperator[i].length()) == seperator[i])
            {
                if (head != tail)
                {
                    result.push_back(str.substr(head, tail-head+seperator[i].length()));
                }
                head = tail+seperator[i].length();
                // 只从seperator中选一个分隔符。
                break;
            }
        }
        tail++;
    }
    // 将最后一个分隔符右边的部分压入result中
    result.push_back(str.substr(head));
    return result;
}

void DataPretreat::get_vocabulary(void)
{
    std::vector<std::string>::iterator iter;
    for(int i = 0; i < words_segment_list.size(); i++)
    {
        for(int j= 0; j < words_segment_list[i].size(); j++)
        {
            iter = std::find(vocabulary_list.begin(),vocabulary_list.end(),words_segment_list[i][j]);
            // 如果words_segment_list[i][j]在vocabulary_list中不存在的话，就插入
            if (iter == vocabulary_list.end())
            {
                vocabulary_list.push_back(words_segment_list[i][j]);
            }
        }
    }
}

void DataPretreat::one_hot_encoding(void)
{
    vocabulary_size = vocabulary_list.size();
    
    for(int i = 0; i < vocabulary_size; i++)
    {
        // 键：word
        std::string word = vocabulary_list[i];
        // 值：one_hot
        std::vector<double> one_hot(vocabulary_size, 0.0);
        one_hot[i] = 1.0;
        one_hot_encode_list.insert(std::pair<std::string,std::vector<double>>(word,one_hot));
    }
    
}

void DataPretreat::cbow_pretreat(int windows_size)
{
    for(int i=0; i < words_segment_list.size(); i++)
    {
//        std::cout << "words_segment_list[" << i << "]===" << words_segment_list[i] << std::endl;
        for(int j=0; j < words_segment_list[i].size(); j++)
        {
            std::string target_word = words_segment_list[i][j];
            std::string context_word;
            std::vector<double> context(vocabulary_size,0.0);
            std::vector<double> target(vocabulary_size,0.0);
            context_target_type context_target;
            
            int divisor = 0;
            target = one_hot_encode_list[target_word];
            for(int k=-windows_size; k <= windows_size; k++)
            {
                // context 不为target，不越下界，不越上界
                if(k != 0 && j+k >= 0 && j+k < words_segment_list[i].size())
                {
                    context_word = words_segment_list[i][j+k];
                    context = plus(context, one_hot_encode_list[context_word]);
                    context_target.context_word = context_target.context_word + "||" + context_word;
                    divisor++;
                }
            }
            // context是所有windows_size中上下文的平均值
            context = divide(context, divisor);
            // (context, target)
            context_target.target_word = target_word;
            context_target.context = context;
            context_target.target  = target;
//            std::cout << "context_word:" << context_target.context_word << std::endl;
//            std::cout << "context:" << context_target.context << std::endl;
//            std::cout << "target_word:" << context_target.target_word << std::endl;
//            std::cout << "target:" << context_target.target << std::endl;
            // input_output_list = ((context1, target1),(context2, target2),...)
            input_output_list.push_back(context_target);
        }
    }
}
void DataPretreat::skip_gram_pretreat(int windows_size)
{
    for(int i=0; i < words_segment_list.size(); i++)
    {
//        std::cout << "words_segment_list[" << i << "]===" << words_segment_list[i] << std::endl;
        for(int j=0; j < words_segment_list[i].size(); j++)
        {
            std::string target_word = words_segment_list[i][j];
            std::vector<double> target(vocabulary_size,0.0);
            
            target = one_hot_encode_list[target_word];
            
            for(int k=-windows_size; k <= windows_size; k++)
            {
                // context 不为target，不越下界，不越上界
                if(k != 0 && j+k >= 0 && j+k < words_segment_list[i].size())
                {
                    std::vector<double> context(vocabulary_size,0.0);
                    std::string context_word;
                    context_target_type context_target;
                    
                    context_word = words_segment_list[i][j+k];
                    context = one_hot_encode_list[context_word];
                    
                    // (target, context)
                    context_target.context_word = context_word;
                    context_target.target_word = target_word;
                    context_target.target  = target;
                    context_target.context = context;
//                    std::cout << "context_word:" << context_target.context_word << std::endl;
//                    std::cout << "context:" << context_target.context << std::endl;
//                    std::cout << "target_word:" << context_target.target_word << std::endl;
//                    std::cout << "target:" << context_target.target << std::endl;
                    
                    // input_output_list = ((target1, context1),(target2, context2),...)
                    input_output_list.push_back(context_target);
                }
            }
        }
    }
}

void DataPretreat::swap(context_target_type &a, context_target_type &b)
{
    context_target_type c;
    c = a;
    a = b;
    b = c;
}

vector_type DataPretreat::string_to_vector(std::string str, int vector_size)
{
    vector_type v(vector_size, 0.0);
    str.erase(std::remove(str.begin(),str.end(),'['),str.end());
    str.erase(std::remove(str.begin(),str.end(),']'),str.end());
    int i = 0;
    double element;
    std::string temp_element;
    std::string::size_type pos1 = 0;
    std::string::size_type pos2 = str.find(",");
    
    while(pos2 != std::string::npos)
    {
        temp_element =str.substr(pos1, pos2-pos1);
        element = atof(temp_element.c_str());
        v[i] = element;
        pos1 = pos2 + 2;
        pos2 = str.find(",",pos1);
        i++;
        if (pos2 == std::string::npos)
        {
            v[i] = atof(str.substr(pos1, str.length()).c_str());
        }
    }
    
    return v;
}

std::vector<double> DataPretreat::plus(const std::vector<double> v1, const std::vector<double> v2)
{
    std::vector<double> sum(vocabulary_size, 0.0);
    for(int i=0; i < vocabulary_size; i++)
    {
        sum[i] = v1[i] + v2[i];
    }
    
    return sum;
}
std::vector<double> DataPretreat::divide(const std::vector<double> dividend, const int divisor)
{
    std::vector<double> quotient(vocabulary_size, 0.0);
    if(divisor == 0)
    {
        return dividend;
    }
    else
    {
        for(int i=0; i < vocabulary_size; i++)
        {
            quotient[i] = dividend[i] / divisor;
        }
        
        return quotient;
    }
    
}

