/*
 * File:   main.cpp
 * Author: jordi
 *
 * Created on 9 de octubre de 2014, 21:24
 */

#include <vector>
#include <cstdlib>
#include <string>

#include "nn.hpp"
#include "cli.hpp"


int main(int argc, char** argv) {
    const std::string train_file = "train-images.idx3-ubyte";
    const std::string train_labels_file = "train-labels.idx1-ubyte";
    const std::string test_file = "t10k-images.idx3-ubyte";
    const std::string test_labels_file = "t10k-labels.idx1-ubyte";
    
    nn nn1;
    
    // cli CLI(nn1);
    
    // load nn structure
    std::vector<cl_uint> neuralnet = {784, 1024, 1024, 16};
    nn1.load_NN(neuralnet);
    
    // load training and test data
    nn1.load_MNIST_train_and_test_DATA(
        train_file,
        train_labels_file,
        test_file,
        test_labels_file);
    
    nn1.populate_normal_random_weights();
    
    nn1.init_training();
    //nn1.test_matrix_multiplication();
    // CLI.loop();
    
    nn1.train();
    
    nn1.save_NN("neural.nn");
    
    return 0;
}

