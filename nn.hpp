
#ifndef NN_HPP_
#define NN_HPP_

#define __CL_ENABLE_EXCEPTIONS  // enable use of exceptions of the OpenCL API

#include <CL/cl.hpp>

#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

#include "common.hpp"
#include "mg.hpp"
#include "OpenCLKernels.hpp"

class nn {
    bool neuralNetworkDefined = false;
    bool trainDataLoaded = false;
    bool testDataLoaded = false;
    
    bool trainRunning = false;
    
    bool stopTraining = false;
    
    const cl_uint BUFFER_ERROR_SIZE = 2*1048576;
    
    cl_uint numberOfNeurons;
    cl_uint numberOfWeights;
    cl_uint numberOfTrainingData;
    cl_uint numberOfTestData;
    cl_uint numberOfLayers;

    bool enableMomentumRule = false;
    // enableNAG true uses Nesterov-accelerated gradient.
    // enableNAG false uses Classical Momentum
    bool enableNAG = false;
 
    bool enableL2Regularization = false;
    
    cl_uint epoch = 0;  // epoch of training
    cl_float ce = 0.0;
    cl_float ce_test = 0.0;
    
    cl_uint minibatchSize = 128;
    cl_float learningRate = 0.1f;  // Typìcal value 0.3
    cl_float momentum = 0.9f;      // Typical value 0.9
    size_t maxEpochs = 100000;      // Typical value 5000000
    cl_float minError = 0.001f;     // Typical value 0.01
    cl_float lambda = 1.0f;     // L2 reg. param. (0, 1 , 10, etc.)
    
    size_t printEpochs = 100;      // Typical value 1000
    
    std::vector<cl_uint> elementsPerLayer;
    
    // Whole training data set
    std::vector<cl_float> training_data;
    std::vector<cl_float> training_data_output;
    
    // activations of all the neurons for all the training data for one epoch
    std::vector<cl_float> activations_host;
    // activations of all the neurons for all the test data for one epoch
    std::vector<cl_float> activations_test_host;
    // bias
    std::vector<cl_float> bias_host;
    // weights of all neurons
    std::vector<cl_float> weights_host;
    // last weight increment calculated from back propagation
    std::vector<cl_float> increment_weights_host;
    // last bias increment calculated from back propagation
    // std::vector<cl_float> increment_bias_host;
    // deltas of all activation layers
    std::vector<cl_float> deltas_host;
    // output values of the training data
    std::vector<cl_float> t_host;
    // output values of the test data
    std::vector<cl_float> t_test_host;
    // vector required for the host side calculation of the cross entropy
    // after first reduce in device
    std::vector<cl_float> buffer_error_host;
    
    // offsets required for finding activation values over the vector
    std::vector<cl_uint> activations_offsets;
    std::vector<cl_uint> activations_test_offsets;
    std::vector<cl_uint> weights_offsets;
    std::vector<cl_uint> bias_offsets;
    std::vector<cl_uint> deltas_offsets;
    
    // random vectors
    std::vector<cl_uchar> mask_weights_host;
    std::vector<cl_uchar> mask_bias_host;
    
    std::vector<cl_uchar> s0_host;
    std::vector<cl_uchar> s1_host;
      
    // classes for mapping the host memory with the device memory
    host_device_memory_map<cl_float> activations;
    host_device_memory_map<cl_float> activations_test;
    host_device_memory_map<cl_float> bias;
    // inputs and calculated activations
    host_device_memory_map<cl_float> weights;  // all the weights of the NN
    host_device_memory_map<cl_float> increment_weights;  // all the inc weights of the NN
    // host_device_memory_map<cl_float> increment_bias;  // all the inc bias of the NN
    host_device_memory_map<cl_float> deltas;   // delta errors (Backprop)
    host_device_memory_map<cl_float> t;        // real output value
    host_device_memory_map<cl_float> t_test;        // real output value
    host_device_memory_map<cl_float> buffer_error;  // real output value
    
    // host_device_memory_map<cl_uint> minibatch_idx;
    
    host_device_memory_map<cl_uchar> mask_weights;
    host_device_memory_map<cl_uchar> mask_bias;
    host_device_memory_map<cl_uchar> s0;  // random seed/status 0
    host_device_memory_map<cl_uchar> s1;  // random seed/status 1
    
    cl::Context *context;   // unique OpenCL context
    std::vector<cl::Device> devices;
    cl::CommandQueue *queue;   // unique OpenCL command queue;

    OpenCLKernels *openclKernels;
        
    /*
     * Momentum update rule extracted from "On the importance of
     * initialization and momentum in deep learning",
     * Hinton et al. 2013.
     * According to this paper:
     * momentum_max is chosen between 0.999, 0.995, 0.99, 0.9 and 0
     * learning rate is chosen between 0.05, 0.01, 0.005, 0.001, 0.0005, 0.0001
     */
    inline void update_momentum_rule_Hinton2013(cl_uint t) {
        const cl_float momentum_max = 0.9;
        // Values used: 0.999, 0.995, 0.99, 0.9, 0
        const cl_float new_momentum =
            1.0f - std::pow(2.0f, -1.0f -
                             std::log2(t / 250.0f + 1.0f));
        momentum = std::min(momentum_max, new_momentum);
    }
    
    void update_lr_rule_Dropconnect2013(cl_uint t);

    void print_results_data_header_with_L2_regularization();
    void print_results_data_with_L2_regularization(
                            cl_float ce1,
                            cl_float ce2,
                            cl_float ce,
                            cl_float ce1_test,
                            cl_float ce2_test,
                            cl_float ce_test);

    void print_results_data_header();
    void print_results_data(cl_float ce,
                            cl_float ce_test);
    void print_data();
    
    // Nesterov Accelerated Gradient functions
    void NAG_preupdate();
    void NAG_postupdate();
    
    // OpenCL initialization
    void opencl_init();
    
    // allocation functions

    void allocate_NN_memory_on_host();
    void allocate_DATA_memory_on_host();
    void calculate_offsets();
    void allocate_memory_on_device();
    void load_data_to_device();
    
    void FF(host_device_memory_map<cl_float> &act,
            std::vector<cl_uint> &off,
            cl_uint rows,
            bool average = false);

    cl_float percentage_classification_results(
            host_device_memory_map<cl_float> &act,
            std::vector<cl_uint> &act_off,
            host_device_memory_map<cl_float> &out,
            cl_uint rows);
    
    // Cross Entropy Error Function Calculation
    cl_float CE(
            host_device_memory_map<cl_float> &act,
            std::vector<cl_uint> &off,
            host_device_memory_map<cl_float> &out,
            cl_uint rows);
    
    void seed_random_status();
    void generate_masks();
    
 public:
    
    nn();
    ~nn();

    
    // void test_dropout();
    
    inline bool isTraining() { return trainRunning; }
    inline void stopTrain() { stopTraining = true; }
    
    inline void setLR(cl_float lr) { learningRate = lr; }
    inline void setM(cl_float m) { momentum = m; }
    
    void populate_normal_sparse_weights(const cl_float mean = 0.0f,
                                        const cl_float stddev = 0.1f,
                                        const cl_uint initElementsPerLayer = 15);
    
    void populate_normal_random_weights(cl_float mean = 0.0f,
                                        cl_float stddev = 0.1f);
    
    
    inline void load_float_vector(std::string filename,
                                  std::vector<cl_float> & v) {
        load_csv_vector(filename, v);
    }

    inline void save_float_vector(std::string filename,
                                  std::vector<cl_float> & v) {
        save_csv_vector(filename, v);
    }
    
    inline void FF_train() {
        FF(activations, activations_offsets, minibatchSize);
    }
    inline void FF_test() {
        FF(activations_test, activations_test_offsets, numberOfTestData, true);
    }

    inline cl_float percentage_classification_results_train() {
        return percentage_classification_results(
                activations,
                activations_offsets,
                t,
                minibatchSize);
    }

    inline cl_float percentage_classification_results_test() {
        return percentage_classification_results(
                activations_test,
                activations_test_offsets,
                t_test,
                numberOfTestData);
    }
    
    inline cl_float CE_train() {
        return CE(
                activations,
                activations_offsets,
                t,
                minibatchSize);
    }

    inline cl_float CE_test() {
        return CE(
                activations_test,
                activations_test_offsets,
                t_test,
                numberOfTestData);
    }

    cl_float L2_regularization();
    
    void BP();  // Backpropagation calculation (all sigmoid))
    void WA();  // weight actualization
    
    void train();   // Training for all sigmoid + output softmax
    
    void save_NN(const std::string filename);
    void load_NN(const std::string filename);
    
    inline void load_NN(std::vector<cl_uint> elemPerLayer) {
        numberOfLayers = elemPerLayer.size();
        elementsPerLayer.resize(numberOfLayers);
        for (size_t i = 0; i < elemPerLayer.size(); i++)
            elementsPerLayer[i] = elemPerLayer[i];
        allocate_NN_memory_on_host();
        
        neuralNetworkDefined = true;
    }
    // Classification neural network (all sigmoid except last layer -> softmax)
    
    void load_MNIST_train_and_test_DATA(
             const std::string &train_file,
             const std::string &train_labels_file,
             const std::string &test_file,
             const std::string &test_labels_file);
    
    inline void init_training() {
        allocate_DATA_memory_on_host();
        calculate_offsets();
        seed_random_status();   // dropconnect
        allocate_memory_on_device();
        load_data_to_device();
    }
    
    void test_matrix_multiplication();    
};

#endif
