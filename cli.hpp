/* 
 * File:   CLI.hpp
 * Author: jdelatorre
 *
 * Created on 23 de diciembre de 2014, 14:53
 */

#ifndef CLI_HPP
#define CLI_HPP

#include <thread>
#include <string>
#include <iostream>
#include "nn.hpp"

class cli {
 public:
    inline explicit cli(nn & nn_v) : neural_network(nn_v) {}
    virtual ~cli();
    
    void loop();
 private:
    nn & neural_network;
    
    void set(std::istringstream & is, const std::string & cmd);
    void load(std::istringstream & is, const std::string & cmd);
    void save(std::istringstream & is, const std::string & cmd);
    void train(std::istringstream & is, const std::string & cmd);
    void plot();
    
    inline void unknown_command_msg(const std::string & cmd) {
        std::cout << "Unknown command: " << cmd << std::endl;
    };

    inline void TODO_msg(const std::string & cmd) {
        std::cout << "TODO option: " << cmd << std::endl;
    };
       
};

#endif /* CLI_HPP */

