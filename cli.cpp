/* 
 * File:   CLI.cpp
 * Author: jdelatorre
 * 
 * Created on 23 de diciembre de 2014, 14:53
 */

#include <cstdlib>  // system()
#include <string>
#include <sstream>
#include <stdexcept>      // std::invalid_argument
#include <iostream>

#include "CL/cl.hpp"

#include "cli.hpp"
#include "nn.hpp"


cli::~cli() {
}

void cli::set(std::istringstream & is, const std::string & cmd) {
    if (neural_network.isTraining()) {
        std::cout << "Error: NN training.\n";
        std::cout << "       Use <pause> or <stop> before using <set> command\n";
        return;
    }
    
    bool error = false;
    
    std::string token;
    is >> token;
    
    if (token == "lr" || token == "momentum") {
        std::string val;
        is >> val;
        cl_float v = 0.0f;
        try {
            v = std::stof(val);
        } catch(const std::invalid_argument & ia) {
            error = true;
        }
        if (v < 0.0f || v > 1.0f) error = true;
        
        if (!error) {
            if (token == "lr") {
                neural_network.setLR(v);
            } else {
                neural_network.setM(v);
            }
        } else {
          std::cout << "Error: Not valid value. Should be between 0.0 and 1.0\n";
        }
    } else if (token == "nag") {    // set NAG
        TODO_msg(cmd);
    } else if (token == "rule") {   // set a new rule
        TODO_msg(cmd);
    } else if (token == "nn") {  // set a NN architecture
        TODO_msg(cmd);
    } else {
        unknown_command_msg(cmd);
    }
}

void cli::load(std::istringstream & is, const std::string & cmd) {
    std::string what, filepath;
    
    is >> what;
    is >> filepath;  // should be "/file/path"
    if (filepath[0] == '\"' && filepath[filepath.size()-1] == '\"') {
        filepath = filepath.substr(1, filepath.size()-2);
    }
    
    if (!filepath.empty()) {
        if (what == "trainingset") {
            TODO_msg(cmd);
            return;
        } else if (what == "testset") {
            TODO_msg(cmd);
            return;
        } else if (what == "nn") {
            neural_network.load_NN(filepath);
            return;
        }
    }
    // if no return before => command error
    unknown_command_msg(cmd);
}

void cli::save(std::istringstream & is, const std::string & cmd) {
    std::string what, filepath;
    
    is >> what;
    is >> filepath;  // should be "/file/path"
    if (filepath[0] == '\"' && filepath[filepath.size()-1] == '\"') {
        filepath = filepath.substr(1, filepath.size()-2);
    }
    
    if (!filepath.empty()) {
        if (what == "nn") {
            neural_network.save_NN(filepath);
            return;
        }
    }
    // if no return before => command error
    unknown_command_msg(cmd);
}

void cli::train(std::istringstream & is, const std::string & cmd) {
    std::string what;
    
    is >> what;
    
    if (what == "run") {
        if (neural_network.isTraining()) {
            std::cout << "NN already running" << "\n";
        } else {
            std::thread t(&nn::train, &neural_network);
            t.detach();
        }
    } else if (what == "pause" || what == "stop") {
        if (neural_network.isTraining()) {
            neural_network.stopTrain();
            std::cout << "Stopping training...\n";
            while (neural_network.isTraining());
            std::cout << "Training Stopped\n";
        }
    } else {
        unknown_command_msg(cmd);
    }
}

void cli::plot() {
    
}

void cli::loop() {
    std::string token, cmd;
    
    std::cout << "Neural network command line interface v0.0:\n";
    std::cout << ">> ";
    do {
        if (!getline(std::cin, cmd))  // Block here waiting for input
            cmd = "quit";
        
        std::istringstream is(cmd);

        token.clear();  // getline() could return empty or blank line
        is >> std::skipws >> token;

        if (token == "quit") {
            if (neural_network.isTraining()) {
                neural_network.stopTrain();
                std::cout << "Stopping training...\n";
                while (neural_network.isTraining());
                std::cout << "Training Stopped\n";
            }
            break;
        } else if (token == "set") {
            set(is, cmd);
        } else if (token == "load") {
            load(is, cmd);
        } else if (token == "save") {
            save(is, cmd);
        } else if (token == "train") {
            train(is, cmd);
        } else if (token == "plot") {
            plot();
        } else {
            unknown_command_msg(cmd);
        }
        std::cout << ">> ";
    } while (token != "quit");

}
