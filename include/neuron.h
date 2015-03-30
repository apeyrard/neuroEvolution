#pragma once

#include <vector>
#include <functional>

#include "functions.h"

namespace neuroev
{

class Neuron
{
public:
    Neuron(std::vector<double> weights, std::function<double(double)> activationFunction=sigmoid);
    ~Neuron();

    double computeOutput(std::vector<double>const &inputs, double* net=nullptr)const;

    std::vector<double> m_weights;
protected:
    std::function<double(double)> m_activationFunction;
};

}
