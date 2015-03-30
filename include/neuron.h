#pragma once

#include <vector>
#include <functional>

class Neuron
{
public:
    Neuron(std::function<double(double)> activationFunction, std::vector<double> weights);
    ~Neuron();

    double computeOutput(std::vector<double>const &inputs, double* net=nullptr)const;

    std::vector<double> m_weights;
protected:
    std::function<double(double)> m_activationFunction;
};
