#include "neuron.h"

#include <iostream>

namespace neuroev
{

Neuron::Neuron(std::vector<double> weights, std::function<double(double)> activationFunction)
    : m_weights(weights)
    , m_activationFunction(activationFunction)
{
}

Neuron::~Neuron()
{
}

double Neuron::computeOutput(std::vector<double>const &inputs, double* net) const
{
    //std::cout << "Neuron called" << std::endl;
    double sum = 0.0f;
    for (unsigned int i = 0; i < inputs.size(); ++i)
    {
        //std::cout << "i : "<<inputs[i]<<" w :"<<m_weights[i]<<std::endl;
        sum += inputs[i]*m_weights[i];
        //std::cout << "sum : "<<sum<<std::endl;
    }
    //std::cout << "out : "<<m_activationFunction(sum)<<std::endl;
    return m_activationFunction(sum);
}

}
