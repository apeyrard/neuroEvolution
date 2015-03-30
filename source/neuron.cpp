#include "neuron.h"

#include <iostream>

Neuron::Neuron(std::function<double(double)> activationFunction, std::vector<double> weights)
    : m_weights(weights)
    , m_activationFunction(activationFunction)
{

}

Neuron::~Neuron()
{

}

double Neuron::computeOutput(std::vector<double>const &inputs, double* net) const
{
    //std::cout << std::endl << "Neuron activated" << std::endl;
    double sum = 0.0f;
    for (unsigned int i = 0; i < inputs.size(); ++i)
    {
        //std::cout << "Input : " << inputs[i] << " Weight : "<< m_weights[i] << std::endl;
        sum += inputs[i]*m_weights[i];
    }
    if (net)
    {
        *net = sum;
    }
    //std::cout << "Sum : " << sum << std::endl;
    //std::cout << "Output : " << m_activationFunction(sum) << std::endl;
    return m_activationFunction(sum);
}
