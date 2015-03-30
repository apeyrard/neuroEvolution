#include "neuron.h"

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
    double sum = 0.0f;
    for (unsigned int i = 0; i < inputs.size(); ++i)
    {
        sum += inputs[i]*m_weights[i];
    }
    if (net)
    {
        *net = sum;
    }
    return m_activationFunction(sum);
}

}
