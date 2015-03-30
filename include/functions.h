#pragma once

#include <functional>
#include <vector>

namespace neuroev
{

template <typename T> std::vector<T> vectorize(const std::function<T(T,T)> &func, const std::vector<T> &inputs, const std::vector<T> &targets)
{
    std::vector<T> outputs;
    for (int i = 0; i < inputs.size(); ++i)
    {
        outputs.push_back(func(inputs[i], targets[i]));
    }
    return outputs;
}

double sigmoid(double x);

double sigmoid_prime(double x);

double mean_square_derivative(double output, double target);

double identity(double x);

double dRand(double dMin, double dMax);

}
