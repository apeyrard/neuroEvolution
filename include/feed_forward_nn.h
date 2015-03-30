#pragma once

#include <stdlib.h>
#include <functional>
#include <vector>
#include <eigen3/Eigen/Core>
#include "neuron.h"

class Feed_forward_nn
{
    static const int Dynamic = Eigen::Dynamic;
    using RowArrayXd = Eigen::Array<double, 1, Dynamic>;
    template<typename T, int T1, int T2> using Array = Eigen::Array<T, T1, T2>;
    using uint = unsigned int;
    using layer = std::vector<Neuron>;
    using vectord = std::vector<double>;

    double randDouble(double min, double max)
    {
        double random = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
        random *= (max - min);
        random += min;
        return random;
    }

    layer create_layer(std::function<double(double)> activation,
        uint nbNeurons,
        uint nbWeights)
    {
        layer l;
        for (uint i=0; i < nbNeurons; i++)
        {
            //random weights
            vectord weights;
            for (uint j = 0; j < nbWeights; j++)
            {
                weights.push_back(randDouble(-1.0f,1.0f));
            }

            Neuron n(activation, weights);
            l.push_back(n);
        }
        return l;
    }

public:
    Feed_forward_nn(std::function<double(double)> activation, std::vector<uint> sizeList);

    ~Feed_forward_nn();

    std::vector<Array<double, Dynamic, Dynamic> > GetWeights() const;
    void SetWeights(std::vector<Array<double, Dynamic, Dynamic> > &weights);

    vectord FeedForward(vectord &inputs, std::vector<vectord >* history=nullptr, std::vector<vectord >* netList=nullptr);

    std::vector<Array<double, Dynamic, Dynamic> > back_propagate(vectord inputs, vectord targets, std::function<vectord(vectord, vectord)> cost_derivative, std::function<double(double)> activation_prime);

private:
    std::vector<layer> m_vecLayers;
};
