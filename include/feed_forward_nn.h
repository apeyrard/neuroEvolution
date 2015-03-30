#pragma once

#include <stdlib.h>
#include <functional>
#include <vector>
#include <eigen3/Eigen/Core>

#include "neuron.h"
#include "functions.h"

namespace neuroev
{

class Feed_forward_nn
{
    static const int Dynamic = Eigen::Dynamic;
    using RowArrayXd = Eigen::Array<double, 1, Dynamic>;
    template<typename T, int T1, int T2> using Array = Eigen::Array<T, T1, T2>;
    using uint = unsigned int;
    using layer = std::vector<Neuron>;
    using vectord = std::vector<double>;

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
                weights.push_back(dRand(-1.0f,1.0f));
            }

            Neuron n(weights, activation);
            l.push_back(n);
        }
        return l;
    }

public:
    Feed_forward_nn(std::vector<uint> sizeList, std::function<double(double)> activation=sigmoid);

    ~Feed_forward_nn();

    std::vector<Array<double, Dynamic, Dynamic> > GetWeights() const;
    void SetWeights(std::vector<Array<double, Dynamic, Dynamic> > &weights);

    vectord FeedForward(vectord &inputs, std::vector<vectord >* history=nullptr, std::vector<vectord >* netList=nullptr);

    std::vector<Array<double, Dynamic, Dynamic> > back_propagate(vectord inputs, vectord targets, std::function<double(double, double)> cost_derivative=mean_square_derivative, std::function<double(double)> activation_prime=sigmoid_prime);

private:
    std::vector<layer> m_vecLayers;
};

}
