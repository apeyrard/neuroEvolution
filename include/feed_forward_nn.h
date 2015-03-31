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
    using layer = std::vector<Neuron>;
    using vectord = std::vector<double>;

public:
    Feed_forward_nn(std::vector<int> sizeList, std::function<double(double)> activation=sigmoid, int bias = 1);

    ~Feed_forward_nn();

    std::vector<Array<double, Dynamic, Dynamic> > GetWeights() const;
    void SetWeights(const std::vector<Array<double, Dynamic, Dynamic> > &weights);

    vectord FeedForward(vectord inputs, std::vector<vectord >* history=nullptr);

    void back_propagate(vectord inputs, vectord targets, double eta=0.3, double momentum=0.9, std::function<double(double, double)> cost_derivative=mean_square_derivative, std::function<double(double)> activation_prime=sigmoid_prime);

private:
    int m_bias; //nb of biases usually 1
    std::vector<Array<double, Dynamic, Dynamic> > m_oldChange;
    std::vector<layer> m_vecLayers;

    layer create_layer(std::function<double(double)> activation,
        int nbNeurons,
        int nbWeights);
};

}
