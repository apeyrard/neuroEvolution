#include "feed_forward_nn.h"

#include <iostream>

namespace neuroev{

Feed_forward_nn::layer Feed_forward_nn::create_layer(std::function<double(double)> activation,
    int nbNeurons,
    int nbWeights)
{
    layer l;
    for (int i=0; i < nbNeurons; i++)
    {
        //random weights
        vectord weights;
        for (int j = 0; j < nbWeights; j++)
        {
            weights.push_back(dRand(-1.0,1.0));
        }

        Neuron n(weights, activation);
        l.push_back(n);
    }
    return l;
}

Feed_forward_nn::Feed_forward_nn(std::vector<int> vecNbNeurons, std::function<double(double)> activation, int bias)
    : m_bias(bias)
{
    for (uint i = 1; i < vecNbNeurons.size();i++)
    {   std::cout<<"Creating layer " << i << " n : " << vecNbNeurons[i] << " i : " << vecNbNeurons[i-1] + m_bias << std::endl;
        m_vecLayers.push_back( create_layer(activation, vecNbNeurons[i], vecNbNeurons[i-1] + bias ) );
        Array<double, Dynamic, Dynamic> a(vecNbNeurons[i], vecNbNeurons[i-1] + bias);
        a.setZero();
        m_oldChange.push_back(a);
    }

}

Feed_forward_nn::~Feed_forward_nn()
{
}

std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > Feed_forward_nn::GetWeights() const
{
    std::vector<Array<double, Dynamic, Dynamic> > result;
    for (uint i = 0; i < m_vecLayers.size();i++)
    {
        int rows = m_vecLayers[i].size();
        int cols = m_vecLayers[i][0].m_weights.size();
        Array<double, Dynamic, Dynamic> layerWeights(rows, cols);
        for (int j = 0; j < m_vecLayers[i].size(); j++)
        {
            for (int k = 0; k < m_vecLayers[i][j].m_weights.size();k++)
            {
                    layerWeights(j, k) = m_vecLayers[i][j].m_weights[k];
            }
        }
        result.push_back(layerWeights);
    }
    return result;
}

//TODO verify size before
void Feed_forward_nn::SetWeights(const std::vector<Array<double, Dynamic, Dynamic> > &weights)
{
    for (int i = 0; i < m_vecLayers.size();i++)
    {
        for (int j = 0; j < m_vecLayers[i].size(); j++)
        {
            for (int k = 0; k < m_vecLayers[i][j].m_weights.size();k++)
            {
                //std::cout << weights[i](j,k) << std::endl;
                m_vecLayers[i][j].m_weights[k] = weights[i](j, k);
            }
        }
    }
}

Feed_forward_nn::vectord Feed_forward_nn::FeedForward(vectord inputs, std::vector<vectord >* history)
{
    if(history)
        history->push_back(inputs);

    vectord outputs;

    for (int i = 0; i < m_vecLayers.size(); ++i)
    {
        outputs.clear();
        for (int j = 0; j < m_bias; ++j)
            inputs.push_back(1.0); //biases

        for (int j = 0; j < m_vecLayers[i].size(); ++j)
        {
            outputs.push_back(m_vecLayers[i][j].computeOutput(inputs));
        }
        inputs = outputs;

        if(history)
        {
            for (int j = 0; j < m_bias; ++j)
                outputs.push_back(1.0); //biases
            history->push_back(outputs);
        }

    }

    return outputs;
}

//TODO Vectorize everything
void Feed_forward_nn::back_propagate(vectord inputs, vectord targets, double eta, double momentum, std::function<double(double, double)> cost_derivative, std::function<double(double)> activation_prime)
{
    //Forward pass
    std::vector<vectord> history;
    std::vector<double> result = FeedForward(inputs, &history);
    //std::cout << "res " << result.back() << std::endl;
    //std::cout << "hist back " << history.back()[0] << std::endl;

    //Backward pass
    std::vector<Array<double, Dynamic, Dynamic> > oldWeights = GetWeights();
    auto newWeights = oldWeights;
    RowArrayXd oldErrors;

    //Output layer
    {
        RowArrayXd error(m_vecLayers.back().size());
        vectord diff = vectorize<double>(cost_derivative,   history[2], targets);
        for (int i = 0; i < error.size();i++)
        {
            //std::cout << "diff : " << diff[i] << std::endl;
            error(i)=(diff[i] *     activation_prime(history.back()[i]) );
            //std::cout << "error : " << error(i) << std::endl;
        }
        oldErrors = error;
    }
    //Updating weight
    for (int i = 0; i < oldErrors.size(); ++i)
    {
        for(int j = 0; j < newWeights.back().cols();++j)
        {
            //std::cout << "oldW : " << oldWeights.back()(i,j) << std::endl;
            //std::cout << "oldErr" << oldErrors[i] << std::endl;
            //std::cout << "truc" << history[history.size()-2][j] << std::endl;
            newWeights.back()(i, j) = oldWeights.back()(i,j) + -eta * (oldErrors(i)*history[history.size()-2][j]) + momentum * m_oldChange.back()(i,j);
            m_oldChange.back()(i,j) = newWeights.back()(i,j) - oldWeights.back()(i,j);
            //std::cout << "new old: " << m_oldChange.back()(i,j) << std::endl;
            //std::cout << "newW : " << newWeights.back()(i,j) << std::endl;
        }
    }

    //hidden layers
    for (int i = 2; i < m_vecLayers.size();i++)
    {
        int x = m_vecLayers.size()-i;
        RowArrayXd error(m_vecLayers[x].size());
        for (int j = 0; i < error.size(); ++j)
        {
            error(j) = 0;
            for (int k = 0; k < oldErrors.size(); ++k)
            {
                error(j) += oldErrors(k)*oldWeights[x+1](k, j);
            }
            error(j) *= activation_prime(history[x][j]);
        }

        //Updating weight
        for (int j = 0; j < error.size(); ++j)
        {
            for(int k = 0; k < newWeights[x].cols();++k)
            {
                newWeights[x](j, k) = oldWeights[x](j,k) + -eta * (error(j)*history[x-1][k]) + momentum * m_oldChange[x](j, k);
                m_oldChange[x](j,k) = newWeights[x](j, k) - oldWeights[x](j,k);
            }
        }
        oldErrors = error;
    }

    SetWeights(newWeights);
}

}
