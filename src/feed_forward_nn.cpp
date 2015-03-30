#include "feed_forward_nn.h"

#include <iostream>

namespace neuroev{
//TODO wrapper to apply function to vector
//TODO test limit cases (empty nn, 1 level nn....)
//TODO allow different biases
//TODO sigmoid as default
//TODO profile, and maybe remove std to rely on c style arrays, or matrices
Feed_forward_nn::Feed_forward_nn(std::vector<uint> vecNbNeurons, std::function<double(double)> activation)
{
    m_vecLayers.push_back( create_layer(activation, vecNbNeurons[0], 1) );
    for (uint i = 1; i < vecNbNeurons.size();i++)
    {
        m_vecLayers.push_back( create_layer(activation, vecNbNeurons[i], m_vecLayers[i-1].size() + 1 ) );  //+1 for bias
    }
}

Feed_forward_nn::~Feed_forward_nn()
{
}

//TODO replace with weight matrices
std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > Feed_forward_nn::GetWeights() const
{
    std::vector<Array<double, Dynamic, Dynamic> > result;
    for (uint i = 0; i < m_vecLayers.size();i++)
    {
        uint rows = m_vecLayers[i].size();
        uint cols = m_vecLayers[i][0].m_weights.size();
        Array<double, Dynamic, Dynamic> layerWeights(rows, cols);
        for (uint j = 0; j < m_vecLayers[i].size(); j++)
        {
            for (uint k = 0; k < m_vecLayers[i][j].m_weights.size();k++)
            {
                    layerWeights(j, k) = m_vecLayers[i][j].m_weights[k];
            }
        }
        result.push_back(layerWeights);
    }
    return result;
}

//TODO verify size before
void Feed_forward_nn::SetWeights(std::vector<Array<double, Dynamic, Dynamic> > &weights)
{
    for (uint i = 0; i < m_vecLayers.size();i++)
    {
        for (uint j = 0; j < m_vecLayers[i].size(); j++)
        {
            for (uint k = 0; k < m_vecLayers[i][j].m_weights.size();k++)
            {
                //std::cout << weights[i](j,k) << std::endl;
                m_vecLayers[i][j].m_weights[k] = weights[i](j, k);
            }
        }
    }
}

Feed_forward_nn::vectord Feed_forward_nn::FeedForward(vectord &inputs, std::vector<vectord >* history, std::vector<vectord >* netList)
{
    if(history)
        history->push_back(inputs);

    vectord outputs;
    vectord tmpNet;

    // Input layer:
    for (uint i = 0; i < m_vecLayers[0].size();i++)
    {
        vectord tmpVec;
        tmpVec.push_back(inputs[i]);
        if(netList)
        {
            double tmp;
            outputs.push_back(m_vecLayers[0][i].computeOutput(tmpVec, &tmp));
            tmpNet.push_back(tmp);
        }
        else
            outputs.push_back(m_vecLayers[0][i].computeOutput(tmpVec));
    }
    inputs = outputs;


    if(netList)
        netList->push_back(tmpNet);
    tmpNet.clear();
    if(history)
        history->push_back(outputs);

    // Other layers
    for (uint i = 1; i < m_vecLayers.size(); ++i)
    {
        outputs.clear();
        inputs.push_back(1.0f); //bias
        for (uint j = 0; j < m_vecLayers[i].size(); ++j)
        {
            if(netList)
            {
                double tmp;
                outputs.push_back(m_vecLayers[i][j].computeOutput(inputs, &tmp));
                tmpNet.push_back(tmp);
            }
            else
                outputs.push_back(m_vecLayers[i][j].computeOutput(inputs));
        }
        inputs = outputs;

        if(netList)
            netList->push_back(tmpNet);
        tmpNet.clear();
        if(history)
            history->push_back(outputs);
    }

    return outputs;
}

std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > Feed_forward_nn::back_propagate(vectord inputs, vectord targets, std::function<double(double, double)> cost_derivative, std::function<double(double)> activation_prime)
{
    std::vector<Array<double, Dynamic, Dynamic> > weights = GetWeights();
    std::vector<Array<double, Dynamic, Dynamic> > nabla_w;

    for (uint i = 0; i < weights.size(); i++)
    {
        int rows = weights[i].rows();
        int cols = weights[i].cols();
        Array<double, Dynamic, Dynamic> mat(rows, cols);
        mat.setZero(rows, cols);
        nabla_w.push_back(mat);
    }

    std::vector<vectord> history;
    std::vector<vectord> listNet;
    //Forward pass
    FeedForward(inputs, &history, &listNet);

    //Backward pass
    RowArrayXd delta(targets.size());
    for (uint i = 0; i < targets.size();i++)
    {
        vectord diff = vectorize<double>(cost_derivative, history.back(), targets);
        delta(i)=(diff[i] * activation_prime(listNet.back()[i]) );
    }

    for (int j = 0; j < delta.size();j++)
    {
        for (uint k = 0; k < history[history.size()-2].size(); k++)
        {
            nabla_w.back()(j,k) = delta(j) * history[history.size()-2][k];
        }
    }

    for (uint i = 2; i < m_vecLayers.size();i++)
    {
        uint x = m_vecLayers.size()-i;
        vectord z = listNet[x];
        RowArrayXd spv(z.size());
        for (uint j = 0; j < z.size();j++)
        {
            spv(j) = (activation_prime(z[j]));
        }
        for (uint j = 0; j < delta.size();j++)
        {
            double tmp = 0;
            for(uint k = 0; k < weights[x+1].rows();k++)
            {
                tmp += delta(j) * weights[x+1](j,k);
            }
            delta(j) = tmp * spv(j);
        }

        for (int j = 0; j < delta.size();j++)
        {
            for (uint k = 0; k < history[history.size()-1-x].size(); k++)
            {
                nabla_w[x](j,k) = delta(j) * history[history.size()-1-x][k];
            }
        }
    }
    return nabla_w;
}

}
