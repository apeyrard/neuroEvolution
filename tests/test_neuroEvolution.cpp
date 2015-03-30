#include <gtest/gtest.h>
#include <vector>

#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <eigen3/Eigen/Core>

#include "neuroevolution.h"

using namespace neuroev;

TEST(Functions, identity)
{
    EXPECT_EQ(0.0,identity(0.0));
    EXPECT_EQ(12.2,identity(12.2));
    EXPECT_EQ(-15.2,identity(-15.2));
}

TEST(Functions, sigmoid)
{
    EXPECT_DOUBLE_EQ(sigmoid(0.0),0.5);
    EXPECT_DOUBLE_EQ(sigmoid(124.2365),1.0/(1.0+exp(-124.2365)));
    EXPECT_DOUBLE_EQ(sigmoid(-1000.0),1.0/(1.0+exp(1000.0)));
}

TEST(Functions, sigmoid_prime)
{
    EXPECT_DOUBLE_EQ(sigmoid_prime(0.0),0.25);
}

TEST(Functions, vectorize)
{
    std::vector<double> inputs;
    std::vector<double> targets;

    for (int i = 0; i < 10; ++i)
    {
        inputs.push_back(dRand(-100.0, 100.0));
        targets.push_back(dRand(-100.0, 100.0));
    }

    std::vector<double> results = vectorize<double>(mean_square_derivative, inputs, targets);

    for (int i = 0; i < results.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(results[i], inputs[i]-targets[i]);
    }

}

TEST(Neuron, DefaultConstructor)
{
    std::vector<double> weights;
    weights.push_back(2.5);
    Neuron n(weights, identity);
    ASSERT_EQ(n.m_weights.back(), 2.5);
}

TEST(Neuron, simpleActivation)
{
    std::vector<double> weights;
    std::vector<double> inputs;
    weights.push_back(1.13);
    inputs.push_back(-12.5);
    Neuron n(weights, identity);
    EXPECT_DOUBLE_EQ(n.computeOutput(inputs), -14.125);
}

TEST(Neuron, randomActivation)
{
    // Let's test 10 times with 0 to 10 inputs

    for (int i = 0; i <= 10; ++i)
    {
        std::vector<double> weights;
        std::vector<double> inputs;
        double sum = 0.0;

        for (int j = 0; j <= i; ++j)
        {
            double in = dRand(-1000.0,1000.0);
            double w = dRand(-1000.0,1000.0);
            inputs.push_back(in);
            weights.push_back(w);
            sum += in*w;
        }

        Neuron n(weights, identity);
        EXPECT_DOUBLE_EQ(n.computeOutput(inputs), sum);
    }
}

TEST(Neuron, sigmoidActivation)
{
    // Let's test 10 times with 0 to 10 inputs

    for (int i = 0; i <= 10; ++i)
    {
        std::vector<double> weights;
        std::vector<double> inputs;
        double sum = 0.0;

        for (int j = 0; j <= i; ++j)
        {
            double in = dRand(-1.0,1.0);
            double w = dRand(-1.0,1.0);
            inputs.push_back(in);
            weights.push_back(w);
            sum += in*w;
        }

        Neuron n(weights, sigmoid);
        EXPECT_DOUBLE_EQ(n.computeOutput(inputs), sigmoid(sum));
    }
}

TEST(Feed_forward_nn, emptyNN)
{
    std::vector<int> nbNeurons;
    Feed_forward_nn n(nbNeurons);

    std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > weights(n.GetWeights());
    EXPECT_EQ(weights.empty(),true);
}

TEST(Feed_forward_nn, simpleNN)
{
    std::vector<int> nbNeurons;
    nbNeurons.push_back(2);
    nbNeurons.push_back(2);
    nbNeurons.push_back(1);
    Feed_forward_nn n(nbNeurons);

    std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > weights(n.GetWeights());

    EXPECT_EQ(weights.size(),3);

    std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > newWeights;
    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> a(2, 1);
    a(0, 0) = 98.5;
    a(1, 0) = -50.12;
    newWeights.push_back(a);

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> b(2, 3);
    b(0, 0) = 0.12;
    b(0, 1) = -5.19;
    b(0, 2) = 3.1415;
    b(1, 0) = 7.0;
    b(1, 1) = 0.0;
    b(1, 2) = 321.12;
    newWeights.push_back(b);

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> c(1, 3);
    c(0, 0) = 1.0;
    c(0, 1) = 42.42;
    c(0, 2) = -48.0;
    newWeights.push_back(c);

    n.SetWeights(newWeights);
    weights.clear();
    weights = n.GetWeights();

    EXPECT_EQ(weights.size(),3);

    EXPECT_DOUBLE_EQ(weights[0](0, 0), 98.5);
    EXPECT_DOUBLE_EQ(weights[0](1, 0), -50.12);
    EXPECT_DOUBLE_EQ(weights[1](0, 0), 0.12);
    EXPECT_DOUBLE_EQ(weights[1](0, 1), -5.19);
    EXPECT_DOUBLE_EQ(weights[1](0, 2), 3.1415);
    EXPECT_DOUBLE_EQ(weights[1](1, 0), 7.0);
    EXPECT_DOUBLE_EQ(weights[1](1, 1), 0.0);
    EXPECT_DOUBLE_EQ(weights[1](1, 2), 321.12);
    EXPECT_DOUBLE_EQ(weights[2](0, 0), 1.0);
    EXPECT_DOUBLE_EQ(weights[2](0, 1), 42.42);
    EXPECT_DOUBLE_EQ(weights[2](0, 2), -48.0);

    std::vector<double> result;
    std::vector<double> inputs;
    std::vector<std::vector<double> > history;
    std::vector<std::vector<double> > netList;
    inputs.push_back(14.12);
    inputs.push_back(-19.5);

    result = n.FeedForward(inputs, &history, &netList);

    EXPECT_EQ(history.size(),4);
    EXPECT_EQ(netList.size(),3);
    EXPECT_EQ(history[0].size(),2);
    EXPECT_EQ(history[1].size(),2);
    EXPECT_EQ(history[2].size(),2);
    EXPECT_EQ(history[3].size(),1);
    EXPECT_EQ(netList[0].size(),2);
    EXPECT_EQ(netList[1].size(),2);
    EXPECT_EQ(netList[2].size(),1);

    EXPECT_DOUBLE_EQ(history[0][0], 14.12);
    EXPECT_DOUBLE_EQ(history[0][1], -19.5);

    EXPECT_DOUBLE_EQ(netList[0][0], 1390.82);
    EXPECT_DOUBLE_EQ(netList[0][1], 977.34);
    EXPECT_DOUBLE_EQ(history[1][0], sigmoid(1390.82));
    EXPECT_DOUBLE_EQ(history[1][1], sigmoid(977.34));
}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    srand(time(NULL));
    return RUN_ALL_TESTS();
}
