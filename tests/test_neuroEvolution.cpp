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

    EXPECT_EQ(weights.size(),2);

    std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > newWeights;

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> a(2, 3);
    a(0, 0) = 0.12;
    a(0, 1) = -5.19;
    a(0, 2) = 3.1415;
    a(1, 0) = 7.0;
    a(1, 1) = 0.0;
    a(1, 2) = 321.12;
    newWeights.push_back(a);

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> b(1, 3);
    b(0, 0) = 1.0;
    b(0, 1) = 42.42;
    b(0, 2) = -48.0;
    newWeights.push_back(b);

    n.SetWeights(newWeights);
    weights.clear();
    weights = n.GetWeights();

    EXPECT_EQ(weights.size(),2);

    EXPECT_DOUBLE_EQ(weights[0](0, 0), 0.12);
    EXPECT_DOUBLE_EQ(weights[0](0, 1), -5.19);
    EXPECT_DOUBLE_EQ(weights[0](0, 2), 3.1415);
    EXPECT_DOUBLE_EQ(weights[0](1, 0), 7.0);
    EXPECT_DOUBLE_EQ(weights[0](1, 1), 0.0);
    EXPECT_DOUBLE_EQ(weights[0](1, 2), 321.12);
    EXPECT_DOUBLE_EQ(weights[1](0, 0), 1.0);
    EXPECT_DOUBLE_EQ(weights[1](0, 1), 42.42);
    EXPECT_DOUBLE_EQ(weights[1](0, 2), -48.0);

    std::vector<double> result;
    std::vector<double> inputs;
    std::vector<std::vector<double> > history;
    std::vector<std::vector<double> > netList;
    inputs.push_back(14.12);
    inputs.push_back(-19.5);

    result = n.FeedForward(inputs, &history);

    EXPECT_EQ(history.size(),3);
    EXPECT_EQ(history[0].size(),2);
    EXPECT_EQ(history[1].size(),2);
    EXPECT_EQ(history[2].size(),1);

    //Values hard to check since double, but seem ok
    //when outputting neuron values*/
}

TEST(Feed_forward_nn, knownNN)
{
    std::vector<int> nbNeurons;
    nbNeurons.push_back(2);
    nbNeurons.push_back(2);
    nbNeurons.push_back(1);
    Feed_forward_nn n(nbNeurons, sigmoid, 0);

    std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > weights(n.GetWeights());

    EXPECT_EQ(weights.size(),2);

    std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > newWeights;

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> a(2, 2);
    a(0, 0) = 0.1;
    a(0, 1) = 0.8;
    a(1, 0) = 0.4;
    a(1, 1) = 0.6;
    newWeights.push_back(a);

    Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> b(1, 2);
    b(0, 0) = 0.3;
    b(0, 1) = 0.9;
    newWeights.push_back(b);

    n.SetWeights(newWeights);
    weights.clear();

    std::vector<double> result;
    std::vector<double> inputs;
    std::vector<std::vector<double> > history;
    inputs.push_back(0.35);
    inputs.push_back(0.9);

    result = n.FeedForward(inputs, &history);

    EXPECT_DOUBLE_EQ(result.back(), 0.69);
    EXPECT_DOUBLE_EQ(history[2][0], 0.69);
    EXPECT_DOUBLE_EQ(history[1][0], 0.68);
    EXPECT_DOUBLE_EQ(history[1][1], 0.6637);
    EXPECT_DOUBLE_EQ(history[0][0], 0.35);
    EXPECT_DOUBLE_EQ(history[0][1], 0.9);

    std::vector<double> target;
    target.push_back(0.5);

    n.back_propagate(inputs, target, 1.0);

    history.clear();
    result = n.FeedForward(inputs, &history);

    EXPECT_DOUBLE_EQ(result.back(), 0.68205);


}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    srand(time(NULL));
    return RUN_ALL_TESTS();
}
