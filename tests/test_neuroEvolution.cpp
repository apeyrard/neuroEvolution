#include <gtest/gtest.h>
#include <vector>

#include <stdlib.h>
#include <time.h>
#include <math.h>

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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    srand(time(NULL));
    return RUN_ALL_TESTS();
}
