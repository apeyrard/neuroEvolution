#include <gtest/gtest.h>
#include <vector>

#include "neuron.h"

double identity(double x)
{
    return x;
}

TEST(Neuron, DefaultConstructor)
{
    std::vector<double> weights;
    weights.push_back(2.5);
    Neuron n(identity, weights);
    ASSERT_EQ(n.m_weights.back(), 2.5);
}
