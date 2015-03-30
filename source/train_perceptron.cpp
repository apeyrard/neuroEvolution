#include "neuron.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

void train(Neuron &p, std::vector<double> const &inputs, double correct, double learningRate)
{
    double output = p.computeOutput(inputs);
    double error = correct - output;

    for (unsigned int i = 0; i < p.m_weights.size(); i++)
    {
        p.m_weights[i] += learningRate * error * inputs[i];
    }
}

double randFloat(double min, double max)
{
    double random = static_cast<double>(rand()) / static_cast<double>(RAND_MAX);
    random *= (max - min);
    random -= min;
    return random;
}

double f(double x)
{
    return 2*x+12;
}

double sigmoid(double x)
{
    double p = 1.0f;
    return ( 1.0f / ( 1.0f + exp(-x / p) ) );
}

double activate(double x)
{
    if (x > 0.0f)
    {

        return 1.0f;
    }
    else
    {
        return -1.0f;
    }
}

int main(int argc, char *argv[])
{
    srand (static_cast <unsigned> (time(0)));

    std::vector<double> weights;
    weights.push_back(2.0f);
    weights.push_back(0.3f);
    weights.push_back(-0.5f);

    double rate = 0.01f;

    Neuron p = Neuron(activate, weights);

    //train
    for (int i = 0; i < 1000000; i++)
    {
        double x = randFloat(-100.0f, 100.0f);
        double y = randFloat(-100.0f, 100.0f);

        double yline = f(x);
        double answer;
        if (y < yline)
        {
            answer = -1.0f;
        }
        else
        {
            answer = 1.0f;
        }
        std::vector<double> inputs;
        inputs.push_back(x);
        inputs.push_back(y);
        train(p, inputs, answer, rate);
    }

    //verify
    int ok = 0;
    int ko = 0;
    for (int i = 0; i < 10000; i++)
    {
        double x = randFloat(-100.0f, 100.0f);
        double y = randFloat(-100.0f, 100.0f);

        double yline = f(x);
        double answer;
        if (y < yline)
        {
            answer = -1.0f;
        }
        else
        {
            answer = 1.0f;
        }
        std::vector<double> inputs;
        inputs.push_back(x);
        inputs.push_back(y);
        double result = p.computeOutput(inputs);
        if (result == answer)
        {
            ok ++;
        }
        else
        {
            ko++;
        }
    }

    printf("OK : %d\n",ok);
    printf("KO : %d\n",ko);
    return 0;
}
