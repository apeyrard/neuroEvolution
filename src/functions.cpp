#include <math.h>

#include "functions.h"

namespace neuroev
{

double sigmoid(double x)
{
    double p = 1.0;
    return ( 1.0 / ( 1.0 + exp(-x / p) ) );
}

double sigmoid_prime(double x)
{
    return (sigmoid(x) * (1.0 - sigmoid(x)));
}

double mean_square_derivative(double output, double target)
{
    return output - target;
}

double identity(double x)
{
    return x;
}

double dRand(double dMin, double dMax)
{
    double d = (double)rand() / RAND_MAX;
    return dMin + d * (dMax - dMin);
}

}
