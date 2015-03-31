#include <stdio.h>
#include <math.h>
#include <algorithm>
#include <random>
#include <eigen3/Eigen/Core>
#include <time.h>
#include <stdlib.h>

#include "neuroevolution.h"

#include <iostream>

using namespace neuroev;

using data = std::vector<std::pair<std::vector<double>, std::vector<double> > >;


//TODO nn ref const ?
/*void update_batch(Feed_forward_nn &nn, data batch, double eta)
{
    std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > weights = nn.GetWeights();
    std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > nabla_w;
    std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > newWeights;

    for (uint i = 0; i < weights.size(); i++)
    {
        int rows = weights[i].rows();
        int cols = weights[i].cols();
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> mat(rows, cols);
        mat.setZero(rows, cols);
        nabla_w.push_back(mat);
    }

    for (int i = 0; i < batch.size();i++)
    {
        std::vector<double> input = batch[i].first;
        std::vector<double> target = batch[i].second;
        std::vector<Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> > delta_nabla_w = nn.back_propagate(input, target);
        for (uint j = 0; j < nabla_w.size();j++)
        {
            nabla_w[j] += delta_nabla_w[j];
        }
    }

    for (int i = 0; i < weights.size(); i++)
    {
        //std::cout << "Weight " << weights[i] << " eta " << eta << " nabla " << nabla_w[i] << std::endl;
        Eigen::Array<double, Eigen::Dynamic, Eigen::Dynamic> newWeight = weights[i] - (eta/batch.size())*nabla_w[i];
        //std::cout << "newWeight " << newWeight << std::endl;
        newWeights.push_back(newWeight);
    }
    nn.SetWeights(newWeights);
}

void SGD(Feed_forward_nn &nn, data trainingData, int epochs, int batch_size, double eta)
{
    auto engine = std::default_random_engine{};
    for (uint i = 0; i < epochs; i++)
    {
        data shuffled = trainingData;
        std::shuffle(std::begin(shuffled), std::end(shuffled), engine);
        std::vector<data> batches;

        //Number of batches needed:
        double nbBatches = ceil(static_cast<float>(trainingData.size()) / static_cast<float>(batch_size));

        for (uint j = 0; j < nbBatches;j++)
        {
            data batch;
            for (uint k = 0; k < batch_size; k++)
            {
                if (!shuffled.empty())
                {
                    batch.push_back(shuffled.back());
                    shuffled.pop_back();
                }
            }
            batches.push_back(batch);
        }

        for (uint j = 0; j < batches.size(); j++)
        {
            update_batch(nn, batches[j], eta);
        }
    }
}*/

int main(int argc, char *argv[])
{
    srand(time(NULL));
    std::vector<int> sizes;
    sizes.push_back(1);
    sizes.push_back(5);
    sizes.push_back(1);


    Feed_forward_nn nn  = Feed_forward_nn(sizes);

    //Learning OR
    for (int i = 0; i < 10000000; ++i)
    {
        int a = rand() % 2;
        int b = rand() % 2;
        int result;
        if (((a==0) && (b==0)) || ((a==1)&&(b==1)))
            result = 0;
        else
            result = 1;

        std::vector<double> inputs;
        std::vector<double> targets;
        inputs.push_back(a);
        inputs.push_back(b);
        targets.push_back(result);
        nn.back_propagate(inputs, targets);
    }
    // testing OR
    int ok = 0;
    int ko = 0;
    int a,b,result;
    for (int i = 0; i < 100; ++i)
    {
        a = rand() % 2;
        b = rand() % 2;
        std::vector<double> source;
        source.push_back(a);
        source.push_back(b);
        if (((a==0) && (b==0)) || ((a==1)&&(b==1)))
            result = 0;
        else
            result = 1;
        double result2 = nn.FeedForward(source).back();
        if (result2 > 0.9)
            result2 = 1.0;
        else if (result2 < 0.1)
            result2 = 0.0;
        else
            result2 = 0.5;
        printf("result : %f\n", result2);

        if (result == result2)
            ++ok;
        else
            ++ko;
    }
    printf("ok : %d\n", ok);
    printf("ko : %d\n", ko);

    return 0;
}
