#include "src/NNInterpolation/NeuralNetwork.h"
#include <iostream>
#include <vector>
#include <cmath>

/**
 * @brief Example of using the NeuralNetwork class.
 *
 * This example approximates the sine function over the interval [0, pi/2].
 * It generates training data, creates a default model using NeuralNetwork::DefaultModel(),
 * trains the network, and then prints predictions vs. actual sin(x) values.
 */
int main() {
    using Real = NeuralNetwork::Real;
    using Sample = std::vector<Real>;

    // Generate training data: sin(x) for x in [0, π/2]
    const int numSamples = 300;
    std::vector<Sample> trainInputs;
    std::vector<Sample> trainOutputs;
    float pi = 3.141592653589793;
    for (int i = 0; i < numSamples; ++i) {
        Real x = static_cast<Real>(i) / (numSamples - 1) * (pi / 2);
        Real y = std::sin(x);
        trainInputs.push_back({ x });
        trainOutputs.push_back({ y });
    }

    // Create the default model using NeuralNetwork::DefaultModel()
    NeuralNetwork nn = NeuralNetwork::DefaultModel();

    // Set training hyperparameters.
    Real learningRate = 0.01;
    int epochs = 500;

    std::cout << "Training Neural Network with " << numSamples
        << " samples, " << epochs << " epochs, learning rate = "
        << learningRate << std::endl;

    nn.train(trainInputs, trainOutputs, epochs, learningRate);

    // Test the network on values from 0 to 2pi.
    std::cout << "\nTesting network predictions vs. sin(x):" << std::endl;
    for (Real x = 0.0; x <= (pi/2); x += (pi/2)/10) {
        Sample prediction = nn.predict({ x });
        std::cout << "x = " << x
            << ", predicted sin(x) = " << prediction[0]
            << ", actual sin(x) = " << std::sin(x) << std::endl;
    }

    return 0;
}
