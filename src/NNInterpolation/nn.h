#ifndef NN_H
#define NN_H

#include <iostream>
#include <vector>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <immintrin.h>
#include <memory>

#include "layer.h"
#include "losses.h"

/**
 * Overloading operator<< for printing vectors.
 */
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& vec) {
    os << "[";
    if (!vec.empty()) {
        os << vec[0];
        for (size_t i = 1; i < vec.size(); ++i) {
            os << ", " << vec[i];
        }
    }
    os << "]";
    return os;
}

/**
 * NN class (Neural Network):
 * 
 * - Maintains a sequence of Layer objects in a std::vector of std::unique_ptr<Layer>.
 * - Provides methods to:
 *     - Add a layer (add)
 *     - Generate predictions (predict)
 *     - Perform forward propagation (forward_propagation)
 *     - Perform backward propagation (backward_propagation)
 *     - Train the network (fit) using a simplistic loop over epochs and samples.
 */
class NN
{
public:
    // List of pointers to Layer objects. Each layer is owned by the network (unique_ptr).
    std::vector<std::unique_ptr<Layer>> layers;

    /**
     * add:
     *     - Accepts a raw pointer to a Layer
     *       (which will be wrapped into a unique_ptr).
     *     - The network then "owns" this Layer object.
     */
    void add(Layer *layer)
    {
        layers.emplace_back(layer);
    }

    /**
     * predict:
     *     - Runs forward propagation on the input
     *       and returns the final output of the network.
     */
    std::vector<float> predict(std::vector<float> input)
    {
        return forward_propagation(input);
    }

    /**
     * forward_propagation:
     *     - Iterates over each layer in order.
     *     - Passes the output of one layer as the input to the next layer.
     *     - Returns the final output vector.
     */
    std::vector<float> forward_propagation(const std::vector<float> input)
    {
        std::vector<float> output = input;
        for (const auto &layer : layers)
        {
            output = layer->forward(output);
        }
        return output;
    }

    /**
     * backward_propagation:
     *     - Starts from the final layer, moving backwards.
     *     - Each layer receives the gradient from the next layer (or from the loss).
     *     - Each layer's backward function returns the gradient for the previous layer.
     *     - The 'grad' vector is updated at each step in reverse order.
     */
    void backward_propagation(const std::vector<float> &error, float learning_rate)
    {
        std::vector<float> grad = error; // The incoming error from the loss function
        for (auto it = layers.rbegin(); it != layers.rend(); ++it)
        {
            grad = (*it)->backward(grad, learning_rate);
        }
    }

    /**
     * fit:
     *     - Trains the network using the provided data (X) and targets (y).
     *     - For each epoch:
     *         * Reset a total_loss counter.
     *         * For each sample in X :
     *              1) Perform a forward pass to get the output.
     *              2) Compute the MSE loss.
     *              3) Calculate the derivative of that loss w.r.t. the output.
     *              4) Perform backpropagation to update the network's parameters.
     *         * Print the average loss over all samples in X.
     *
     * @param X Input features: a vector of samples, each being a vector<float>.
     * @param y Ground truth: the corresponding target values for each sample.
     * @param epochs Number of passes through the entire dataset.
     * @param learning_rate Step size for parameter updates.
     */
    void fit(const std::vector<std::vector<float>> &X, const std::vector<std::vector<float>> &y,
             int epochs, float learning_rate)
    {
        for (int epoch = 0; epoch < epochs; ++epoch)
        {
            float total_loss = 0.0f;

            // Iterate over each sample in the dataset
            for (size_t i = 0; i < X.size(); ++i)
            {
                std::vector<float> output = forward_propagation(X[i]);

                float loss = MSELoss(y[i], output);
                total_loss += loss;

                std::vector<float> loss_derivative = MSELossDerivative(y[i], output);

                backward_propagation(loss_derivative, learning_rate);
            }

            // Print average loss for this epoch
            std::cout << "Epoch " << epoch + 1 << "/" << epochs
                      << " - Loss: " << total_loss / X.size() << std::endl;
        }
    }
};

#endif