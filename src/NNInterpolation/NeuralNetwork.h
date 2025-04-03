#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include "nn.h"
#include "layer.h"
#include "losses.h"
#include "utils.h"
#include <memory>
#include <vector>
#include <stdexcept>
#include <iostream>

/**
 * @brief NeuralNetwork class wraps the underlying NN (from nn.h)
 *        into a modern interface that performs input validations,
 *        manages hyperparameters, and provides a unified interface.
 *
 * References:
 *   - "Deep Learning", Goodfellow et al.
 *
 * Complexity:
 *   - Forward/backward passes cost roughly O(n_layers * cost_per_layer).
 *
 * All implementations are provided inline.
 */
class NeuralNetwork {
public:
    using Real = float;

    NeuralNetwork() = default;

    /**
     * @brief Adds a layer to the network.
     * Ownership of the layer is transferred.
     */
    void addLayer(Layer* layer) {
        if (!layer)
            throw std::invalid_argument("Cannot add a null layer.");
        network.add(layer);
    }

    /**
     * @brief Trains the network.
     *
     * @param X Input samples (each sample is a vector<Real>).
     * @param y Corresponding target values.
     * @param epochs Number of epochs.
     * @param learningRate Learning rate for parameter updates.
     */
    void train(const std::vector<std::vector<Real>>& X,
        const std::vector<std::vector<Real>>& y,
        int epochs, Real learningRate) {
        if (X.empty() || y.empty() || X.size() != y.size())
            throw std::invalid_argument("Training data and targets must be non-empty and of equal size.");
        size_t dim = X.front().size();
        for (const auto& sample : X) {
            if (sample.size() != dim)
                throw std::invalid_argument("All input samples must have the same dimension.");
        }
        network.fit(X, y, epochs, learningRate);
    }

    /**
     * @brief Predicts output for a single sample.
     *
     * @param input A sample input.
     * @return std::vector<Real> The network's prediction.
     */
    std::vector<Real> predict(const std::vector<Real>& input) {
        if (input.empty())
            throw std::invalid_argument("Input sample cannot be empty.");
        return network.predict(input);
    }

    /**
     * @brief Returns the underlying NN for advanced usage.
     */
    NN& getInternalNetwork() { return network; }

    /**
     * @brief Creates and returns a default neural network model.
     *
     * The default model structure is:
     *   - Linear layer: 1 input -> 50 neurons
     *   - ReLU activation
     *   - Linear layer: 50 -> 30 neurons
     *   - ReLU activation
     *   - Linear layer: 30 -> 1 neuron
     *   - Relu activation (output in [-1, 1])
     *
     * @return NeuralNetwork A fully configured default model.
     */
    static NeuralNetwork DefaultModel() {
        NeuralNetwork model;
        model.addLayer(new Linear(1, 50));
        model.addLayer(new Relu());
        model.addLayer(new Linear(50, 30));
        model.addLayer(new Relu());
        model.addLayer(new Linear(30, 1));
        return model;
    }

private:
    NN network; // Underlying NN implementation.
};

#endif // NEURAL_NETWORK_H
