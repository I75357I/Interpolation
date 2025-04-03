#ifndef LAYER_HH
#define LAYER_HH

#include <vector>
#include <cstring>      // For std::memcpy
#include <immintrin.h>  // (Optional) For advanced SIMD intrinsics
#include <omp.h>        // (Optional) For multi-threading (OpenMP)

#include "activation.h" // For activation functions and their derivatives
#include "utils.h"      // Utility functions (e.g., initializers, dot product, etc.)

/**
 * Base class for all layers.
 *
 * Each layer has:
 *   - input: vector holding the input data for the layer.
 *   - output: vector holding the result after the forward pass.
 *
 * The virtual functions ensure that derived classes must implement
 * their own forward and backward methods.
 */
class Layer
{
public:
    std::vector<float> input;
    std::vector<float> output;

    /**
     * Forward pass (pure virtual).
     *    - Takes input data and computes the output of the layer.
     *    - Returns a vector of floats representing the layer's output.
     */
    virtual std::vector<float> forward(const std::vector<float> input_data) = 0;

    /**
     * Backward pass (pure virtual).
     *    - Takes the error (from the next layer or from the loss function)
     *      and the learning rate.
     *    - Returns the error with respect to this layer's input (so that
     *      previous layers can continue backpropagating).
     */
    virtual std::vector<float> backward(std::vector<float>& error, float learning_rate) = 0;
};

/**
 * Sigmoid activation layer.
 *
 * forward():
 *   Applies the sigmoid function to each element of the input:
 *       sigmoid(x) = 1 / (1 + e^(-x))
 *
 * backward():
 *   Uses the derivative of the sigmoid function to compute the gradient w.r.t. the input:
 *       sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 *   Multiplies element-wise by the incoming error to get the gradient for backpropagation.
 */
class Sigmoid : public Layer
{
public:
    std::vector<float> forward(const std::vector<float> input_data) override
    {
        input = input_data;
        // vectSigmoid applies sigmoid to each element in 'input'
        output = vectSigmoid(input);
        return output;
    }

    std::vector<float> backward(std::vector<float>& error, float learning_rate) override
    {
        std::vector<float> derivative = vectSigmoidDerivative(input);
        std::vector<float> grad_input;

        // Multiply each derivative value by the corresponding error
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

/**
 * ReLU (Rectified Linear Unit) activation layer.
 *
 * forward():
 *   Applies ReLU to each element of input:
 *       ReLU(x) = max(0, x)
 *
 * backward():
 *   Uses the derivative of ReLU:
 *       ReLU'(x) = 1 if x >= 0, else 0
 *   Multiplies element-wise by the incoming error.
 */
class Relu : public Layer
{
public:
    std::vector<float> forward(const std::vector<float> input_data) override
    {
        input = input_data;
        // vectRelu applies ReLU to each element in 'input'
        output = vectRelu(input);
        return output;
    }

    std::vector<float> backward(std::vector<float>& error, float learning_rate) override
    {
        derivative.clear();
        grad_input.clear();

        // vectReluDerivative returns 1 for x >= 0, else 0
        derivative = vectReluDerivative(input);

        // Multiply derivative by incoming error
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }

    // These members store temporary computation for backward pass:
    std::vector<float> derivative;
    std::vector<float> grad_input;
};

/**
 * Leaky ReLU activation layer.
 *
 * forward():
 *   Applies Leaky ReLU:
 *       LeakyReLU(x) = x        if x > 0
 *                      alpha*x  otherwise
 *
 * backward():
 *   Uses derivative of Leaky ReLU:
 *       LeakyReLU'(x) = 1       if x >= 0
 *                        alpha  otherwise
 */
class LeakyRelu : public Layer
{
public:
    float alpha = 0.01;

    std::vector<float> forward(const std::vector<float> input_data) override
    {
        input = input_data;
        output = vectLeakyRelu(input, alpha);
        return output;
    }

    std::vector<float> backward(std::vector<float>& error, float learning_rate) override
    {
        std::vector<float> derivative = vectLeakyReluDerivative(input, alpha);
        std::vector<float> grad_input;

        // Multiply derivative by incoming error
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

/**
 * Tanh activation layer.
 *
 * forward():
 *   Applies tanh to each element of input:
 *       tanh(x) = (e^x - e^-x) / (e^x + e^-x)
 *
 * backward():
 *   Uses derivative of tanh:
 *       tanh'(x) = 1 - tanh^2(x)
 */
class Tanh : public Layer
{
public:
    std::vector<float> forward(const std::vector<float> input_data) override
    {
        input = input_data;
        output = vectTanh(input);
        return output;
    }

    std::vector<float> backward(std::vector<float>& error, float learning_rate) override
    {
        std::vector<float> derivative = vectTanhDerivative(input);
        std::vector<float> grad_input;

        // Multiply derivative by incoming error
        for (int i = 0; i < derivative.size(); ++i)
        {
            grad_input.push_back(derivative[i] * error[i]);
        }
        return grad_input;
    }
};

/**
 * Fully connected (Linear) layer.
 *
 * This layer performs:
 *    output = weights * input + bias
 *
 * Where:
 *    - 'weights' is a 2D array (dimensions: [output_neuron][input_neuron]).
 *    - 'bias' is a 1D array of length 'output_neuron'.
 *
 * Also handles the backward pass, computing:
 *    - The gradient w.r.t. the weights (weight_error).
 *    - The gradient w.r.t. the bias (bias_error).
 *    - The gradient w.r.t. the input (input_error), to propagate the
 *      errors back to previous layers.
 *
 * After calculating these gradients, the layer updates its weights
 * and biases using the provided 'learning_rate'.
 */
class Linear : public Layer
{
public:
    int input_neuron;   // Number of inputs (size of input vector)
    int output_neuron;  // Number of outputs (size of output vector)

    float** weights;         // 2D array [output_neuron][input_neuron]
    float* bias;             // 1D array [output_neuron]
    float* input_error;      // 1D array [input_neuron]
    float** weight_error;    // 2D array [output_neuron][input_neuron]
    float* bias_error;       // 1D array [output_neuron]
    float** weight_transpose;// 2D array [input_neuron][output_neuron]
    float* delta_bias;       // Temporary storage to update bias

    /**
     * Constructor: Initializes the weights and bias
     * with random or uniform values and allocates memory.
     */
    Linear(int num_in, int num_out)
        : input_neuron(num_in), output_neuron(num_out)
    {
        allocateMemory();
        initializeWeightsAndBias(num_in, num_out);
    }

    /**
     * Destructor: Frees dynamically allocated memory.
     */
    ~Linear()
    {
        releaseMemory();
    }

    /**
     * Forward pass:
     *   1. Stores the input data.
     *   2. Performs the linear transformation: output = W * input + bias.
     */
    std::vector<float> forward(std::vector<float> input_data) override
    {
        input = input_data;
        return computeForward();
    }

    /**
     * Backward pass:
     *   1. Computes error for bias (simply copy the 'error' vector).
     *   2. Computes error for weights (outer product of error and input).
     *   3. Computes error for the previous layer's input (dot product with W^T).
     *   4. Updates bias and weights using the learning_rate.
     *   5. Returns the input_error vector (the gradient w.r.t. this layer's input).
     */
    std::vector<float> backward(std::vector<float>& error, float learning_rate) override
    {
        computeBiasError(error);
        computeWeightError(error);
        computeInputError();
        updateBias(learning_rate);
        updateWeights(learning_rate);

        // Return the error w.r.t. the input (so the previous layer can backprop)
        return std::vector<float>(input_error, input_error + input_neuron);
    }

private:
    /**
     * Memory allocation for all internal arrays.
     */
    void allocateMemory() {
        delta_bias = new float[output_neuron];

        // weights: [output_neuron][input_neuron]
        weights = allocate2DArray(output_neuron, input_neuron);

        // bias: [output_neuron]
        bias = new float[output_neuron];

        // input_error: [input_neuron]
        input_error = new float[input_neuron];

        // weight_error: [output_neuron][input_neuron]
        weight_error = allocate2DArray(output_neuron, input_neuron);

        // bias_error: [output_neuron]
        bias_error = new float[output_neuron];

        // weight_transpose: [input_neuron][output_neuron]
        weight_transpose = allocate2DArray(input_neuron, output_neuron);
    }

    /**
     * Helper function for creating a 2D array of floats.
     */
    float** allocate2DArray(int rows, int cols) {
        float** array = new float* [rows];
        for (int i = 0; i < rows; i++) {
            array[i] = new float[cols];
        }
        return array;
    }

    /**
     * Initializes weights and biases (you might use random or uniform distributions).
     */
    void initializeWeightsAndBias(int num_in, int num_out) {
        // Provided by "utils.h" - for example, sets random or uniform values
        uniformWeightInitializer(num_out, num_in, weights);
        biasInitializer(num_out, bias);
    }

    /**
     * Compute the forward pass: output = W * input + bias
     */
    std::vector<float> computeForward() {
        output.clear();
        for (int i = 0; i < output_neuron; i++)
        {
            // dotProduct(weights[i], input) is a utility function that
            // computes the dot product of row weights[i] with the input vector.
            output.emplace_back(dotProduct(weights[i], input) + bias[i]);
        }
        return output;
    }

    /**
     * Copies the incoming error to bias_error.
     */
    void computeBiasError(const std::vector<float>& error) {
        // Copy all values from 'error' into 'bias_error'
        std::memcpy(bias_error, error.data(), output_neuron * sizeof(float));
    }

    /**
     * Compute the weight_error by multiplying the error with the input
     * (like an outer product).
     */
    void computeWeightError(const std::vector<float>& error) {
        for (int j = 0; j < output_neuron; j++)
        {
            for (int i = 0; i < input_neuron; i++)
            {
                weight_error[j][i] = error[j] * input[i];
            }
        }
    }

    /**
     * Compute the input_error to pass back to the previous layer:
     *   input_error[i] = Sum_j(weights[j][i] * error[j])
     */
    void computeInputError() {
        for (int i = 0; i < input_neuron; i++) {
            input_error[i] = 0.0f;
            for (int j = 0; j < output_neuron; j++) {
                input_error[i] += weights[j][i] * bias_error[j];
            }
        }
    }

    /**
     * Update the bias terms:
     *   bias[j] -= learning_rate * bias_error[j]
     */
    void updateBias(float learning_rate) {
        for (int j = 0; j < output_neuron; j++) {
            delta_bias[j] = bias_error[j] * learning_rate;
            bias[j] -= delta_bias[j];
        }
    }

    /**
     * Update the weights:
     *   weights[j][i] -= learning_rate * weight_error[j][i]
     */
    void updateWeights(float learning_rate) {
        for (int j = 0; j < output_neuron; j++) {
            for (int i = 0; i < input_neuron; i++) {
                weights[j][i] -= weight_error[j][i] * learning_rate;
            }
        }
    }

    /**
     * Frees the memory for all dynamic allocations in the destructor.
     */
    void releaseMemory() {
        deallocate2DArray(weights, output_neuron);
        delete[] bias;
        delete[] input_error;
        delete[] bias_error;
        deallocate2DArray(weight_error, output_neuron);
        deallocate2DArray(weight_transpose, input_neuron);
        delete[] delta_bias;
    }

    /**
     * Helper function for deallocating the 2D arrays.
     */
    void deallocate2DArray(float** array, int rows) {
        for (int i = 0; i < rows; i++) {
            delete[] array[i];
        }
        delete[] array;
    }
};

#endif
