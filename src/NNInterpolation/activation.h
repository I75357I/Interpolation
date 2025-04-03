#ifndef ACTIVATION_HH
#define ACTIVATION_HH

#include <cmath>
#include <vector>

/**
 * Sigmoid function.
 *
 * This function takes a single float 'x' and applies the sigmoid activation:
 *    sigmoid(x) = 1 / (1 + e^(-x))
 *
 * The sigmoid activation maps any real value into the range (0, 1).
 * It's commonly used in output layers for binary classification
 * (e.g., logistic regression).
 */
float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

/**
 * Derivative of the Sigmoid function.
 *
 * The derivative of the sigmoid can be expressed in multiple ways.
 * One common form is:
 *    sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
 *
 * Here, we're using an alternative but mathematically equivalent expression:
 *    derivative = e^x / (e^x + 1)^2
 *
 * This can be derived by directly differentiating the sigmoid function.
 */
float sigmoidDerivative(float x) {
    return std::exp(x) / std::pow((std::exp(x) + 1.0f), 2);
}

/**
 * Vectorized Sigmoid function.
 *
 * Applies the sigmoid function to each element in the input vector 'x'.
 * Returns a new vector with the same size, where each element
 * is the sigmoid of the corresponding input element.
 */
std::vector<float> vectSigmoid(const std::vector<float> x)
{
    std::vector<float> result;
    result.reserve(x.size());
    for (float i : x) {
        result.push_back(sigmoid(i));
    }
    return result;
}

/**
 * Vectorized Sigmoid derivative function.
 *
 * Applies the sigmoid derivative to each element in the input vector 'x'.
 * Returns a new vector with the same size, where each element
 * is the derivative of the sigmoid of that input element.
 */
std::vector<float> vectSigmoidDerivative(const std::vector<float> x)
{
    std::vector<float> result;
    result.reserve(x.size());
    for (float i : x) {
        result.push_back(sigmoidDerivative(i));
    }
    return result;
}

/**
 * ReLU (Rectified Linear Unit) function.
 *
 * Formula:
 *    ReLU(x) = max(0, x)
 *
 * If x > 0, it returns x, otherwise 0. ReLU is very common in deep neural
 * networks because it helps mitigate the vanishing gradient problem.
 */
float relu(float x) {
    return (x > 0.0f) ? x : 0.0f;
}

/**
 * Derivative of ReLU.
 *
 *    ReLU'(x) = 1 if x >= 0
 *              0 otherwise
 *
 * Note that some definitions consider the derivative undefined at x=0,
 * but in practice we often set it to 1 or 0 for x=0 as a convenience.
 */
float reluDerivative(float x) {
    return (x >= 0.0f) ? 1.0f : 0.0f;
}

/**
 * Vectorized ReLU function.
 *
 * Applies the ReLU function to each element in 'x' and returns
 * a new vector. If you only want to fill an existing 'result' vector,
 * see the overload below.
 */
std::vector<float> vectRelu(const std::vector<float> x)
{
    std::vector<float> result;
    result.reserve(x.size());
    for (float i : x) {
        result.push_back(relu(i));
    }
    return result;
}

/**
 * Vectorized ReLU function (in-place variant).
 *
 * This variant takes a reference to 'result' and fills it with the
 * ReLU outputs of each element in 'x'. This can help reduce
 * unnecessary allocations if you already have a vector for 'result'.
 */
void vectRelu(std::vector<float>& result, const std::vector<float>& x)
{
    for (float i : x) {
        result.push_back(relu(i));
    }
}

/**
 * Vectorized derivative of ReLU.
 *
 * Applies the derivative of ReLU to each element in 'x' and returns
 * a new vector. Each element will be 1 if the corresponding element
 * in 'x' is >= 0, otherwise 0.
 */
std::vector<float> vectReluDerivative(const std::vector<float> x)
{
    std::vector<float> result;
    result.reserve(x.size());
    for (float i : x) {
        result.push_back(reluDerivative(i));
    }
    return result;
}

/**
 * Leaky ReLU function.
 *
 * Formula:
 *    LeakyReLU(x) = x if x > 0
 *                   alpha * x otherwise
 *
 * A small alpha allows some negative values to pass through, which
 * helps fix the "dying ReLU" problem.
 */
float leakyRelu(float x, float alpha = 0.01f) {
    return (x > 0.0f) ? x : alpha * x;
}

/**
 * Derivative of Leaky ReLU.
 *
 *    LeakyReLU'(x) = 1 if x >= 0
 *                    alpha otherwise
 */
float leakyReluDerivative(float x, float alpha = 0.01f) {
    return (x >= 0.0f) ? 1.0f : alpha;
}

/**
 * Vectorized Leaky ReLU function.
 *
 * Applies the Leaky ReLU function to each element in 'x',
 * with a specified alpha (default is 0.01).
 */
std::vector<float> vectLeakyRelu(const std::vector<float> x, float alpha = 0.01f)
{
    std::vector<float> result;
    result.reserve(x.size());
    for (float i : x) {
        result.push_back(leakyRelu(i, alpha));
    }
    return result;
}

/**
 * Vectorized derivative of Leaky ReLU.
 *
 * Applies the derivative of Leaky ReLU to each element in 'x',
 * with a specified alpha (default is 0.01).
 */
std::vector<float> vectLeakyReluDerivative(const std::vector<float> x, float alpha = 0.01f)
{
    std::vector<float> result;
    result.reserve(x.size());
    for (float i : x) {
        result.push_back(leakyReluDerivative(i, alpha));
    }
    return result;
}

/**
 * Hyperbolic Tangent function.
 *
 * The standard library provides std::tanh(x). It maps real values
 * into the range (-1, 1).
 */
inline float tanhe(float x) {
    return std::tanh(x);
}

/**
 * Derivative of the Tanh function.
 *
 *    tanh'(x) = 1 - tanh^2(x)
 *
 * This derivative is used when backpropagating errors in neural networks
 * that use tanh as the activation function.
 */
inline float tanhDerivative(float x) {
    return 1.0f - std::pow(std::tanh(x), 2);
}

/**
 * Vectorized Tanh function.
 *
 * Applies std::tanh to each element in the input vector 'x'
 * and returns the transformed values.
 */
std::vector<float> vectTanh(const std::vector<float> x)
{
    std::vector<float> result;
    result.reserve(x.size());
    for (float i : x) {
        result.push_back(tanh(i));
    }
    return result;
}

/**
 * Vectorized Tanh derivative function.
 *
 * Applies the derivative of tanh to each element in the input vector 'x'
 * and returns the result.
 */
std::vector<float> vectTanhDerivative(const std::vector<float> x)
{
    std::vector<float> result;
    result.reserve(x.size());
    for (float i : x) {
        result.push_back(tanhDerivative(i));
    }
    return result;
}

#endif
