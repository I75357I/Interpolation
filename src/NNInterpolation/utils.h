#ifndef UTILS_HH
#define UTILS_HH

#include <vector>
#include <random>
#include <functional>
#include <algorithm>
#include <chrono>

/**
 * @brief Initializes a C-style 2D array of weights with uniform random values.
 *
 * This function fills the provided C-style 2D array (weights) with random
 * float values uniformly distributed between -1.0 and 1.0.
 *
 * @param rows   Number of rows in the weights array.
 * @param cols   Number of columns in the weights array.
 * @param weights Pointer to the C-style 2D array of weights.
 */
void uniformWeightInitializer(int rows, int cols, float** weights) {
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Fill the C-style array with random values.
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            weights[i][j] = dis(gen);
        }
    }
}

/**
 * @brief Initializes a C-style array of bias values with uniform random values.
 *
 * This function fills the provided C-style array (bias) with random float
 * values uniformly distributed between -1.0 and 1.0.
 *
 * @param size Number of bias values.
 * @param bias Pointer to the C-style array to store bias values.
 */
void biasInitializer(int size, float* bias) {
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    // Fill the bias array with random values.
    for (int i = 0; i < size; ++i) {
        bias[i] = dis(gen);
    }
}

/**
 * @brief Creates and returns a 2D vector (matrix) of weights with uniform random values.
 *
 * This function allocates a 2D vector of dimensions [rows x cols] and fills it with random
 * float values uniformly distributed between -1.0 and 1.0.
 *
 * @param rows Number of rows (typically corresponding to output neurons).
 * @param cols Number of columns (typically corresponding to input neurons).
 * @return std::vector<std::vector<float>> The initialized weight matrix.
 */
std::vector<std::vector<float>> uniformWeightInitializer(int rows, int cols)
{
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<std::vector<float>> weights(rows, std::vector<float>(cols));

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            weights[i][j] = dis(gen);
        }
    }

    return weights;
}

/**
 * @brief Creates and returns a vector of bias values with uniform random values.
 *
 * This function allocates a vector of the given size and fills it with random float values
 * uniformly distributed between -1.0 and 1.0.
 *
 * @param size Number of bias values.
 * @return std::vector<float> The initialized bias vector.
 */
std::vector<float> biasInitailizer(int size)
{
    std::random_device rd;
    std::mt19937 gen(rd() ^ std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_real_distribution<> dis(-1.0, 1.0);

    std::vector<float> bias(size);

    for (int i = 0; i < size; ++i)
    {
        bias[i] = dis(gen);
    }
    return bias;
}

/**
 * @brief Computes the dot product of two vectors.
 *
 * This function computes the dot product (sum of element-wise multiplications)
 * of two vectors v1 and v2.
 *
 * @param v1 Reference to the first vector.
 * @param v2 Reference to the second vector.
 * @return float The resulting dot product.
 */
float dotProduct(std::vector<float>& v1, std::vector<float>& v2)
{
    float result = 0;
    for (int i = 0; i < v1.size(); i++)
    {
        result += v1[i] * v2[i];
    }
    return result;
}

/**
 * @brief Computes the dot product of a C-style array and a vector.
 *
 * This function computes the dot product between a C-style array of floats and a vector.
 *
 * @param v1 Pointer to the C-style array.
 * @param v2 Reference to the vector.
 * @return float The resulting dot product.
 */
float dotProduct(float* v1, std::vector<float>& v2)
{
    float result = 0;
    for (int i = 0; i < v2.size(); i++)
    {
        result += v1[i] * v2[i];
    }
    return result;
}

/**
 * @brief Transposes a 2D vector (matrix).
 *
 * This function returns the transpose of the input matrix. The number of rows becomes the
 * number of columns and vice versa.
 *
 * @param m Reference to the input matrix.
 * @return std::vector<std::vector<float>> The transposed matrix.
 */
std::vector<std::vector<float>> transpose(std::vector<std::vector<float>>& m)
{
    std::vector<std::vector<float>> trans_vec(m[0].size(), std::vector<float>());

    for (int i = 0; i < m.size(); i++)
    {
        for (int j = 0; j < m[i].size(); j++)
        {
            if (trans_vec[j].size() != m.size())
                trans_vec[j].resize(m.size());
            trans_vec[j][i] = m[i][j];
        }
    }
    return trans_vec;
}

/**
 * @brief Subtracts two vectors element-wise.
 *
 * This function returns a new vector that is the element-wise difference of v1 and v2 (v1 - v2).
 *
 * @param v1 Reference to the first vector.
 * @param v2 Reference to the second vector.
 * @return std::vector<float> The result of the subtraction.
 */
std::vector<float> subtract(std::vector<float>& v1, std::vector<float>& v2)
{
    std::vector<float> out;
    std::transform(v1.begin(), v1.end(), v2.begin(), std::back_inserter(out), std::minus<float>());
    return out;
}

/**
 * @brief Subtracts two vectors element-wise and stores the result in the provided vector.
 *
 * This function computes the element-wise difference between v1 and v2 (v1 - v2) and appends
 * the result to the vector 'out'.
 *
 * @param out Reference to the vector where the result will be stored.
 * @param v1 Reference to the first vector.
 * @param v2 Reference to the second vector.
 */
void subtract(std::vector<float>& out, const std::vector<float>& v1, const std::vector<float>& v2)
{
    std::transform(v1.begin(), v1.end(), v2.begin(), std::back_inserter(out), std::minus<float>());
}

/**
 * @brief Multiplies each element of a vector by a scalar.
 *
 * This function performs an in-place multiplication of each element in the vector by the scalar value.
 *
 * @param v Reference to the vector to be scaled.
 * @param scalar The scalar value to multiply each element.
 * @return std::vector<float> The vector after scalar multiplication.
 */
std::vector<float> scalarVectorMultiplication(std::vector<float>& v, float scalar)
{
    std::transform(v.begin(), v.end(), v.begin(), std::bind(std::multiplies<float>(), std::placeholders::_1, scalar));
    return v;
}

#endif
