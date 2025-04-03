#ifndef LOSSES_HH
#define LOSSES_HH

#include <vector>
#include <cmath>

#include "utils.h"

/**
 * Mean Squared Error (MSE) Loss
 *
 * MSE = (1 / N) * ? (y_i - x_i)^2
 *
 * @param true_label Vector of ground-truth values.
 * @param pred Vector of model predictions.
 * @return Computed MSE loss (float).
 */
float MSELoss(const std::vector<float>& true_label, const std::vector<float>& pred)
{
    float sum = 0.0f;
    for (int i = 0; i < true_label.size(); i++)
    {
        // Accumulate (true_label[i] - pred[i])^2
        sum += std::pow(true_label[i] - pred[i], 2.0f);
    }

    int size = static_cast<int>(true_label.size());
    float loss = (1.0f / static_cast<float>(size)) * sum;
    return loss;
}

/**
 * Derivative of Mean Squared Error (MSE) Loss
 *
 * If MSE = (1 / N) * ? (y_i - x_i)^2,
 * the derivative w.r.t. x_i is:
 *      d(MSE)/dx_i = 2 * (x_i - y_i) / N   (often there's a 2 factor, or 2/N factor)
 *
 * The code below uses two functions from "utils.h":
 *   - subtract(sub, pred, true_label) -> sub = pred - true_label (element-wise)
 *   - scalarVectorMultiplication(sub, 2) -> multiplies each element by 2
 *
 * @param true_label Vector of ground-truth values.
 * @param pred Vector of model predictions.
 * @return Vector of partial derivatives of MSE w.r.t. each predicted element.
 */
std::vector<float> MSELossDerivative(const std::vector<float>& true_label, const std::vector<float>& pred)
{
    // sub = (pred - true_label)
    std::vector<float> sub;
    subtract(sub, pred, true_label);

    // dev = sub * 2
    std::vector<float> dev = scalarVectorMultiplication(sub, 2.0f);

    return dev;
}

#endif