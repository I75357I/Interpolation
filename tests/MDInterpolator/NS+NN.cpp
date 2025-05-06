#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <cassert>
#include <fstream>
#include <sstream>
#include <chrono>
#include "NearestSimplexInterpolationMD.h"
#include "src/NNInterpolation/NeuralNetwork.h"

/**
 * @brief Demonstrates interpolation methods with neural network error correction and evaluation metrics.
 *
 * This example reads training data, applies Nearest Simplex interpolation,
 * trains a neural network to predict interpolation error,
 * corrects the interpolated values, and computes performance metrics (MAE, MSE, RMSE, percentiles).
 */

// Reads data from a file into a vector of rows (N inputs + 1 output)
template <size_t N>
std::vector<std::array<double, N + 1>> read_data(const char* filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error: unable to open " << filename << "\n";
        std::exit(1);
    }
    std::vector<std::array<double, N + 1>> data;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::array<double, N + 1> row;
        for (size_t i = 0; i < N + 1; ++i) ss >> row[i];
        data.push_back(row);
    }
    return data;
}

// Structure to hold evaluation metrics
struct Metrics {
    double mae = 0;
    double mse = 0;
    double q95 = 0;
    double q97 = 0;
    double r2 = 0;
    double rmse = 0;
    double mrae = 0;
    double rmse_val() const { return std::sqrt(mse); }
};

// Computes metrics comparing predictions to true values
auto calc_metrics = [&](const std::vector<double>& Yp, const std::vector<double>& Ytrue) {
    size_t n = Yp.size();
    Metrics m;
    std::vector<double> errs(n);
    double sum_y_true = 0;
    double sum_squared_y_true = 0;
    double sum_residuals = 0;

    // Calculate relative errors and accumulate for MAE, MSE
    for (size_t i = 0; i < n; ++i) {
        double rel = std::abs((Ytrue[i] - Yp[i]) / Ytrue[i]);
        m.mae += rel;
        m.mse += rel * rel;
        errs[i] = rel;
        sum_y_true += Ytrue[i];
        sum_squared_y_true += Ytrue[i] * Ytrue[i];
        sum_residuals += (Ytrue[i] - Yp[i]) * (Ytrue[i] - Yp[i]);
    }

    // Finalize MAE and MSE, compute RMSE
    m.mae /= n;
    m.mse /= n;
    m.rmse = m.rmse_val();

    // Sort errors to compute percentiles
    std::sort(errs.begin(), errs.end());
    m.q95 = errs[size_t(std::floor(0.95 * n))];
    m.q97 = errs[size_t(std::floor(0.97 * n))];

    return m;
    };

int main() {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    constexpr size_t DIM = 5;
    using Row = std::array<double, DIM + 1>;
    using Point = std::array<double, DIM>;

    // Load training data
    auto train = read_data<DIM>("train_gen.txt");

    // Copy training data into a separate container
    std::vector<Row> train_data;
    for (const auto& row : train) {
        train_data.push_back(row);
    }

    // Create Nearest Simplex interpolator using training data
    NearestSimplexInterpolationMD::Interpolator<DIM, double> interp(train_data);

    // Containers for interpolated values and corresponding true outputs
    std::vector<double> interpolated_values;
    std::vector<Point> Xtrain;
    std::vector<double> Ytrue;

    for (const auto& row : train) {
        Point pt;
        for (size_t d = 0; d < DIM; ++d) {
            pt[d] = row[d];
        }
        double pred = interp.interpolate(pt);      // interpolated value
        double true_val = row[DIM];               // actual value
        interpolated_values.push_back(pred);
        Xtrain.push_back(pt);
        Ytrue.push_back(true_val);
    }

    // Inputs/outputs for neural network training
    using Real = NeuralNetwork::Real;
    using Sample = std::vector<Real>;

    std::vector<Sample> trainInputs;
    std::vector<Sample> trainOutputs;

    // Input is the interpolated value; output is the interpolation error
    for (size_t i = 0; i < interpolated_values.size(); ++i) {
        trainInputs.push_back({ static_cast<Real>(interpolated_values[i]) });
        trainOutputs.push_back({ static_cast<Real>(Ytrue[i] - interpolated_values[i]) });
    }

    // Initialize neural network
    NeuralNetwork nn = NeuralNetwork::DefaultModel();

    // Training parameters
    Real learningRate = 0.001;
    int epochs = 500;

    std::cout << "Training neural network..." << std::endl;
    nn.train(trainInputs, trainOutputs, epochs, learningRate);

    // Use trained network to correct interpolated values
    std::vector<double> predictions;
    for (const auto& pt : Xtrain) {
        double base_interp = interp.interpolate(pt);
        Sample correction = nn.predict({ static_cast<Real>(base_interp) });
        double corrected = base_interp + static_cast<double>(correction[0]);
        predictions.push_back(corrected);
    }

    // Compute evaluation metrics
    Metrics metrics = calc_metrics(predictions, Ytrue);

    // Measure total duration
    auto t1 = Clock::now();
    double dt = std::chrono::duration<double>(t1 - t0).count();

    // Display metrics
    std::cout << "\nEvaluation Metrics:" << std::endl;
    std::cout << "MAE: " << metrics.mae << std::endl;
    std::cout << "MSE: " << metrics.mse << std::endl;
    std::cout << "RMSE: " << metrics.rmse << std::endl;
    std::cout << "95th percentile error: " << metrics.q95 << std::endl;
    std::cout << "97th percentile error: " << metrics.q97 << std::endl;
    std::cout << "Total duration: " << dt << " seconds" << std::endl;

    return 0;
}