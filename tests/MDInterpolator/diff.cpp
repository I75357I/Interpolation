#include <iostream>
#include <fstream>
#include <vector>
#include <array>
#include <string>
#include <sstream>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <limits>

#include "DelauneInterpolation_MD.h"
#include "NearestSimplexInterpolationMD.h"
#include "src/NNInterpolation/NeuralNetwork.h"

/**
 * @brief Demonstrates and compares multiple multidimensional interpolation methods.
 *
 * This program reads training and test datasets, applies three interpolation techniques:
 *  - Simple Nearest Neighbor
 *  - Nearest Simplex (Delaunay-based)
 *  - Delaunay interpolation
 * For each method, it predicts test values, computes error metrics (MAE, MSE, RMSE, 95th/97th percentiles),
 * and measures execution time.
 */

// --- Simple Nearest Neighbor ---
template <size_t N, typename T>
class NearestNeighborMD {
public:
    using Point = std::array<T, N>;
    using Row = std::array<T, N + 1>;

    NearestNeighborMD(const std::vector<Row>& train) : train_(train) {}

    T interpolate(const Point& p) const {
        T bestDist2 = std::numeric_limits<T>::infinity();
        T bestVal = T{};
        for (auto& row : train_) {
            T d2 = T{};
            for (size_t i = 0; i < N; ++i) {
                T d = row[i] - p[i];
                d2 += d * d;
            }
            if (d2 < bestDist2) {
                bestDist2 = d2;
                bestVal = row[N];
            }
        }
        return bestVal;
    }
private:
    std::vector<Row> train_;
};

// --- Reading data ---
template <size_t N>
std::vector<std::array<double, N + 1>> read_data(const char* filename) {
    std::ifstream in(filename);
    if (!in) {
        std::cerr << "Error: cannot open " << filename << "\n";
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

int main() {
    constexpr size_t DIM = 5;
    using Row = std::array<double, DIM + 1>;
    using Point = std::array<double, DIM>;

    // 1) Read train and test sets
    auto train = read_data<DIM>("train_gen.txt");
    auto test = read_data<DIM>("test_gen.txt");

    // 2) Prepare Xtest and Ytrue
    std::vector<Point>  Xtest(test.size());
    std::vector<double> Ytrue(test.size());
    for (size_t i = 0; i < test.size(); ++i) {
        for (size_t d = 0; d < DIM; ++d) {
            Xtest[i][d] = test[i][d];
        }
        Ytrue[i] = test[i][DIM];
    }

    // 3) Metrics computation
    struct Metrics {
        double mae = 0, mse = 0, q95 = 0, q97 = 0;
        double rmse() const { return std::sqrt(mse); }
    };
    auto calc = [&](const std::vector<double>& Yp) {
        size_t n = Yp.size();
        Metrics m;
        std::vector<double> errs(n);
        for (size_t i = 0; i < n; ++i) {
            double rel = std::abs((Ytrue[i] - Yp[i]) / Ytrue[i]);
            m.mae += rel;
            m.mse += rel * rel;
            errs[i] = rel;
        }
        m.mae /= n;
        m.mse /= n;
        std::sort(errs.begin(), errs.end());
        m.q95 = errs[size_t(std::floor(0.95 * n))];
        m.q97 = errs[size_t(std::floor(0.97 * n))];
        return m;
        };

    // Output header
    std::cout << "Method\tMAE\tMSE\tRMSE\tQ95\tQ97\tTime(s)\n";

    // Prediction buffers
    std::vector<double> y_nn(test.size()),
        y_smp(test.size()),
        y_del(test.size()),
        y_nnnet(test.size());

    // A) Nearest Neighbor
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        NearestNeighborMD<DIM, double> interp(train);
        for (size_t i = 0; i < test.size(); ++i) {
            y_nn[i] = interp.interpolate(Xtest[i]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        Metrics M = calc(y_nn);
        double dt = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "NearestNeighbor\t"
            << M.mae << "\t"
            << M.mse << "\t"
            << M.rmse() << "\t"
            << M.q95 << "\t"
            << M.q97 << "\t"
            << dt << "\n";
    }

    // B) Nearest Simplex
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        NearestSimplexInterpolationMD::Interpolator<DIM, double> interp(train);
        for (size_t i = 0; i < test.size(); ++i) {
            y_smp[i] = interp.interpolate(Xtest[i]);
        }
        auto t1 = std::chrono::high_resolution_clock::now();

        Metrics M = calc(y_smp);
        double dt = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "NearestSimplex\t"
            << M.mae << "\t"
            << M.mse << "\t"
            << M.rmse() << "\t"
            << M.q95 << "\t"
            << M.q97 << "\t"
            << dt << "\n";
    }

    // C) Delaunay
    {
        using Clock = std::chrono::high_resolution_clock;
        auto t0 = Clock::now();

        DelauneInterpolationMD::Interpolator<DIM, double> interp(train);
        std::vector<double> Yp;
        Yp.reserve(Xtest.size());
        for (size_t i = 0; i < Xtest.size(); ++i) {
            Yp.push_back(interp.interpolate(Xtest[i]));
        }
        auto t1 = Clock::now();

        Metrics M = calc(Yp);
        double dt = std::chrono::duration<double>(t1 - t0).count();
        std::cout << "Delaunay\t"
            << M.mae << "\t"
            << M.mse << "\t"
            << M.rmse() << "\t"
            << M.q95 << "\t"
            << M.q97 << "\t"
            << dt << "\n";
    }

    return 0;
}