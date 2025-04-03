#include "CubicSplineInterpolator.h"
#include "InterpolatorCommon.h"
#include <algorithm>
#include <iterator>

namespace interpolation {

    CubicSplineInterpolator::CubicSplineInterpolator(const std::vector<Point2D>& train) {
        // Use common validations from InterpolatorCommon.h
        validateMinimumPoints(train, 3);
        validateSortedData(train);

        x_.reserve(train.size());
        y_.reserve(train.size());
        for (const auto& pt : train) {
            x_.push_back(pt.first);
            y_.push_back(pt.second);
        }
        // Additional check: x values must be strictly increasing.
        validateStrictlyIncreasing(x_);
        computeCoefficients();
    }

    real CubicSplineInterpolator::interpolate(real xi) const {
        // Find the correct interval using binary search (O(log n))
        auto it = std::upper_bound(x_.begin(), x_.end(), xi);
        size_t j = (it == x_.begin()) ? 0 : (std::distance(x_.begin(), it) - 1);
        if (j >= x_.size() - 1) {
            j = x_.size() - 2;  // Clamp to the last interval for extrapolation
        }
        auto dx = xi - x_[j];
        return y_[j] + b_[j] * dx + c_[j] * dx * dx + d_[j] * dx * dx * dx;
    }

    std::vector<Point2D> CubicSplineInterpolator::predict(const std::vector<Point2D>& test) const {
        std::vector<Point2D> result;
        result.reserve(test.size());
        for (const auto& pt : test) {
            result.push_back({ pt.first, interpolate(pt.first) });
        }
        return result;
    }

    void CubicSplineInterpolator::computeCoefficients() {
        const auto n = x_.size();
        std::vector<real> h(n - 1);
        for (size_t i = 0; i < n - 1; ++i) {
            h[i] = x_[i + 1] - x_[i];
        }

        std::vector<real> alpha(n, 0.0);
        for (size_t i = 1; i < n - 1; ++i) {
            alpha[i] = (3.0 / h[i]) * (y_[i + 1] - y_[i])
                - (3.0 / h[i - 1]) * (y_[i] - y_[i - 1]);
        }

        std::vector<real> l(n), mu(n), z(n);
        l[0] = 1.0;
        mu[0] = 0.0;
        z[0] = 0.0;

        for (size_t i = 1; i < n - 1; ++i) {
            l[i] = 2.0 * (x_[i + 1] - x_[i - 1]) - h[i - 1] * mu[i - 1];
            if (l[i] == 0.0) {
                throw std::runtime_error("Division by zero encountered while computing spline coefficients.");
            }
            mu[i] = h[i] / l[i];
            z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i];
        }
        l[n - 1] = 1.0;
        z[n - 1] = 0.0;
        c_.resize(n);
        b_.resize(n - 1);
        d_.resize(n - 1);
        c_[n - 1] = 0.0;

        // Back substitution: O(n)
        for (int j = static_cast<int>(n) - 2; j >= 0; --j) {
            c_[j] = z[j] - mu[j] * c_[j + 1];
            b_[j] = (y_[j + 1] - y_[j]) / h[j] - h[j] * (c_[j + 1] + 2.0 * c_[j]) / 3.0;
            d_[j] = (c_[j + 1] - c_[j]) / (3.0 * h[j]);
        }
    }

} // namespace interpolation
