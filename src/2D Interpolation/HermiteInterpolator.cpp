#include "HermiteInterpolator.h"
#include "InterpolatorCommon.h"
#include <algorithm>
#include <iterator>

namespace interpolation {

    HermiteInterpolator::HermiteInterpolator(const std::vector<Point2D>& train) {
        // Use common validations: need at least 2 points and sorted order.
        validateMinimumPoints(train, 2);
        validateSortedData(train);

        x_train_.reserve(train.size());
        y_train_.reserve(train.size());
        for (const auto& pt : train) {
            x_train_.push_back(pt.first);
            y_train_.push_back(pt.second);
        }
        computeTangents();
    }

    void HermiteInterpolator::computeTangents() {
        const auto n = x_train_.size();
        std::vector<real> tangents(n - 1);
        for (size_t i = 0; i < n - 1; ++i) {
            real h = x_train_[i + 1] - x_train_[i];
            if (h == 0.0) {
                throw std::runtime_error("Zero segment length in Hermite interpolation.");
            }
            tangents[i] = (y_train_[i + 1] - y_train_[i]) / h;
        }
        m_.resize(n, 0.0);
        for (size_t i = 1; i < n - 1; ++i) {
            m_[i] = 0.5 * (tangents[i - 1] + tangents[i]);
        }
        // Boundary tangents m_[0] and m_[n-1] remain 0.
    }

    std::vector<Point2D> HermiteInterpolator::predict(const std::vector<Point2D>& test) const {
        std::vector<Point2D> result;
        result.reserve(test.size());
        for (const auto& pt : test) {
            real xx = pt.first;
            // Find the subinterval: x_train_[j] <= xx < x_train_[j+1]
            auto it = std::lower_bound(x_train_.begin(), x_train_.end(), xx);
            size_t j = (it == x_train_.begin()) ? 0 : (std::distance(x_train_.begin(), it) - 1);
            if (j >= x_train_.size() - 1) {
                j = x_train_.size() - 2; // Clamp to the last interval if out of range
            }
            real h = x_train_[j + 1] - x_train_[j];
            real t = (xx - x_train_[j]) / h; // Normalize t to [0, 1]
            const auto t2 = t * t;
            const auto t3 = t2 * t;
            // Hermite basis functions
            const auto h00 = 2.0 * t3 - 3.0 * t2 + 1.0;
            const auto h10 = t3 - 2.0 * t2 + t;
            const auto h01 = -2.0 * t3 + 3.0 * t2;
            const auto h11 = t3 - t2;
            const auto y_val = h00 * y_train_[j] +
                h10 * h * m_[j] +
                h01 * y_train_[j + 1] +
                h11 * h * m_[j + 1];
            result.push_back({ xx, y_val });
        }
        return result;
    }

} // namespace interpolation
