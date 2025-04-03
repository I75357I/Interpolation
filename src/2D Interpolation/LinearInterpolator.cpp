#include "LinearInterpolator.h"
#include "InterpolatorCommon.h"
#include <algorithm>

namespace interpolation {

    LinearInterpolator::LinearInterpolator(const std::vector<Point2D>& train) {
        // Validate training data using common functions.
        validateMinimumPoints(train, 2);
        validateSortedData(train);
        train_ = train;
    }

    std::vector<Point2D> LinearInterpolator::predict(const std::vector<Point2D>& test) const {
        std::vector<Point2D> result;
        result.reserve(test.size());
        for (const auto& pt : test) {
            const auto x = pt.first;
            // Use binary search to find the lower bound (O(log n))
            auto lower = std::lower_bound(train_.begin(), train_.end(), x,
                [](const Point2D& a, const real& val) { return a.first < val; });
            if (lower == train_.end()) {
                // x is greater than the largest training x; use the last interval.
                lower = train_.end() - 1;
            }
            if (lower == train_.begin()) {
                // x is less than the smallest training x; interpolate between the first two points.
                const auto upper = train_.begin() + 1;
                const auto x1 = train_.front().first;
                const auto y1 = train_.front().second;
                const auto x2 = upper->first;
                const auto y2 = upper->second;
                if (x2 == x1) {
                    throw std::runtime_error("Division by zero: training x values are equal.");
                }
                const auto ratio = (x - x1) / (x2 - x1);
                const auto y_val = y1 + ratio * (y2 - y1);
                result.push_back({ x, y_val });
            }
            else {
                // Interpolate between the two points surrounding x.
                const auto upper = lower;
                const auto prev = lower - 1;
                const auto x1 = prev->first;
                const auto y1 = prev->second;
                const auto x2 = upper->first;
                const auto y2 = upper->second;
                if (x2 == x1) {
                    throw std::runtime_error("Division by zero: training x values are equal.");
                }
                const auto ratio = (x - x1) / (x2 - x1);
                const auto y_val = y1 + ratio * (y2 - y1);
                result.push_back({ x, y_val });
            }
        }
        return result;
    }

} // namespace interpolation
