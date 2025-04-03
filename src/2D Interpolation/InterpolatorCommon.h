#ifndef INTERPOLATOR_COMMON_H
#define INTERPOLATOR_COMMON_H

#include <utility>
#include <vector>
#include <stdexcept>
#include <algorithm>
#include <string>

namespace interpolation {

    // Type alias for real numbers
    using real = double;

    // A point in 2D space: first = x, second = y
    using Point2D = std::pair<real, real>;

    /**
     * @brief Validates that the training data has at least minPoints points.
     * @param train Vector of training points.
     * @param minPoints Minimum number of points required.
     * @throws std::runtime_error if there are fewer than minPoints points.
     */
    inline void validateMinimumPoints(const std::vector<Point2D>& train, size_t minPoints = 2) {
        if (train.size() < minPoints) {
            throw std::runtime_error("Training data must have at least " + std::to_string(minPoints) + " points.");
        }
    }

    /**
     * @brief Validates that the training data is sorted in ascending order by x.
     * @param train Vector of training points.
     * @throws std::runtime_error if training points are not sorted by x.
     */
    inline void validateSortedData(const std::vector<Point2D>& train) {
        if (!std::is_sorted(train.begin(), train.end(),
            [](const Point2D& a, const Point2D& b) { return a.first < b.first; })) {
            throw std::runtime_error("Training data must be sorted in ascending order by x.");
        }
    }

    /**
     * @brief Validates that the x values in the given vector are strictly increasing.
     * @param x Vector of x values.
     * @throws std::runtime_error if any x value is not strictly greater than its predecessor.
     */
    inline void validateStrictlyIncreasing(const std::vector<real>& x) {
        for (size_t i = 1; i < x.size(); ++i) {
            if (x[i] <= x[i - 1]) {
                throw std::runtime_error("x values must be strictly increasing.");
            }
        }
    }

} // namespace interpolation

#endif // INTERPOLATOR_COMMON_H
