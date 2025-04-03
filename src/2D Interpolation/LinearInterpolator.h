#ifndef LINEAR_INTERPOLATOR_H
#define LINEAR_INTERPOLATOR_H

#include <vector>
#include <stdexcept>
#include "InterpolatorCommon.h"

namespace interpolation {

    /**
     * @brief LinearInterpolator class for performing linear interpolation.
     *
     * Uses standard linear interpolation between training points.
     *
     * References:
     * - [Linear interpolation](https://en.wikipedia.org/wiki/Linear_interpolation)
     * - "Numerical Analysis" by Burden & Faires
     *
     * Complexity: Each interpolation uses binary search (O(log n) per test point).
     */
    class LinearInterpolator {
    public:
        /**
         * @brief Constructor accepting training points (sorted by x).
         * @param train Vector of training points.
         * @throws std::runtime_error if there are fewer than 2 training points.
         */
        explicit LinearInterpolator(const std::vector<Point2D>& train);

        /**
         * @brief Predicts interpolated y-values for the provided test points.
         * @param test Vector of test points (only x is used; y is computed).
         * @return Vector of points (x, interpolated y).
         */
        std::vector<Point2D> predict(const std::vector<Point2D>& test) const;

    private:
        std::vector<Point2D> train_;
    };

} // namespace interpolation

#endif // LINEAR_INTERPOLATOR_H
