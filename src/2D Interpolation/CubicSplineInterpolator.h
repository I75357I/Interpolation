#ifndef CUBIC_SPLINE_INTERPOLATOR_H
#define CUBIC_SPLINE_INTERPOLATOR_H

#include <vector>
#include <stdexcept>
#include <memory>
#include "InterpolatorCommon.h"

namespace interpolation {

    /**
     * @brief CubicSplineInterpolator class for performing cubic spline interpolation.
     *
     * Implements the standard cubic spline algorithm (see [Numerical Recipes](https://numerical.recipes)
     * and the [Wikipedia article on Spline Interpolation](https://en.wikipedia.org/wiki/Spline_interpolation)).
     *
     * Complexity:
     * - Coefficient computation: O(n)
     * - Interpolation per point: O(log n) due to binary search.
     */
    class CubicSplineInterpolator {
    public:
        /**
         * @brief Constructor accepting training points (sorted by x).
         * @param train Vector of training points.
         * @throws std::invalid_argument if there are fewer than 3 points or if x-values are not strictly increasing.
         */
        explicit CubicSplineInterpolator(const std::vector<Point2D>& train);

        /**
         * @brief Interpolates the spline at a given x value.
         * @param xi The x-value for interpolation.
         * @return The interpolated y-value.
         */
        real interpolate(real xi) const;

        /**
         * @brief Predicts y-values for a set of test points.
         * @param test Vector of test points (only x is used; y is computed).
         * @return Vector of points (x, interpolated y).
         */
        std::vector<Point2D> predict(const std::vector<Point2D>& test) const;

    private:
        std::vector<real> x_, y_;
        // Spline coefficients: b, c, d (a equals y)
        std::vector<real> b_, c_, d_;

        /**
         * @brief Computes the cubic spline coefficients.
         * Complexity: O(n)
         */
        void computeCoefficients();
    };

} // namespace interpolation

#endif // CUBIC_SPLINE_INTERPOLATOR_H
