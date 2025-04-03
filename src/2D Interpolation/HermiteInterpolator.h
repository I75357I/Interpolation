#ifndef HERMITE_INTERPOLATOR_H
#define HERMITE_INTERPOLATOR_H

#include <vector>
#include <stdexcept>
#include "InterpolatorCommon.h"

namespace interpolation {

    /**
     * @brief HermiteInterpolator class for performing piecewise Hermite interpolation.
     *
     * This class computes tangents using the average slope for interior points and sets
     * boundary tangents to zero.
     *
     * References:
     * - https://en.wikipedia.org/wiki/Hermite_interpolation
     * - https://habr.com/ru/articles/749288/
     *
     * Complexity:
     * - Precomputation (tangent computation): O(n)
     * - Interpolation per point: O(log n) due to binary search.
     */
    class HermiteInterpolator {
    public:
        /**
         * @brief Constructor accepting training points (sorted by x).
         * @param train Vector of training points.
         * @throws std::runtime_error if fewer than 2 points are provided.
         */
        explicit HermiteInterpolator(const std::vector<Point2D>& train);

        /**
         * @brief Predicts interpolated y-values for the provided test points.
         * @param test Vector of test points (only x is used; y is computed).
         * @return Vector of points (x, interpolated y).
         */
        std::vector<Point2D> predict(const std::vector<Point2D>& test) const;

    private:
        std::vector<real> x_train_, y_train_;
        std::vector<real> m_; // Tangents at training points

        /**
         * @brief Computes tangents at the training points.
         * Complexity: O(n)
         */
        void computeTangents();
    };

} // namespace interpolation

#endif // HERMITE_INTERPOLATOR_H
