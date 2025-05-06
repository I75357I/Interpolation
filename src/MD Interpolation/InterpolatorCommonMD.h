#ifndef INTERPOLATOR_COMMON_MD_H
#define INTERPOLATOR_COMMON_MD_H

#include <vector>
#include <array>
#include <stdexcept>
#include <cmath>
#include <algorithm>

/**
 * @brief Common utilities for M-dimensional interpolation.
 *
 * Provides point and simplex definitions, linear solver, and input validation.
 */
namespace InterpolatorCommonMD {

    /// D-dimensional point type alias
    template <size_t D, typename Real = double>
    using PointND = std::array<Real, D>;

    /// Simplex in D dimensions: D+1 vertices
    template <size_t D, typename Real = double>
    struct Simplex {
        std::array<PointND<D, Real>, D + 1> vert;
        bool operator==(const Simplex& other) const noexcept {
            return vert == other.vert;
        }
    };

    /**
     * @brief Solve a D×D linear system A * x = b using Gaussian elimination with partial pivoting.
     * @param A The coefficient matrix.
     * @param b The right-hand side vector.
     * @return Solution vector x.
     * @throws std::runtime_error if the matrix is singular.
     */
    template <size_t D, typename Real>
    std::array<Real, D> solveLinear(
        std::array<std::array<Real, D>, D> A,
        std::array<Real, D> b
    ) {
        for (size_t i = 0; i < D; ++i) {
            // Partial pivoting
            size_t pivot = i;
            for (size_t j = i + 1; j < D; ++j) {
                if (std::fabs(A[j][i]) > std::fabs(A[pivot][i])) {
                    pivot = j;
                }
            }
            std::swap(A[i], A[pivot]);
            std::swap(b[i], b[pivot]);

            Real diag = A[i][i];
            if (std::fabs(diag) < Real(1e-15)) {
                throw std::runtime_error("Singular matrix in solveLinear");
            }

            // Normalize pivot row
            for (size_t k = i; k < D; ++k) {
                A[i][k] /= diag;
            }
            b[i] /= diag;

            // Eliminate
            for (size_t j = 0; j < D; ++j) {
                if (j == i) continue;
                Real factor = A[j][i];
                for (size_t k = i; k < D; ++k) {
                    A[j][k] -= factor * A[i][k];
                }
                b[j] -= factor * b[i];
            }
        }
        return b;
    }

    /**
     * @brief Check that the dataset contains at least M+1 points.
     *        Overload for spatial point sets (size M).
     * @tparam M Spatial dimension.
     * @tparam Real Floating-point type.
     * @param points Vector of spatial points: each is PointND<M,Real>.
     * @param minPts Minimum required points (default M+1).
     * @return true if points.size() >= minPts.
     */
    template <size_t M, typename Real = double>
    inline bool checkMinimumPoints(
        const std::vector<PointND<M, Real>>& points,
        size_t minPts = M + 1
    ) {
        return points.size() >= minPts;
    }

    /**
     * @brief Check that the dataset contains at least M+1 training points.
     *        Overload for training point sets (size M+1 arrays: coords + value).
     * @tparam M Spatial dimension.
     * @tparam Real Floating-point type.
     * @param points Vector of training points: each array of size M+1 (coords + value).
     * @param minPts Minimum required points (default M+1).
     * @return true if points.size() >= minPts.
     */
    template <size_t M, typename Real = double>
    inline bool checkMinimumPoints(
        const std::vector<std::array<Real, M + 1>>& points,
        size_t minPts = M + 1
    ) {
        return points.size() >= minPts;
    }

    /**
     * @brief Validate that spatial coordinates are unique (no duplicates within tolerance).
     * @throws std::invalid_argument if duplicates found.
     */
    template <size_t M, typename Real = double>
    inline void validateUniqueSpatial(
        const std::vector<std::array<Real, M + 1>>& points,
        Real tol = static_cast<Real>(1e-14)
    ) {
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = i + 1; j < points.size(); ++j) {
                bool same = true;
                for (size_t k = 0; k < M; ++k) {
                    if (std::fabs(points[i][k] - points[j][k]) > tol) {
                        same = false;
                        break;
                    }
                }
                if (same) {
                    throw std::invalid_argument(
                        "Duplicate spatial coordinates found in training data.");
                }
            }
        }
    }

    /**
     * @brief Check if a query point lies within the bounding box of training data.
     * @param query Array of M spatial coordinates.
     * @param points Vector of training points: each size M+1 (coords + value).
     * @param tol Tolerance for bounding-box comparison.
     * @return true if query in [min - tol, max + tol] for all dims.
     */
    template <size_t M, typename Real = double>
    inline bool isPointWithinBoundingBox(
        const PointND<M, Real>& query,
        const std::vector<std::array<Real, M + 1>>& points,
        Real tol = static_cast<Real>(1e-14)
    ) {
        if (points.empty()) return false;
        PointND<M, Real> minP = {};
        PointND<M, Real> maxP = {};
        // initialize bounds
        for (size_t k = 0; k < M; ++k) {
            minP[k] = points[0][k];
            maxP[k] = points[0][k];
        }
        // compute bounds
        for (const auto& pt : points) {
            for (size_t k = 0; k < M; ++k) {
                minP[k] = std::min(minP[k], pt[k]);
                maxP[k] = std::max(maxP[k], pt[k]);
            }
        }
        // check query
        for (size_t k = 0; k < M; ++k) {
            if (query[k] < minP[k] - tol || query[k] > maxP[k] + tol) {
                return false;
            }
        }
        return true;
    }

} // namespace InterpolatorCommonMD

#endif // INTERPOLATOR_COMMON_MD_H