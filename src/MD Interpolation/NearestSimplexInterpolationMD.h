#ifndef NEAREST_SIMPLEX_INTERPOLATION_MD_H
#define NEAREST_SIMPLEX_INTERPOLATION_MD_H

#include "InterpolatorCommonMD.h"
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <limits>

/**
 * @brief Nearest-simplex interpolation in M dimensions: selects the M+1 closest points
 *        to form a simplex, performs barycentric interpolation, and falls back
 *        to nearest-neighbor on failure.
 */
namespace NearestSimplexInterpolationMD {
    template <size_t M, typename Real = double>
    class Interpolator {
    public:
        using PointND = InterpolatorCommonMD::PointND<M, Real>;
        using TrainingPt = std::array<Real, M + 1>;  // first M coords, last is value

        /**
         * @brief Construct with training data.
         */
        explicit Interpolator(const std::vector<TrainingPt>& trainPoints) {
            InterpolatorCommonMD::validateUniqueSpatial<M, Real>(trainPoints);
            m_trainPoints = trainPoints;
        }

        /**
         * @brief Interpolate the value at x.
         *        Uses barycentric weights on the nearest simplex; nearest-neighbor fallback.
         */
        Real interpolate(const PointND& x) const {
            // compute distances to all training points
            std::vector<std::pair<Real, size_t>> dists;
            dists.reserve(m_trainPoints.size());
            for (size_t i = 0; i < m_trainPoints.size(); ++i) {
                Real d2 = Real(0);
                for (size_t k = 0; k < M; ++k) {
                    Real diff = m_trainPoints[i][k] - x[k];
                    d2 += diff * diff;
                }
                dists.emplace_back(d2, i);
            }
            // select M+1 nearest
            std::nth_element(dists.begin(), dists.begin() + M, dists.end());
            std::array<PointND, M + 1> verts;
            std::array<Real, M + 1> values;
            for (size_t j = 0; j <= M; ++j) {
                auto& pr = dists[j];
                const auto& tp = m_trainPoints[pr.second];
                for (size_t k = 0; k < M; ++k) {
                    verts[j][k] = tp[k];
                }
                values[j] = tp[M];
            }
            // attempt barycentric interpolation
            return barycentric(x, verts, values);
        }

        /**
         * @brief Batch predict for multiple points.
         */
        std::vector<TrainingPt> predict(const std::vector<PointND>& points) const {
            std::vector<TrainingPt> out;
            out.reserve(points.size());
            for (auto& x : points) {
                auto tp = TrainingPt();
                for (size_t k = 0; k < M; ++k) tp[k] = x[k];
                tp[M] = interpolate(x);
                out.push_back(tp);
            }
            return out;
        }

    private:
        std::vector<TrainingPt> m_trainPoints;

        /**
         * @brief Compute barycentric coordinates in M-simplex.
         * @throws std::runtime_error on degenerate simplex or coords outside.
         */
        Real barycentric(
            const PointND& x,
            const std::array<PointND, M + 1>& verts,
            const std::array<Real, M + 1>& values
        ) const {
            // build linear system for weights w[1..M], w[0] = 1 - sum
            std::array<std::array<Real, M>, M> A{};
            std::array<Real, M> b{};
            // base vertex
            const auto& v0 = verts[0];
            for (size_t i = 0; i < M; ++i) {
                b[i] = x[i] - v0[i];
                for (size_t j = 0; j < M; ++j) {
                    A[i][j] = verts[j + 1][i] - v0[i];
                }
            }
            auto w_vec = InterpolatorCommonMD::solveLinear<M, Real>(A, b);
            std::array<Real, M + 1> w_all;
            Real sum = Real(0);
            for (size_t i = 0; i < M; ++i) {
                w_all[i + 1] = w_vec[i];
                sum += w_vec[i];
            }
            w_all[0] = Real(1) - sum;

            // interpolate value
            Real interp = Real(0);
            for (size_t i = 0; i <= M; ++i) {
                interp += w_all[i] * values[i];
            }
            return interp;
        }
    };

} // namespace NearestSimplexInterpolationMD

#endif // NEAREST_SIMPLEX_INTERPOLATION_MD_H
