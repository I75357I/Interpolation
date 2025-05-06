#ifndef DELAUNE_INTERPOLATION_MD_H
#define DELAUNE_INTERPOLATION_MD_H

#include <array>
#include <vector>
#include <stdexcept>
#include <limits>
#include "InterpolatorCommonMD.h"
#include "Delaune_BW_MD.h"

/*
 * DelauneInterpolationMD
 * -----------------------
 * Implements D-dimensional interpolation using Delaunay triangulation and barycentric coordinates. 
 *
 * Complexity:
 * - Triangulation: O(n log n) generally and O(n^2) in the worst case (using Bowyer-Watson algorithm for D-dimensional space).
 * - Interpolating a point: O(n log n) for finding nearest points and O(D^3) for barycentric interpolation.
 * - Prediction: O(m * (n log n + D^3)), where m is the number of points to predict.
 *
 */

namespace DelauneInterpolationMD {

    template <size_t D, typename Real = double>
    class Interpolator {
    public:
        using PointND = InterpolatorCommonMD::PointND<D, Real>;
        using TrainingPt = std::array<Real, D + 1>;
        using Simplex = InterpolatorCommonMD::Simplex<D, Real>;

        explicit Interpolator(const std::vector<TrainingPt>& trainPoints) {
            if (trainPoints.size() < D + 1)
                throw std::invalid_argument("Need at least D+1 training points");
            InterpolatorCommonMD::validateUniqueSpatial<D, Real>(trainPoints);
            m_trainPoints = trainPoints;
            std::vector<PointND> spatial;
            spatial.reserve(m_trainPoints.size());
            for (auto& tp : m_trainPoints) {
                PointND p;
                for (size_t i = 0; i < D; ++i)
                    p[i] = tp[i];
                spatial.push_back(p);
            }
            m_simplices = DelauneUtils::bowyerWatsonND<D, Real>(spatial);
        }

        
        //Interpolate at point x
        
        Real interpolate(const PointND& x) const {
            if (m_simplices.empty()) {
                std::vector<TrainingPt> nearestPoints = findNearestPoints(x);
                Simplex nearestSimplex = createSimplexFromPoints(nearestPoints);
                auto w = barycentricWeights(x, nearestSimplex);
                return applyWeights(w, nearestSimplex);
            }

            const auto& s = m_simplices.front();
            auto w = barycentricWeights(x, s);
            return applyWeights(w, s);
        }


        
        //Predict: overwrites last element of each point with interpolated z.
         
        std::vector<TrainingPt> predict(const std::vector<TrainingPt>& points) const {
            std::vector<TrainingPt> out;
            out.reserve(points.size());
            for (auto pt : points) {
                PointND x;
                for (size_t i = 0; i < D; ++i)
                    x[i] = pt[i];
                pt[D] = interpolate(x);
                out.push_back(pt);
            }
            return out;
        }

        std::vector<TrainingPt> findNearestPoints(const PointND& x) const {
            std::vector<std::pair<Real, TrainingPt>> distances;

            for (const auto& tp : m_trainPoints) {
                Real dist = 0;
                for (size_t i = 0; i < D; ++i) {
                    dist += std::pow(tp[i] - x[i], 2);
                }
                dist = std::sqrt(dist);
                distances.push_back({ dist, tp });
            }

            std::sort(distances.begin(), distances.end(),
                [](const std::pair<Real, TrainingPt>& a, const std::pair<Real, TrainingPt>& b) {
                    return a.first < b.first;
                });

            std::vector<TrainingPt> nearestPoints;
            for (size_t i = 0; i < D + 1; ++i) {
                nearestPoints.push_back(distances[i].second);
            }

            return nearestPoints;
        }

        Simplex createSimplexFromPoints(const std::vector<TrainingPt>& points) const {
            Simplex simplex;
            for (size_t i = 0; i < D + 1; ++i) {
                for (size_t j = 0; j < D; ++j) {
                    simplex.vert[i][j] = points[i][j];
                }
            }
            return simplex;
        }

    private:
        // Compute barycentric weights for x in simplex s
        std::array<Real, D + 1> barycentricWeights(const PointND& x, const Simplex& s) const {
            std::array<std::array<Real, D>, D> M{};
            std::array<Real, D> rhs{};
            const auto& v0 = s.vert[0];
            for (size_t i = 0; i < D; ++i) {
                rhs[i] = x[i] - v0[i];
                for (size_t j = 0; j < D; ++j)
                    M[i][j] = s.vert[j + 1][i] - v0[i];
            }
            auto wv = InterpolatorCommonMD::solveLinear<D, Real>(M, rhs);
            std::array<Real, D + 1> w;
            Real sum = Real(0);
            for (size_t i = 0; i < D; ++i) {
                w[i + 1] = wv[i];
                sum += wv[i];
            }
            w[0] = Real(1) - sum;
            return w;
        }

        // Apply weights to compute value
        Real applyWeights(const std::array<Real, D + 1>& w, const Simplex& s) const {
            Real val = Real(0);
            for (size_t i = 0; i <= D; ++i) {
                std::array<Real, D> coord;
                for (size_t j = 0; j < D; ++j)
                    coord[j] = s.vert[i][j];
                val += w[i] * findZ(coord);
            }
            return val;
        }

        // Find z-value matching coordinate (exact match within tol)
        Real findZ(const std::array<Real, D>& coord) const {
            const Real tol = Real(1e-14);
            for (auto& tp : m_trainPoints) {
                bool match = true;
                for (size_t i = 0; i < D; ++i) {
                    if (std::fabs(tp[i] - coord[i]) > tol) {
                        match = false;
                        break;
                    }
                }
                if (match)
                    return tp[D];
            }
            throw std::runtime_error("Missing z-value for simplex vertex");
        }

        std::vector<TrainingPt> m_trainPoints;
        std::vector<Simplex>    m_simplices;
    };

} // namespace DelauneInterpolationMD

#endif // DELAUNE_INTERPOLATION_MD_H