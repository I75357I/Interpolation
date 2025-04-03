#include "NearestTriangularInterpolation3D.h"
#include "InterpolatorCommon3D.h"
#include <cmath>
#include <stdexcept>
#include <numeric>
#include <limits>
#include <algorithm>

NearestTriangularInterpolation3D::NearestTriangularInterpolation3D(const std::vector<Point3D>& trainPoints)
    : m_trainPoints(trainPoints)
{
    if (!InterpolatorCommon3D::checkMinimumPoints(m_trainPoints, 3)) {
        throw std::invalid_argument("At least 3 points are required for interpolation.");
    }
    InterpolatorCommon3D::validateUniquePoints(m_trainPoints);
}

NearestTriangularInterpolation3D::Real NearestTriangularInterpolation3D::interpolate(Real x, Real y) const {
    std::vector<size_t> indices(m_trainPoints.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
        [this, x, y](size_t a, size_t b) {
            Real dx1 = std::get<0>(m_trainPoints[a]) - x;
            Real dy1 = std::get<1>(m_trainPoints[a]) - y;
            Real dx2 = std::get<0>(m_trainPoints[b]) - x;
            Real dy2 = std::get<1>(m_trainPoints[b]) - y;
            return std::hypot(dx1, dy1) < std::hypot(dx2, dy2);
        });

    size_t k = std::min<size_t>(10, m_trainPoints.size());
    std::vector<size_t> candidates(indices.begin(), indices.begin() + k);

    Real bestArea = std::numeric_limits<Real>::max();
    bool found = false;
    Real bestZ = std::numeric_limits<Real>::quiet_NaN();

    for (size_t i = 0; i < candidates.size(); i++) {
        for (size_t j = i + 1; j < candidates.size(); j++) {
            for (size_t l = j + 1; l < candidates.size(); l++) {
                Triangle tri;
                tri.p[0] = { std::get<0>(m_trainPoints[candidates[i]]), std::get<1>(m_trainPoints[candidates[i]]) };
                tri.p[1] = { std::get<0>(m_trainPoints[candidates[j]]), std::get<1>(m_trainPoints[candidates[j]]) };
                tri.p[2] = { std::get<0>(m_trainPoints[candidates[l]]), std::get<1>(m_trainPoints[candidates[l]]) };

                try {
                    Real z = barycentricInterpolate(x, y, tri);

                    const auto& A = tri.p[0];
                    const auto& B = tri.p[1];
                    const auto& C = tri.p[2];
                    Real area = 0.5 * std::fabs((B.first - A.first) * (C.second - A.second) -
                        (C.first - A.first) * (B.second - A.second));

                    if (area < bestArea) {
                        bestArea = area;
                        bestZ = z;
                        found = true;
                    }
                }
                catch (const std::exception&) {

                }
            }
        }
    }

    if (found) {
        return bestZ;
    }
    else {
        return nearestNeighbor(x, y);
    }
}


std::vector<NearestTriangularInterpolation3D::Point3D>
NearestTriangularInterpolation3D::predict(const std::vector<Point3D>& testPoints) const {
    std::vector<Point3D> output;
    output.reserve(testPoints.size());
    for (const auto& pt : testPoints) {
        Real x = std::get<0>(pt);
        Real y = std::get<1>(pt);
        Real z = interpolate(x, y);
        output.emplace_back(x, y, z);
    }
    return output;
}

NearestTriangularInterpolation3D::Real NearestTriangularInterpolation3D::barycentricInterpolate(Real x, Real y, const Triangle& T) const {
    const auto& A = T.p[0];
    const auto& B = T.p[1];
    const auto& C = T.p[2];

    Real zA = findZ(A);
    Real zB = findZ(B);
    Real zC = findZ(C);

    Real denom = (B.second - C.second) * (A.first - C.first) + (C.first - B.first) * (A.second - C.second);
    if (std::fabs(denom) < 1e-14) {
        throw std::runtime_error("Degenerate triangle encountered during interpolation.");
    }
    Real w1 = ((B.second - C.second) * (x - C.first) + (C.first - B.first) * (y - C.second)) / denom;
    Real w2 = ((C.second - A.second) * (x - C.first) + (A.first - C.first) * (y - C.second)) / denom;
    Real w3 = 1.0 - w1 - w2;

    if (w1 < -1e-12 || w1 > 1.0 + 1e-12 ||
        w2 < -1e-12 || w2 > 1.0 + 1e-12 ||
        w3 < -1e-12 || w3 > 1.0 + 1e-12)
    {
        throw std::runtime_error("Interpolation point lies outside the triangle.");
    }
    return w1 * zA + w2 * zB + w3 * zC;
}

NearestTriangularInterpolation3D::Real NearestTriangularInterpolation3D::findZ(const Point2D& pt) const {
    for (const auto& p : m_trainPoints) {
        if (std::fabs(std::get<0>(p) - pt.first) < 1e-14 &&
            std::fabs(std::get<1>(p) - pt.second) < 1e-14)
        {
            return std::get<2>(p);
        }
    }
    throw std::runtime_error("Corresponding z-value for a vertex not found.");
}

NearestTriangularInterpolation3D::Real NearestTriangularInterpolation3D::nearestNeighbor(Real x, Real y) const {
    Real minDist = std::numeric_limits<Real>::max();
    Real nearestZ = std::numeric_limits<Real>::quiet_NaN();
    for (const auto& p : m_trainPoints) {
        Real dx = std::get<0>(p) - x;
        Real dy = std::get<1>(p) - y;
        Real dist = std::hypot(dx, dy);
        if (dist < minDist) {
            minDist = dist;
            nearestZ = std::get<2>(p);
        }
    }
    return nearestZ;
}
