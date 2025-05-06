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
    Real bestD1 = std::numeric_limits<Real>::infinity();
    Real bestD2 = std::numeric_limits<Real>::infinity();
    Real bestD3 = std::numeric_limits<Real>::infinity();
    size_t i1 = 0, i2 = 0, i3 = 0;

    for (size_t i = 0; i < m_trainPoints.size(); ++i) {
        Real dx = std::get<0>(m_trainPoints[i]) - x;
        Real dy = std::get<1>(m_trainPoints[i]) - y;
        Real d = std::hypot(dx, dy);

        if (d < bestD1) {
            bestD3 = bestD2;  i3 = i2;
            bestD2 = bestD1;  i2 = i1;
            bestD1 = d;       i1 = i;
        }
        else if (d < bestD2) {
            bestD3 = bestD2;  i3 = i2;
            bestD2 = d;       i2 = i;
        }
        else if (d < bestD3) {
            bestD3 = d;       i3 = i;
        }
    }

    Triangle T;
    const auto& P1 = m_trainPoints[i1];
    const auto& P2 = m_trainPoints[i2];
    const auto& P3 = m_trainPoints[i3];
    T.p[0] = { std::get<0>(P1), std::get<1>(P1) };
    T.p[1] = { std::get<0>(P2), std::get<1>(P2) };
    T.p[2] = { std::get<0>(P3), std::get<1>(P3) };

    return barycentricInterpolate(x, y, T);
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
