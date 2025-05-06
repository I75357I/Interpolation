#include "DelauneInterpolation.h"
#include "Delaune_BW.h"
#include <cmath>
#include <stdexcept>
#include <limits>
#include <algorithm>

DelauneInterpolation3D::DelauneInterpolation3D(const std::vector<Point3D>& trainPoints)
    : m_trainPoints(trainPoints)
{
    if (!InterpolatorCommon3D::checkMinimumPoints(m_trainPoints, 3)) {
        throw std::invalid_argument("At least 3 points are required for interpolation.");
    }
    InterpolatorCommon3D::validateUniquePoints(m_trainPoints);
    m_triangles = DelauneUtils::bowyerWatson(m_trainPoints);
}

DelauneInterpolation3D::Real DelauneInterpolation3D::interpolate(Real x, Real y) const {
    for (const auto& tri : m_triangles) {
        return barycentricInterpolate(x, y, tri);
    }
}

std::vector<DelauneInterpolation3D::Point3D> DelauneInterpolation3D::predict(const std::vector<Point3D>& testPoints) const {
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

DelauneInterpolation3D::Real DelauneInterpolation3D::barycentricInterpolate(Real x, Real y, const Triangle& T) const {
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

DelauneInterpolation3D::Real DelauneInterpolation3D::findZ(const Point2D& pt) const {
    for (const auto& p : m_trainPoints) {
        if (std::fabs(std::get<0>(p) - pt.first) < 1e-14 &&
            std::fabs(std::get<1>(p) - pt.second) < 1e-14)
        {
            return std::get<2>(p);
        }
    }
    throw std::runtime_error("Corresponding z-value for a triangle vertex not found.");
}
