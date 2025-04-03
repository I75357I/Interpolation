#ifndef INTERPOLATOR_COMMON_3D_H
#define INTERPOLATOR_COMMON_3D_H

#include <vector>
#include <tuple>
#include <stdexcept>
#include <cmath>
#include <algorithm>

// Common type definitions
using Real = double;
using Point2D = std::pair<Real, Real>;
using Point3D = std::tuple<Real, Real, Real>;

namespace InterpolatorCommon3D {

    // Check that the input vector has at least minPoints elements.
    inline bool checkMinimumPoints(const std::vector<Point3D>& points, size_t minPoints = 3) {
        return points.size() >= minPoints;
    }

    // Validate that there are no duplicate (x,y) points in the dataset.
    inline void validateUniquePoints(const std::vector<Point3D>& points, double tolerance = 1e-14) {
        for (size_t i = 0; i < points.size(); ++i) {
            for (size_t j = i + 1; j < points.size(); ++j) {
                if (std::fabs(std::get<0>(points[i]) - std::get<0>(points[j])) < tolerance &&
                    std::fabs(std::get<1>(points[i]) - std::get<1>(points[j])) < tolerance)
                {
                    throw std::invalid_argument("Duplicate (x,y) values found in input points.");
                }
            }
        }
    }

    // Check if a test point is within the bounding box of training points.
    inline bool isPointWithinBoundingBox(const Point3D& testPt, const std::vector<Point3D>& trainPts, double tolerance = 1e-14) {
        if (trainPts.empty()) return false;
        Real min_x = std::get<0>(trainPts[0]);
        Real max_x = min_x;
        Real min_y = std::get<1>(trainPts[0]);
        Real max_y = min_y;
        for (const auto& pt : trainPts) {
            Real x = std::get<0>(pt);
            Real y = std::get<1>(pt);
            min_x = std::min(min_x, x);
            max_x = std::max(max_x, x);
            min_y = std::min(min_y, y);
            max_y = std::max(max_y, y);
        }
        Real x = std::get<0>(testPt);
        Real y = std::get<1>(testPt);
        return (x >= min_x - tolerance && x <= max_x + tolerance &&
            y >= min_y - tolerance && y <= max_y + tolerance);
    }

}

#endif // INTERPOLATOR_COMMON_3D_H