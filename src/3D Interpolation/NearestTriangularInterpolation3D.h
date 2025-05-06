#ifndef NEAREST_TRIANGULAR_INTERPOLATION_3D_H
#define NEAREST_TRIANGULAR_INTERPOLATION_3D_H

#include "InterpolatorCommon3D.h"
#include <vector>
#include <stdexcept>
#include <limits>
#include <array>

/*
 * NearestTriangularInterpolation3D
 * ----------------------------------
 * Implements interpolation by selecting the 3 nearest (x,y) training points and using
 * barycentric interpolation.
 *
 * Complexity:
 * - For each query, the search for the 3 nearest neighbors is O(n).
 *
 * Input validations (duplicate points, minimum number of points) are performed.
 */

class NearestTriangularInterpolation3D {
public:
    using Real = ::Real;
    using Point3D = ::Point3D;
    using Point2D = ::Point2D;

    struct Triangle {
        std::array<Point2D, 3> p;
    };

    // Constructor takes training (x,y,z) points.
    explicit NearestTriangularInterpolation3D(const std::vector<Point3D>& trainPoints);

    // Interpolates the z value for the given (x,y) point using the 3 nearest training points.
    // If barycentric interpolation fails, returns the z value of the nearest training point.
    Real interpolate(Real x, Real y) const;

    // Batch interpolation: returns a vector of (x,y,zInterpolated) for test points.
    std::vector<Point3D> predict(const std::vector<Point3D>& testPoints) const;

private:
    std::vector<Point3D> m_trainPoints;

    // Performs barycentric interpolation for the triangle formed by the 3 nearest points.
    Real barycentricInterpolate(Real x, Real y, const Triangle& T) const;

    // Finds the z value corresponding to a given 2D point from training data.
    Real findZ(const Point2D& pt) const;

};

#endif // NEAREST_TRIANGULAR_INTERPOLATION_3D_H