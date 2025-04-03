#ifndef DELAUNE_INTERPOLATION_H
#define DELAUNE_INTERPOLATION_H

#include "InterpolatorCommon3D.h"
#include "Delaune.h"
#include <vector>
#include <stdexcept>
#include <limits>

/*
 * DelauneInterpolation3D
 * -----------------------
 * Implements interpolation using a Delaune triangulation and barycentric coordinates.
 * If a test point does not fall into any triangle, the interpolated value is set to the
 * z value of the nearest training point (nearest neighbor).
 *
 * References:
 * - "Computational Geometry: Algorithms and Applications", de Berg et al.
 * - Bowyer (1981), Watson (1981).
 *
 * Complexity:
 * - Triangulation: average O(n log n) (worst-case O(n^2)).
 * - Interpolating a point: linear in the number of triangles or training points for nearest neighbor.
 *
 * Input validations (duplicate points, minimum number of points) are performed.
 */

class DelauneInterpolation3D {
public:
    using Real = ::Real;
    using Point3D = ::Point3D;
    using Triangle = ::Triangle;

    // Constructor takes training (x,y,z) points.
    explicit DelauneInterpolation3D(const std::vector<Point3D>& trainPoints);

    // Interpolates the z value for the given (x,y) point.
    // If the point is outside the triangulated region, returns the nearest neighbor's z.
    Real interpolate(Real x, Real y) const;

    // Batch interpolation: returns a vector of (x,y,zInterpolated) for test points.
    std::vector<Point3D> predict(const std::vector<Point3D>& testPoints) const;

private:
    std::vector<Point3D> m_trainPoints;
    std::vector<Triangle> m_triangles;

    // Performs barycentric interpolation within a triangle.
    Real barycentricInterpolate(Real x, Real y, const Triangle& T) const;

    // Finds the z value corresponding to a given 2D point from training data.
    Real findZ(const Point2D& pt) const;

    // Returns the z value of the nearest training point to (x, y)
    Real nearestNeighbor(Real x, Real y) const;
};

#endif // DELAUNE_INTERPOLATION_H