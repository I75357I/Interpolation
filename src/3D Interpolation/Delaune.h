#ifndef DELAUNE_H
#define DELAUNE_H

#include "InterpolatorCommon3D.h"
#include <vector>
#include <tuple>
#include <array>

// References:
// - "Computational Geometry: Algorithms and Applications", de Berg et al.
// - Bowyer (1981), Watson (1981).

// Structure for a 2D triangle (using only x,y)
struct Triangle {
    std::array<Point2D, 3> p;
};

// Structure for a 2D edge
struct Edge {
    Point2D p1, p2;
};

namespace DelauneUtils {

    // Calculates the absolute area of a triangle defined by points A, B, and C.
    Real area(const Point2D& A, const Point2D& B, const Point2D& C);

    // Checks if point (x, y) lies inside triangle T using barycentric coordinates.
    bool pointInTriangle(Real x, Real y, const Triangle& T, Real epsilon = 1e-12);

    // Computes the circumcircle (center and radius) for points A, B, and C.
    std::tuple<Real, Real, Real> circumcircle(const Point2D& A, const Point2D& B, const Point2D& C);

    // Checks if a 3D point P (using only x,y) lies inside the circumcircle of triangle T.
    bool pointInCircumcircle(const Point3D& P, const Triangle& T);

    // Lexicographic comparison of two 2D points.
    bool pointLess(const Point2D& A, const Point2D& B);

    // Normalizes an edge so that the lesser point comes first.
    Edge normalizeEdge(const Point2D& A, const Point2D& B);

    // Constructs a Delaune triangulation using the Bowyer–Watson algorithm.
    std::vector<Triangle> bowyerWatson(const std::vector<Point3D>& points);
}

#endif // DELAUNE_H