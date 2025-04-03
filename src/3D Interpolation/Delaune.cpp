#include "Delaune.h"
#include "InterpolatorCommon3D.h"
#include <cmath>
#include <algorithm>
#include <stdexcept>

namespace DelauneUtils {

    Real area(const Point2D& A, const Point2D& B, const Point2D& C) {
        return 0.5 * std::abs(
            A.first * (B.second - C.second) +
            B.first * (C.second - A.second) +
            C.first * (A.second - B.second)
        );
    }

    bool pointInTriangle(Real x, Real y, const Triangle& T, Real epsilon) {
        const auto& A = T.p[0];
        const auto& B = T.p[1];
        const auto& C = T.p[2];

        Real totalArea = area(A, B, C);
        Real areaPBC = area({ x, y }, B, C);
        Real areaAPC = area(A, { x, y }, C);
        Real areaABP = area(A, B, { x, y });

        return (std::fabs((areaPBC + areaAPC + areaABP) - totalArea) < epsilon);
    }

    std::tuple<Real, Real, Real> circumcircle(const Point2D& A, const Point2D& B, const Point2D& C) {
        Real d = 2.0 * (A.first * (B.second - C.second) +
            B.first * (C.second - A.second) +
            C.first * (A.second - B.second));
        if (std::fabs(d) < 1e-14) {
            return std::make_tuple(0.0, 0.0, 1e18); // degenerate: return huge radius
        }
        Real A_sq = A.first * A.first + A.second * A.second;
        Real B_sq = B.first * B.first + B.second * B.second;
        Real C_sq = C.first * C.first + C.second * C.second;
        Real ux = (A_sq * (B.second - C.second) + B_sq * (C.second - A.second) + C_sq * (A.second - B.second)) / d;
        Real uy = (A_sq * (C.first - B.first) + B_sq * (A.first - C.first) + C_sq * (B.first - A.first)) / d;
        Real r = std::sqrt((A.first - ux) * (A.first - ux) + (A.second - uy) * (A.second - uy));
        return std::make_tuple(ux, uy, r);
    }

    bool pointInCircumcircle(const Point3D& P, const Triangle& T) {
        auto [cx, cy, r] = circumcircle(T.p[0], T.p[1], T.p[2]);
        Real px = std::get<0>(P);
        Real py = std::get<1>(P);
        Real dx = px - cx;
        Real dy = py - cy;
        return std::sqrt(dx * dx + dy * dy) < r;
    }

    bool pointLess(const Point2D& A, const Point2D& B) {
        return (A.first == B.first) ? (A.second < B.second) : (A.first < B.first);
    }

    Edge normalizeEdge(const Point2D& A, const Point2D& B) {
        return pointLess(A, B) ? Edge{ A, B } : Edge{ B, A };
    }

    std::vector<Triangle> bowyerWatson(const std::vector<Point3D>& points) {
        if (!InterpolatorCommon3D::checkMinimumPoints(points, 3)) {
            throw std::runtime_error("At least 3 points are required for triangulation.");
        }

        std::vector<Point3D> pts = points; // local copy

        // Determine bounding box.
        Real min_x = std::get<0>(pts.front());
        Real max_x = min_x;
        Real min_y = std::get<1>(pts.front());
        Real max_y = min_y;
        for (const auto& p : pts) {
            min_x = std::min(min_x, std::get<0>(p));
            max_x = std::max(max_x, std::get<0>(p));
            min_y = std::min(min_y, std::get<1>(p));
            max_y = std::max(max_y, std::get<1>(p));
        }
        Real dx = max_x - min_x;
        Real dy = max_y - min_y;
        Real delta_max = std::max(dx, dy);
        Real mid_x = (min_x + max_x) / 2.0;
        Real mid_y = (min_y + max_y) / 2.0;

        // Create a super-triangle that encloses all points.
        Point2D p1{ mid_x - 2 * delta_max, mid_y - delta_max };
        Point2D p2{ mid_x, mid_y + 2 * delta_max };
        Point2D p3{ mid_x + 2 * delta_max, mid_y - delta_max };

        std::vector<Triangle> triangles;
        triangles.push_back(Triangle{ {p1, p2, p3} });

        // Insert each point into the triangulation.
        for (const auto& P : pts) {
            std::vector<Triangle> badTriangles;
            for (const auto& t : triangles) {
                if (pointInCircumcircle(P, t)) {
                    badTriangles.push_back(t);
                }
            }

            std::vector<Edge> edges;
            for (const auto& bt : badTriangles) {
                for (size_t i = 0; i < 3; ++i) {
                    edges.push_back(normalizeEdge(bt.p[i], bt.p[(i + 1) % 3]));
                }
            }

            // Remove bad triangles.
            triangles.erase(
                std::remove_if(triangles.begin(), triangles.end(),
                    [&](const Triangle& t) {
                        for (const auto& bt : badTriangles) {
                            int matchCount = 0;
                            for (const auto& tp : t.p) {
                                for (const auto& btp : bt.p) {
                                    if (std::fabs(tp.first - btp.first) < 1e-14 &&
                                        std::fabs(tp.second - btp.second) < 1e-14)
                                    {
                                        ++matchCount;
                                    }
                                }
                            }
                            if (matchCount == 3) return true;
                        }
                        return false;
                    }
                ), triangles.end());

            // Remove duplicate edges.
            std::sort(edges.begin(), edges.end(), [](const Edge& A, const Edge& B) {
                if (A.p1.first != B.p1.first) return A.p1.first < B.p1.first;
                if (A.p1.second != B.p1.second) return A.p1.second < B.p1.second;
                if (A.p2.first != B.p2.first) return A.p2.first < B.p2.first;
                return A.p2.second < B.p2.second;
                });

            std::vector<Edge> polygonEdges;
            for (size_t i = 0; i < edges.size();) {
                if (i + 1 < edges.size() &&
                    std::fabs(edges[i].p1.first - edges[i + 1].p1.first) < 1e-14 &&
                    std::fabs(edges[i].p1.second - edges[i + 1].p1.second) < 1e-14 &&
                    std::fabs(edges[i].p2.first - edges[i + 1].p2.first) < 1e-14 &&
                    std::fabs(edges[i].p2.second - edges[i + 1].p2.second) < 1e-14)
                {
                    i += 2; // skip duplicates
                }
                else {
                    polygonEdges.push_back(edges[i]);
                    ++i;
                }
            }

            Real px = std::get<0>(P);
            Real py = std::get<1>(P);
            for (const auto& e : polygonEdges) {
                triangles.push_back(Triangle{ { e.p1, e.p2, {px, py} } });
            }
        }

        // Remove triangles that share vertices with the super-triangle.
        std::vector<Triangle> finalTriangles;
        for (const auto& t : triangles) {
            bool skip = false;
            for (const auto& vertex : t.p) {
                if ((std::fabs(vertex.first - p1.first) < 1e-14 && std::fabs(vertex.second - p1.second) < 1e-14) ||
                    (std::fabs(vertex.first - p2.first) < 1e-14 && std::fabs(vertex.second - p2.second) < 1e-14) ||
                    (std::fabs(vertex.first - p3.first) < 1e-14 && std::fabs(vertex.second - p3.second) < 1e-14))
                {
                    skip = true;
                    break;
                }
            }
            if (!skip) {
                finalTriangles.push_back(t);
            }
        }

        return finalTriangles;
    }

} // namespace DelauneUtils