#ifndef DELAUNE_BW_MD_H
#define DELAUNE_BW_MD_H
#pragma once

#include "InterpolatorCommonMD.h"
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <map>
#include <queue>
#include <cassert>

/*
 * Implement Delaunay triangulation using the Bowyer-Watson algorithm with local conflict walking.
 *
 * Complexity:
 * - Triangulation: O(n log n) on average due to local conflict walking, O(n²) in the worst case.
 *
 */

namespace DelauneUtils {

    // Facet — (D-1)-dimensional face
    template <size_t D, typename Real = double>
    using Facet = std::array<InterpolatorCommonMD::PointND<D, Real>, D>;

    template <size_t D, typename Real>
    bool pointLess(const InterpolatorCommonMD::PointND<D, Real>& a,
        const InterpolatorCommonMD::PointND<D, Real>& b) noexcept {
        for (size_t i = 0; i < D; ++i) {
            if (a[i] < b[i]) return true;
            if (b[i] < a[i]) return false;
        }
        return false;
    }

    // Normalize facet
    template <size_t D, typename Real>
    Facet<D, Real> normalizeFacet(Facet<D, Real> f) noexcept {
        std::sort(f.begin(), f.end(), pointLess<D, Real>);
        return f;
    }

    // Node: simplex + neighbor indices
    template <size_t D, typename Real>
    struct Node {
        InterpolatorCommonMD::Simplex<D, Real> s;
        std::vector<size_t> neighbors;
    };

    // Compute circumsphere
    template <size_t D, typename Real>
    std::pair<InterpolatorCommonMD::PointND<D, Real>, Real>
        circumsphere(const InterpolatorCommonMD::Simplex<D, Real>& sp) {
        std::array<std::array<Real, D>, D> A{};
        std::array<Real, D> b{};
        auto& x0 = sp.vert[0];
        Real sq0 = Real(0);
        for (size_t i = 0; i < D; ++i) sq0 += x0[i] * x0[i];
        for (size_t i = 0; i < D; ++i) {
            auto& xi = sp.vert[i + 1];
            Real sqi = Real(0);
            for (size_t j = 0; j < D; ++j) {
                A[i][j] = xi[j] - x0[j];
                sqi += xi[j] * xi[j];
            }
            b[i] = Real(0.5) * (sqi - sq0);
        }
        auto center = InterpolatorCommonMD::solveLinear<D, Real>(A, b);
        Real r2 = Real(0);
        for (size_t i = 0; i < D; ++i) {
            Real d = x0[i] - center[i]; r2 += d * d;
        }

        return { center, std::sqrt(r2) };
    }

    // Point in sphere
    template <size_t D, typename Real>
    bool pointInSphere(const InterpolatorCommonMD::PointND<D, Real>& p,
        const InterpolatorCommonMD::Simplex<D, Real>& sp,
        Real eps = Real(1e-12)) {
        auto [c, r] = circumsphere<D, Real>(sp);
        Real d2 = Real(0);
        for (size_t i = 0; i < D; ++i) {
            Real diff = p[i] - c[i]; d2 += diff * diff;
        }
        bool inside = d2 <= r * r - eps;

        return inside;
    }

    // Bowyer-Watson with local conflict walking
    template <size_t D, typename Real = double>
    std::vector<InterpolatorCommonMD::Simplex<D, Real>>
        bowyerWatsonND(const std::vector<InterpolatorCommonMD::PointND<D, Real>>& points) {
        using Simplex = InterpolatorCommonMD::Simplex<D, Real>;

        // Compute bounding super-simplex
        auto minP = points.front(), maxP = points.front();
        for (auto& p : points)
            for (size_t i = 0; i < D; ++i) {
                minP[i] = std::min(minP[i], p[i]);
                maxP[i] = std::max(maxP[i], p[i]);
            }
        InterpolatorCommonMD::PointND<D, Real> mid{};
        Real delta = Real(0);
        for (size_t i = 0; i < D; ++i) {
            mid[i] = (minP[i] + maxP[i]) / Real(2);
            delta = std::max(delta, maxP[i] - minP[i]);
        }
        Real factor = delta * Real(3);
        Simplex superS;
        for (size_t k = 0; k < D; ++k) {
            superS.vert[k] = mid;
            superS.vert[k][k] += factor;
        }
        superS.vert[D] = mid;
        for (size_t i = 0; i < D; ++i) superS.vert[D][i] -= factor;

        // T: nodes (simplex + neighbors)
        std::vector<Node<D, Real>> T;
        T.reserve(points.size() * 4);
        T.push_back({ superS,{} });

        // facet->node map for adjacency
        std::multimap<Facet<D, Real>, size_t> facetMap;

        // Insert each point
        for (size_t pi = 0; pi < points.size(); ++pi) {
            const auto& p = points[pi];
            size_t nT = T.size();

            // 1) find conflict region via BFS
            std::vector<bool> visited(nT), removed(nT);
            std::vector<size_t> bad;
            std::queue<size_t> q;
            q.push(0); // start with super-simplex
            while (!q.empty()) {
                size_t idx = q.front(); q.pop();
                if (visited[idx] || removed[idx]) continue;
                visited[idx] = true;
                if (pointInSphere<D, Real>(p, T[idx].s)) {
                    bad.push_back(idx);
                    for (auto nb : T[idx].neighbors)
                        if (!visited[nb]) q.push(nb);
                }
            }

            if (bad.empty()) continue;

            // 2) boundary facets: count occurrences
            facetMap.clear();
            for (auto idx : bad) {
                const auto& s = T[idx].s;
                for (size_t i = 0; i <= D; ++i) {
                    Facet<D, Real> f;
                    for (size_t j = 0, k = 0; j <= D; ++j) {
                        if (j == i) continue;
                        f[k++] = s.vert[j];
                    }
                    facetMap.emplace(normalizeFacet<D, Real>(f), idx);
                }
            }
            std::vector<Facet<D, Real>> boundary;
            for (auto it = facetMap.begin(); it != facetMap.end(); ++it) {
                auto range = facetMap.equal_range(it->first);
                size_t count = std::distance(range.first, range.second);
                if (count == 1) boundary.push_back(it->first);
            }

            // 3) mark removed
            for (auto idx : bad) removed[idx] = true;

            // 4) build new simplices
            std::vector<Node<D, Real>> newNodes;
            for (auto& f : boundary) {
                Simplex ns;
                for (size_t i = 0; i < D; ++i) ns.vert[i] = f[i];
                ns.vert[D] = p;
                newNodes.push_back({ ns,{} });
            }

            // 5) rebuild T
            std::vector<Node<D, Real>> T2;
            T2.reserve(T.size() - bad.size() + newNodes.size());
            for (size_t i = 0; i < T.size(); ++i)
                if (!removed[i]) T2.push_back(std::move(T[i]));
            for (auto& nn : newNodes) T2.push_back(std::move(nn));
            T.swap(T2);

            // 6) rebuild adjacency
            facetMap.clear();
            for (size_t ti = 0; ti < T.size(); ++ti) {
                const auto& s = T[ti].s;
                for (size_t i = 0; i <= D; ++i) {
                    Facet<D, Real> f;
                    for (size_t j = 0, k = 0; j <= D; ++j) {
                        if (j == i) continue;
                        f[k++] = s.vert[j];
                    }
                    facetMap.emplace(normalizeFacet<D, Real>(f), ti);
                }
            }
            for (auto& nnode : T) nnode.neighbors.clear();
            for (auto& pr : facetMap) {
                auto range = facetMap.equal_range(pr.first);
                auto first = range.first->second;
                for (auto it = std::next(range.first); it != range.second; ++it) {
                    auto second = it->second;
                    T[first].neighbors.push_back(second);
                    T[second].neighbors.push_back(first);
                }
            }
        }

        // remove any containing superS vertices
        std::vector<Simplex> res;
        for (auto& node : T) {
            bool has = false;
            for (auto& v : node.s.vert)
                for (auto& sv : superS.vert)
                    if (v == sv) has = true;
            if (!has) res.push_back(node.s);
        }
        return res;
    }

} // namespace DelauneUtils

#endif // DELAUNE_BW_MD_H