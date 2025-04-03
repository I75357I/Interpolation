#include <iostream>
#include <cassert>
#include "src/2D Interpolation/CubicSplineInterpolator.h"
#include "src/2D Interpolation/InterpolatorCommon.h"

using namespace interpolation;

int main() {
    try {
        std::vector<Point2D> train = {
            {0.0, 0.0},
            {1.0, 2.0},
            {2.0, 3.0},
            {3.0, 5.0}
        };
        std::vector<Point2D> testPoints = {
            {0.5, 0.0},
            {1.5, 0.0},
            {2.5, 0.0}
        };
        CubicSplineInterpolator csInterpolator(train);
        auto csResults = csInterpolator.predict(testPoints);
        std::cout << "Cubic Spline Interpolation:\n";
        for (const auto& pt : csResults) {
            std::cout << "x: " << pt.first << ", y: " << pt.second << "\n";
        }
        assert(csResults.size() == testPoints.size());
        std::cout << "\nAll tests passed.\n";
    }
    catch (const std::exception& e) {
        std::cerr << "Test failed: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
