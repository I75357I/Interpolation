#include "NearestTriangularInterpolation3D.h"
#include <iostream>
#include <vector>
#include <tuple>

int main() {
    using Point3D = std::tuple<double, double, double>;

    std::vector<Point3D> trainingData = {
        {0.0, 0.0, 10.0},
        {1.0, 0.0, 20.0},
        {0.0, 1.0, 30.0},
        {1.0, 1.0, 40.0},
        {0.5, 0.5, 25.0}
    };

    NearestTriangularInterpolation3D nearestInterp(trainingData);

    std::vector<Point3D> testPoints = {
        {0.25, 0.25, 0.0},
        {0.75, 0.75, 0.0},
        {1.2, 1.2, 0.0}
    };

    auto interpolatedResults = nearestInterp.predict(testPoints);

    std::cout << "NearestTriangularInterpolation3D Example:\n";
    for (const auto& pt : interpolatedResults) {
        double x, y, z;
        std::tie(x, y, z) = pt;
        std::cout << "Interpolated z value at (" << x << ", " << y << "): " << z << "\n";
    }

    return 0;
}