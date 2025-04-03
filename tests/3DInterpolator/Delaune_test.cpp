#include "src/3D Interpolation/DelauneInterpolation.h"
#include <iostream>
#include <vector>
#include <tuple>
#include <limits>

int main() {
    using Point3D = std::tuple<double, double, double>;

    std::vector<Point3D> trainingData = {
        {0.0, 0.0, 10.0},
        {1.0, 0.0, 20.0},
        {0.0, 1.0, 30.0},
        {1.0, 1.0, 40.0}
    };

    DelauneInterpolation3D delauneInterp(trainingData);

    std::vector<Point3D> testPoints = {
        {0.25, 0.25, 0.0},
        {0.75, 0.75, 0.0},
        {1.5, 1.5, 0.0}
    };

    auto interpolatedResults = delauneInterp.predict(testPoints);

    std::cout << "DelauneInterpolation3D Example:\n";
    for (const auto& pt : interpolatedResults) {
        double x, y, z;
        std::tie(x, y, z) = pt;
        std::cout << "Interpolated z value at (" << x << ", " << y << "): " << z << "\n";
    }

    return 0;
}
